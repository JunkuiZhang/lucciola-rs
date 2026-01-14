use std::sync::Arc;

use anyhow::Result;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig,
    PushKernelArg,
};
use half::bf16;

use crate::layers::kv_cache::KVCache;
use crate::layers::rope::RopeCache;
use crate::ptx;
use crate::utils::load_cuda_funtion;

pub struct CudaFunctions {
    pub(crate) activation: CudaFunction,
    pub(crate) attention: CudaFunction,
    pub(crate) batched_embedding: CudaFunction,
    pub(crate) rmsnorm: CudaFunction,
    pub(crate) rope: CudaFunction,
}

impl CudaFunctions {
    pub(crate) fn load(context: &Arc<CudaContext>, head_dim: usize) -> Result<Self> {
        println!("Loading activation kernel...");
        let activation = load_cuda_funtion(context, ptx::ACTIVATION_PTX, "silu_and_mul_kernel")?;
        println!("Loading attention kernels...");
        let attention = match head_dim {
            64 => "flash_decoding_kernel_64",
            128 => "flash_decoding_kernel_128",
            _ => {
                anyhow::bail!(
                    "Unsupported head dimension for attention kernel: {}",
                    head_dim
                )
            }
        };
        let attention = load_cuda_funtion(context, ptx::ATTENTION_PTX, attention)?;
        println!("Loading embedding kernel...");
        let batched_embedding =
            load_cuda_funtion(context, ptx::EMBEDDING_PTX, "batched_embedding_kernel")?;
        println!("Loading rmsnorm kernel...");
        let rmsnorm = load_cuda_funtion(context, ptx::RMSNORM_PTX, "rmsnorm_nvidia")?;
        println!("Loading rope kernel...");
        let rope = load_cuda_funtion(context, ptx::ROPE_PTX, "rope")?;
        println!("Loading formatted kernels done.");

        Ok(Self {
            activation,
            attention,
            batched_embedding,
            rmsnorm,
            rope,
        })
    }

    pub fn apply_batched_embedding(
        &self,
        stream: &Arc<CudaStream>,
        embedding_table: &CudaSlice<bf16>,
        input_ids: &CudaSlice<u32>,
        output_hidden_states: &mut CudaSlice<bf16>,
        hidden_dim: usize,
        seq_len: usize,
    ) -> Result<()> {
        let launch_config = LaunchConfig {
            grid_dim: (seq_len as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let hidden_dim_i = hidden_dim as i32;

        let mut build = stream.launch_builder(&self.batched_embedding);
        build
            .arg(embedding_table)
            .arg(input_ids)
            .arg(output_hidden_states)
            .arg(&hidden_dim_i);

        unsafe {
            build.launch(launch_config)?;
        }
        Ok(())
    }

    pub fn apply_rmsnorm(
        &self,
        stream: &Arc<CudaStream>,
        out: &mut CudaSlice<bf16>,
        input: Option<&CudaSlice<bf16>>,
        weight: &CudaSlice<bf16>,
        epsilon: f32,
    ) -> Result<()> {
        let num_elements = out.len();
        let num_cols = weight.len();
        let num_rows = num_elements / num_cols;
        let cfg = LaunchConfig {
            grid_dim: (num_rows as u32, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        let num_cols = num_cols as i32;

        let out_ptr = out.device_ptr_mut(stream).0;
        let input_ptr = if let Some(inp) = input {
            inp.device_ptr(stream).0
        } else {
            out_ptr
        };

        let mut build = stream.launch_builder(&self.rmsnorm);
        build
            .arg(&out_ptr)
            .arg(&input_ptr)
            .arg(weight)
            .arg(&epsilon)
            .arg(&num_cols);

        unsafe {
            build.launch(cfg)?;
        }
        Ok(())
    }

    pub fn apply_activation(
        &self,
        stream: &Arc<CudaStream>,
        out: &mut CudaSlice<bf16>,
        gate: Option<&CudaSlice<bf16>>,
        up: &CudaSlice<bf16>,
    ) -> Result<()> {
        let n = out.len() as i32;
        let cfg = LaunchConfig::for_num_elems(n as u32);

        let out_ptr = out.device_ptr_mut(stream).0;
        let gate_ptr = if let Some(g) = gate {
            g.device_ptr(stream).0
        } else {
            out_ptr
        };

        let mut build = stream.launch_builder(&self.activation);
        build.arg(&out_ptr).arg(&gate_ptr).arg(up).arg(&n);

        unsafe {
            build.launch(cfg)?;
        }
        Ok(())
    }

    pub fn apply_rope(
        &self,
        rope: &RopeCache,
        stream: &Arc<CudaStream>,
        q: &mut CudaSlice<bf16>,
        q_offset: usize,
        q_bias: &CudaSlice<bf16>,
        k: &mut CudaSlice<bf16>,
        k_offset: usize,
        k_bias: &CudaSlice<bf16>,
        pos: usize,
        head_dim: usize,
        num_q_heads: usize,
        num_k_heads: usize,
    ) -> Result<()> {
        let total_threads = num_q_heads * (head_dim / 2);
        let cfg = LaunchConfig::for_num_elems(total_threads as u32);

        let pos = pos as i32;
        let head_dim_i32 = head_dim as i32;
        let num_q_heads_i32 = num_q_heads as i32;
        let num_k_heads_i32 = num_k_heads as i32;

        // Use slice_mut. CudaViewMut implements DevicePtrMut AND DevicePtr.
        let mut q_view = q.slice_mut(q_offset..q_offset + (num_q_heads * head_dim));
        let mut k_view = k.slice_mut(k_offset..k_offset + (num_k_heads * head_dim));

        let mut builder = stream.launch_builder(&self.rope);
        builder
            .arg(&mut q_view)
            .arg(q_bias)
            .arg(&mut k_view)
            .arg(k_bias)
            .arg(&rope.cos)
            .arg(&rope.sin)
            .arg(&pos)
            .arg(&head_dim_i32)
            .arg(&num_q_heads_i32)
            .arg(&num_k_heads_i32);
        unsafe {
            builder.launch(cfg)?;
        }
        Ok(())
    }

    pub fn apply_flash_decoding(
        &self,
        stream: &Arc<CudaStream>,
        output: &mut CudaSlice<bf16>,
        out_offset: usize,
        q: &CudaSlice<bf16>,
        q_offset: usize,
        cache: &KVCache,
        layer_idx: usize,
        pos: usize,
        head_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
    ) -> Result<()> {
        let max_seq_len = cache.max_seq_len as i32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let current_num_blocks = (pos + cache.block_size) / cache.block_size;
        let block_table_bytes = current_num_blocks * 4;

        let cfg = LaunchConfig {
            grid_dim: (num_q_heads as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: (head_dim as u32) * 4 + (block_table_bytes as u32) + 128,
        };

        // Create Views inside
        let mut out_view = output.slice_mut(out_offset..out_offset + (num_q_heads * head_dim));
        let q_view = q.slice(q_offset..q_offset + (num_q_heads * head_dim));

        let mut builder = stream.launch_builder(&self.attention);

        let layer_idx = layer_idx as i32;
        let num_q_heads_i32 = num_q_heads as i32;
        let num_kv_heads_i32 = num_kv_heads as i32;
        let head_dim_i32 = head_dim as i32;
        let current_pos = pos as i32;

        let block_size_i32 = cache.block_size as i32;
        let num_layers_i32 = cache.num_layers as i32;

        builder
            .arg(&mut out_view)
            .arg(&q_view)
            .arg(&cache.k_pool)
            .arg(&cache.v_pool)
            .arg(&cache.block_table)
            .arg(&layer_idx)
            .arg(&num_q_heads_i32)
            .arg(&num_kv_heads_i32)
            .arg(&head_dim_i32)
            .arg(&max_seq_len)
            .arg(&current_pos)
            .arg(&scale)
            .arg(&block_size_i32)
            .arg(&num_layers_i32);

        unsafe {
            builder.launch(cfg)?;
        }
        Ok(())
    }
}
