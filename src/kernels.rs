use std::sync::Arc;

use anyhow::Result;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr,
    DevicePtrMut, LaunchConfig, PushKernelArg,
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
    pub(crate) sampling: CudaFunction,
    pub(crate) top_k_filter: CudaFunction,
    pub(crate) fused_top_p_sample_32: CudaFunction,
    pub(crate) fused_top_p_sample_64: CudaFunction,
    pub(crate) fused_top_p_sample_128: CudaFunction,
    pub(crate) fused_top_p_sample_256: CudaFunction,
    pub(crate) fused_top_p_sample_512: CudaFunction,
    pub(crate) fused_top_p_sample_1024: CudaFunction,
}

impl CudaFunctions {
    pub(crate) fn load(context: &Arc<CudaContext>, head_dim: usize) -> Result<Self> {
        println!("Loading activation kernel...");
        let activation =
            load_cuda_funtion(context, ptx::ACTIVATION_PTX, "silu_and_mul_fused_kernel")?;
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
        println!("Loading sampling kernel...");
        let sampling = load_cuda_funtion(context, ptx::SAMPLING_PTX, "argmax_kernel")?;
        println!("Loading optimized sampling kernels...");
        let top_k_filter = load_cuda_funtion(context, ptx::SAMPLING_OPTIMIZED_PTX, "top_k_filter")?;
        let fused_top_p_sample_32 = load_cuda_funtion(
            context,
            ptx::SAMPLING_OPTIMIZED_PTX,
            "fused_top_p_sample_32",
        )?;
        let fused_top_p_sample_64 = load_cuda_funtion(
            context,
            ptx::SAMPLING_OPTIMIZED_PTX,
            "fused_top_p_sample_64",
        )?;
        let fused_top_p_sample_128 = load_cuda_funtion(
            context,
            ptx::SAMPLING_OPTIMIZED_PTX,
            "fused_top_p_sample_128",
        )?;
        let fused_top_p_sample_256 = load_cuda_funtion(
            context,
            ptx::SAMPLING_OPTIMIZED_PTX,
            "fused_top_p_sample_256",
        )?;
        let fused_top_p_sample_512 = load_cuda_funtion(
            context,
            ptx::SAMPLING_OPTIMIZED_PTX,
            "fused_top_p_sample_512",
        )?;
        let fused_top_p_sample_1024 = load_cuda_funtion(
            context,
            ptx::SAMPLING_OPTIMIZED_PTX,
            "fused_top_p_sample_1024",
        )?;

        println!("Loading formatted kernels done.");

        Ok(Self {
            activation,
            attention,
            batched_embedding,
            rmsnorm,
            rope,
            sampling,
            top_k_filter,
            fused_top_p_sample_32,
            fused_top_p_sample_64,
            fused_top_p_sample_128,
            fused_top_p_sample_256,
            fused_top_p_sample_512,
            fused_top_p_sample_1024,
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

    pub fn apply_activation_fused(
        &self,
        stream: &Arc<CudaStream>,
        gate_up_buffer: &mut CudaSlice<bf16>,
        seq_len: usize,
        intermediate_size: usize,
    ) -> Result<()> {
        let total_elements = seq_len * intermediate_size;
        let cfg = LaunchConfig::for_num_elems(total_elements as u32);

        // Kernel args: buffer, rows (seq_len), cols (intermediate_size)
        // thread_id -> index in [0, total_elements)
        let n_elements_i32 = total_elements as i32;
        let mid_stride_i32 = intermediate_size as i32;

        let ptr = gate_up_buffer.device_ptr_mut(stream).0;

        let mut build = stream.launch_builder(&self.activation);
        build.arg(&ptr).arg(&mid_stride_i32).arg(&n_elements_i32);

        unsafe {
            build.launch(cfg)?;
        }
        Ok(())
    }

    pub fn apply_rope(
        &self,
        rope: &RopeCache,
        stream: &Arc<CudaStream>,
        qkv: &mut CudaViewMut<'_, bf16>,
        q_start_idx: usize,
        k_start_idx: usize,
        q_bias: &CudaView<'_, bf16>,
        k_bias: &CudaView<'_, bf16>,
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

        let (base_ptr, _guard) = qkv.device_ptr(stream);
        // 2 bytes per bf16
        // Pointer arithmetic: we need to trust the caller that q_start_idx/k_start_idx are within qkv.
        let q_ptr = base_ptr + (q_start_idx * 2) as u64;
        let k_ptr = base_ptr + (k_start_idx * 2) as u64;

        let mut builder = stream.launch_builder(&self.rope);
        builder
            .arg(&q_ptr)
            .arg(q_bias)
            .arg(&k_ptr)
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

    pub fn apply_argmax(
        &self,
        stream: &CudaStream,
        input: &CudaSlice<bf16>,
        output_idx: &mut CudaSlice<u32>,
        vocab_size: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        let vocab_size_i32 = vocab_size as i32;
        unsafe {
            stream
                .launch_builder(&self.sampling)
                .arg(input)
                .arg(&vocab_size_i32)
                .arg(output_idx)
                .launch(cfg)?;
        };
        Ok(())
    }

    pub fn apply_sort_and_sample(
        &self,
        stream: &CudaStream,
        logits: &CudaSlice<bf16>,
        output_idx: &mut CudaSlice<u32>,
        sort_buffer: &mut CudaSlice<u8>,
        vocab_size: usize,
        top_p: f32,
        top_k: usize,
        rand_val: f32,
    ) -> Result<()> {
        let n: u32 = 1u32 << (32 - (vocab_size as u32 - 1).leading_zeros());
        let _n_i32 = n as i32;
        let vocab_size_i32 = vocab_size as i32;

        // New Logic: 2 Kernels.
        // Determine candidates based on top_k
        let target_candidates = top_k;

        // Determine filter parameters
        // To prevent shared memory overflow (CUDA_ERROR_INVALID_VALUE) at Grid=8 (for 256), we fix Grid=32.
        // vocab_size ~ 152k. Grid=32 -> Partition=4.7k items -> 19KB shared mem -> Safe.

        let filter_grid = 32;
        let filter_k_per_block = target_candidates / filter_grid;

        let total_candidates = filter_grid * filter_k_per_block;

        let cfg_filter = LaunchConfig {
            grid_dim: (filter_grid as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: ((vocab_size / filter_grid as usize) as u32 + 256) * 4, // Float size + padding
        };

        unsafe {
            stream
                .launch_builder(&self.top_k_filter)
                .arg(logits)
                .arg(&mut *sort_buffer) // Reusing sort_buffer as KeyValuePair Output
                .arg(&vocab_size_i32)
                .arg(&filter_k_per_block)
                .launch(cfg_filter)?;
        }

        // 2. Fused Sort & Sample
        // Dynamic dispatch based on total_candidates
        match total_candidates {
            32 => {
                let cfg_fused = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    stream
                        .launch_builder(&self.fused_top_p_sample_32)
                        .arg(&mut *sort_buffer)
                        .arg(&top_p)
                        .arg(&rand_val)
                        .arg(output_idx)
                        .launch(cfg_fused)?;
                }
            }
            64 => {
                let cfg_fused = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (64, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    stream
                        .launch_builder(&self.fused_top_p_sample_64)
                        .arg(&mut *sort_buffer)
                        .arg(&top_p)
                        .arg(&rand_val)
                        .arg(output_idx)
                        .launch(cfg_fused)?;
                }
            }
            128 => {
                let cfg_fused = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    stream
                        .launch_builder(&self.fused_top_p_sample_128)
                        .arg(&mut *sort_buffer)
                        .arg(&top_p)
                        .arg(&rand_val)
                        .arg(output_idx)
                        .launch(cfg_fused)?;
                }
            }
            256 => {
                let cfg_fused = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    stream
                        .launch_builder(&self.fused_top_p_sample_256)
                        .arg(&mut *sort_buffer)
                        .arg(&top_p)
                        .arg(&rand_val)
                        .arg(output_idx)
                        .launch(cfg_fused)?;
                }
            }
            512 => {
                let cfg_fused = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (512, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    stream
                        .launch_builder(&self.fused_top_p_sample_512)
                        .arg(&mut *sort_buffer)
                        .arg(&top_p)
                        .arg(&rand_val)
                        .arg(output_idx)
                        .launch(cfg_fused)?;
                }
            }
            1024 => {
                let cfg_fused = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1024, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    stream
                        .launch_builder(&self.fused_top_p_sample_1024)
                        .arg(&mut *sort_buffer)
                        .arg(&top_p)
                        .arg(&rand_val)
                        .arg(output_idx)
                        .launch(cfg_fused)?;
                }
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported candidate size: {}",
                    total_candidates
                ));
            }
        }

        Ok(())
    }
}
