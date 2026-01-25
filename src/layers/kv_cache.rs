use anyhow::Result;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg,
};
use half::bf16;
use std::sync::Arc;

use crate::config::ModelConfig;
use crate::ptx::KV_CACHE_PTX;
use crate::utils::load_cuda_funtion;

pub struct KVCache {
    pub k_pool: CudaSlice<bf16>,
    pub v_pool: CudaSlice<bf16>,
    pub block_table: CudaSlice<i32>,
    pub block_table_cpu: Vec<i32>,
    pub free_blocks: Vec<i32>,
    pub block_size: usize,
    pub num_physical_blocks: usize,
    pub max_seq_len: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub cuda_function: CudaFunction,
}

impl KVCache {
    pub fn new(
        context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        config: &ModelConfig,
        memory_fraction: f32,
    ) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let block_size = 16;

        let (free, total) = cudarc::driver::result::mem_get_info()?;
        let used = total - free;
        let available_total = (total as f32 * memory_fraction) as usize;
        let Some(allocatable_bytes) = available_total.checked_sub(used) else {
            anyhow::bail!(
                "Not enough available GPU memory for KV Cache allocation, free: {} GB, used: {} GB, total: {} GB, requested fraction: {}",
                free as f32 / 1024.0 / 1024.0 / 1024.0,
                used as f32 / 1024.0 / 1024.0 / 1024.0,
                total as f32 / 1024.0 / 1024.0 / 1024.0,
                memory_fraction
            );
        };

        let bytes_per_block = 2
            * config.num_hidden_layers
            * config.num_key_value_heads
            * block_size
            * head_dim
            * std::mem::size_of::<bf16>();

        let available_physical_blocks = allocatable_bytes / bytes_per_block;
        let num_physical_blocks = available_physical_blocks
            .min((config.max_position_embeddings + block_size - 1) / block_size);

        let pool_size = num_physical_blocks
            * config.num_hidden_layers
            * config.num_key_value_heads
            * block_size
            * head_dim;

        let k_pool = stream.alloc_zeros::<bf16>(pool_size)?;
        let v_pool = stream.alloc_zeros::<bf16>(pool_size)?;

        let max_logical_blocks = (config.max_position_embeddings + block_size - 1) / block_size;
        let block_table = stream.alloc_zeros::<i32>(max_logical_blocks)?;

        let mut free_blocks = Vec::with_capacity(num_physical_blocks);
        for i in (0..num_physical_blocks).rev() {
            free_blocks.push(i as i32);
        }

        let max_tokens = num_physical_blocks * block_size;
        println!(
            "KVCache Initialized: VRAM Budget = {:.2} GB, Pool Capacity = {} tokens ({} blocks), Logical Max = {} tokens, Pool VRAM = {:.2} GB",
            allocatable_bytes as f32 / 1024.0 / 1024.0 / 1024.0,
            max_tokens,
            num_physical_blocks,
            config.max_position_embeddings,
            (pool_size * std::mem::size_of::<bf16>()) as f32 / 1024.0 / 1024.0 / 1024.0
        );

        let cuda_function = load_cuda_funtion(context, KV_CACHE_PTX, "update_kv_cache_kernel")?;

        Ok(KVCache {
            k_pool,
            v_pool,
            block_table,
            block_table_cpu: Vec::new(),
            free_blocks,
            block_size,
            num_physical_blocks,
            max_seq_len: config.max_position_embeddings,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            num_layers: config.num_hidden_layers,
            cuda_function,
        })
    }

    pub fn prepare_for_step(&mut self, stream: &Arc<CudaStream>, pos: usize) -> Result<()> {
        let logical_block_idx = pos / self.block_size;

        if logical_block_idx >= self.block_table_cpu.len() {
            let new_block = self
                .free_blocks
                .pop()
                .expect("KVCache: Out of memory blocks");
            self.block_table_cpu.push(new_block);

            stream.memcpy_htod(
                &self.block_table_cpu,
                &mut self.block_table.slice_mut(0..self.block_table_cpu.len()),
            )?;
        }
        Ok(())
    }

    pub fn update(
        &mut self,
        stream: &Arc<CudaStream>,
        layer_idx: usize,
        pos: usize,
        pos_ptr: &CudaSlice<i32>,
        k_input: &CudaView<'_, bf16>,
        k_offset: usize,
        v_input: &CudaView<'_, bf16>,
        v_offset: usize,
        v_bias: &CudaView<'_, bf16>,
    ) -> Result<()> {
        self.prepare_for_step(stream, pos)?;

        let cfg = LaunchConfig {
            grid_dim: (self.num_kv_heads as u32, 1, 1),
            block_dim: (self.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let k_view = k_input.slice(k_offset..k_offset + (self.num_kv_heads * self.head_dim));
        let v_view = v_input.slice(v_offset..v_offset + (self.num_kv_heads * self.head_dim));

        let mut builder = stream.launch_builder(&self.cuda_function);
        let layer_idx = layer_idx as i32;
        // let pos = pos as i32; // Removed
        let num_layers = self.num_layers as i32;
        let num_kv_heads = self.num_kv_heads as i32;
        let max_seq_len = self.max_seq_len as i32;
        let head_dim = self.head_dim as i32;
        let block_size = self.block_size as i32;

        builder
            .arg(&self.k_pool)
            .arg(&self.v_pool)
            .arg(&k_view)
            .arg(&v_view)
            .arg(v_bias)
            .arg(&self.block_table)
            .arg(&layer_idx)
            .arg(pos_ptr)
            .arg(&num_layers)
            .arg(&num_kv_heads)
            .arg(&max_seq_len)
            .arg(&head_dim)
            .arg(&block_size);

        unsafe { builder.launch(cfg) }?;
        Ok(())
    }
}
