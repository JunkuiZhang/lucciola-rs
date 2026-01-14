use anyhow::Result;
use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use std::sync::Arc;

use crate::config::ModelConfig;

pub struct InferenceBuffers {
    pub hidden_states: CudaSlice<bf16>,
    pub q_states: CudaSlice<bf16>,
    pub k_states: CudaSlice<bf16>,
    pub v_states: CudaSlice<bf16>,
    pub att_output: CudaSlice<bf16>,
    pub mlp_gate: CudaSlice<bf16>,
    pub mlp_up: CudaSlice<bf16>,
    pub norm_buffer: CudaSlice<bf16>,
    pub scores_buf: CudaSlice<bf16>,
}

impl InferenceBuffers {
    pub fn new(stream: &Arc<CudaStream>, config: &ModelConfig) -> Result<Self> {
        let hidden_dim = config.hidden_size;
        let head_dim = hidden_dim / config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let intermediate_size = config.intermediate_size;

        let hidden_states = stream.alloc_zeros::<bf16>(hidden_dim)?;
        let q_states = stream.alloc_zeros::<bf16>(hidden_dim)?;
        let k_states = stream.alloc_zeros::<bf16>(num_kv_heads * head_dim)?;
        let v_states = stream.alloc_zeros::<bf16>(num_kv_heads * head_dim)?;
        let att_output = stream.alloc_zeros::<bf16>(hidden_dim)?;

        let mlp_gate = stream.alloc_zeros::<bf16>(intermediate_size)?;
        let mlp_up = stream.alloc_zeros::<bf16>(intermediate_size)?;

        let norm_buffer = stream.alloc_zeros::<bf16>(hidden_dim)?;

        let num_q_heads = config.num_attention_heads;
        let group_size = num_q_heads / num_kv_heads;
        let max_scores_len = group_size * config.max_position_embeddings;
        let scores_buf = stream.alloc_zeros::<bf16>(max_scores_len)?;

        Ok(Self {
            hidden_states,
            q_states,
            k_states,
            v_states,
            att_output,
            mlp_gate,
            mlp_up,
            norm_buffer,
            scores_buf,
        })
    }
}
