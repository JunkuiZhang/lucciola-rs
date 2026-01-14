use cudarc::driver::CudaSlice;
use half::bf16;

pub struct LayerWeights {
    pub input_layernorm: CudaSlice<bf16>,
    pub q_proj: CudaSlice<bf16>,
    pub q_bias: CudaSlice<bf16>,
    pub k_proj: CudaSlice<bf16>,
    pub k_bias: CudaSlice<bf16>,
    pub v_proj: CudaSlice<bf16>,
    pub v_bias: CudaSlice<bf16>,
    pub o_proj: CudaSlice<bf16>,
    pub post_attention_layernorm: CudaSlice<bf16>,
    pub gate_proj: CudaSlice<bf16>,
    pub up_proj: CudaSlice<bf16>,
    pub down_proj: CudaSlice<bf16>,
}
