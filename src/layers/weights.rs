use cudarc::driver::CudaSlice;
use half::bf16;

pub struct LayerWeights {
    pub input_layernorm: CudaSlice<bf16>,
    pub qkv_proj: CudaSlice<bf16>,
    pub qkv_bias: CudaSlice<bf16>,
    pub o_proj: CudaSlice<bf16>,
    pub post_attention_layernorm: CudaSlice<bf16>,
    pub gate_up_proj: CudaSlice<bf16>,
    pub down_proj: CudaSlice<bf16>,
}
