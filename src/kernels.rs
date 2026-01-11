use std::sync::Arc;

use anyhow::Result;
use cudarc::{
    driver::{CudaContext, CudaFunction},
    nvrtc::Ptx,
};

use crate::ptx;

pub(crate) struct CudaFunctions {
    pub(crate) activation: CudaFunction,
    pub(crate) attention: CudaFunction,
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
        println!("Loading rmsnorm kernel...");
        let rmsnorm = load_cuda_funtion(context, ptx::RMSNORM_PTX, "rmsnorm_nvidia")?;
        println!("Loading rope kernel...");
        let rope = load_cuda_funtion(context, ptx::ROPE_PTX, "rope")?;
        println!("Loading softmax kernel...");

        Ok(Self {
            activation,
            attention,
            rmsnorm,
            rope,
        })
    }
}

pub(crate) fn load_cuda_funtion(
    context: &Arc<CudaContext>,
    ptx_src: &str,
    function_name: &str,
) -> Result<CudaFunction> {
    let ptx = Ptx::from_src(ptx_src);
    let module = context.load_module(ptx)?;
    Ok(module.load_function(function_name)?)
}
