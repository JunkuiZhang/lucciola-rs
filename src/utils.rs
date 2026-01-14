use anyhow::Context;
use anyhow::Result;
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaStream},
    nvrtc::Ptx,
};
use half::bf16;
use safetensors::SafeTensors;
use std::sync::Arc;

pub(crate) fn load_cuda_funtion(
    context: &Arc<CudaContext>,
    ptx_src: &str,
    function_name: &str,
) -> Result<CudaFunction> {
    let ptx = Ptx::from_src(ptx_src);
    let module = context.load_module(ptx)?;
    Ok(module.load_function(function_name)?)
}

pub(crate) fn get_tensor(
    stream: &Arc<CudaStream>,
    tensors: &SafeTensors,
    name: &str,
) -> Result<cudarc::driver::CudaSlice<bf16>> {
    let view = tensors
        .tensor(name)
        .with_context(|| format!("Tensor {} not found", name))?;
    let data = view.data();
    let bf16_data: &[bf16] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const bf16, data.len() / 2) };
    Ok(stream.clone_htod(bf16_data)?)
}

pub(crate) fn concat_tensors(
    stream: &Arc<CudaStream>,
    tensors: &SafeTensors,
    names: &[String],
) -> Result<cudarc::driver::CudaSlice<bf16>> {
    let mut total_len = 0;
    let mut slices = Vec::with_capacity(names.len());

    for name in names {
        let view = tensors
            .tensor(name)
            .with_context(|| format!("Tensor {} not found", name))?;
        let data = view.data();
        let bf16_data: &[bf16] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const bf16, data.len() / 2) };
        slices.push(bf16_data);
        total_len += bf16_data.len();
    }

    let mut combined = Vec::with_capacity(total_len);
    for slice in slices {
        combined.extend_from_slice(slice);
    }
    
    Ok(stream.clone_htod(&combined)?)
}
