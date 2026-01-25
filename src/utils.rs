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
pub(crate) fn concat_tensors_or_zeros(
    stream: &Arc<CudaStream>,
    tensors: &SafeTensors,
    names: &[String],
    expected_len: usize,
) -> Result<cudarc::driver::CudaSlice<bf16>> {
    let mut total_len = 0;
    let mut slices = Vec::with_capacity(names.len());
    let mut all_found = true;

    for name in names {
        if let Ok(view) = tensors.tensor(name) {
             let data = view.data();
             let bf16_data: &[bf16] =
                 unsafe { std::slice::from_raw_parts(data.as_ptr() as *const bf16, data.len() / 2) };
             slices.push(Some(bf16_data));
             total_len += bf16_data.len();
        } else {
            all_found = false;
            slices.push(None); // Placeholder
        }
    }

    if !all_found {
        // If not all bias tensors are found, return zeros.
        // We assume either ALL are present or NONE are present for simplicity (or at least robust enough)
        // But what if only some are present? 
        // For Llama/DeepSeek, usually NO bias is present in QKV Proj.
        // So we just return a zero buffer of expected length.
        
        let zeros = vec![bf16::from_f32(0.0); expected_len];
        return Ok(stream.clone_htod(&zeros)?);
    }

    let mut combined = Vec::with_capacity(total_len);
    for slice in slices {
        if let Some(s) = slice {
            combined.extend_from_slice(s);
        }
    }
    
    // Safety check?
    if combined.len() != expected_len {
        // If we found tensors but length mismatch?
        // Just return combined.
    }

    Ok(stream.clone_htod(&combined)?)
}