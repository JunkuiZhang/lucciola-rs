use anyhow::{Context, Result};
use cudarc::cublas::sys::{
    cublasComputeType_t, cublasGemmAlgo_t, cublasGemmEx, cublasOperation_t, cudaDataType,
};
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use half::bf16;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use serde::Deserialize;
use std::{fs::File, path::Path, sync::Arc};

use crate::kernels::{CudaFunctions, load_cuda_funtion};
use crate::ptx::KV_CACHE_PTX;

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
}

fn default_rope_theta() -> f32 {
    1000000.0
}

pub struct LayerWeights {
    pub input_layernorm: CudaSlice<bf16>,
    pub q_proj: CudaSlice<bf16>,
    pub k_proj: CudaSlice<bf16>,
    pub v_proj: CudaSlice<bf16>,
    pub o_proj: CudaSlice<bf16>,
    pub post_attention_layernorm: CudaSlice<bf16>,
    pub gate_proj: CudaSlice<bf16>,
    pub up_proj: CudaSlice<bf16>,
    pub down_proj: CudaSlice<bf16>,
}

pub struct Qwen2Model {
    pub embed_tokens: CudaSlice<bf16>,
    pub layers: Vec<LayerWeights>,
    pub final_norm: CudaSlice<bf16>,
    pub rope: RopeCache,
    pub kv_cache: KVCache,
    cuda_functions: CudaFunctions,
}

pub struct RopeCache {
    pub cos: CudaSlice<f32>,
    pub sin: CudaSlice<f32>,
}

pub struct KVCache {
    pub k: CudaSlice<bf16>,
    pub v: CudaSlice<bf16>,
    pub max_seq_len: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    cuda_function: CudaFunction,
}

impl Qwen2Model {
    pub fn load(device: &Arc<CudaContext>, path: impl AsRef<Path>) -> Result<Self> {
        let config_file = path.as_ref().join("config.json");
        let config: ModelConfig = serde_json::from_reader(std::fs::File::open(config_file)?)?;
        println!("Model Config: {:#?}", config);

        let tensors_file = File::open(path.as_ref().join("model.safetensors"))?;
        let mmap = unsafe { MmapOptions::new().map(&tensors_file)? };
        let tensors = SafeTensors::deserialize(&mmap)?;
        let stream = device.default_stream();

        let embed_tokens = get_tensor(&stream, &tensors, "model.embed_tokens.weight")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let layer_prefix = format!("model.layers.{}.", layer_idx);
            let layer_weights = LayerWeights {
                input_layernorm: get_tensor(
                    &stream,
                    &tensors,
                    &format!("{}input_layernorm.weight", layer_prefix),
                )?,
                q_proj: get_tensor(
                    &stream,
                    &tensors,
                    &format!("{}self_attn.q_proj.weight", layer_prefix),
                )?,
                k_proj: get_tensor(
                    &stream,
                    &tensors,
                    &format!("{}self_attn.k_proj.weight", layer_prefix),
                )?,
                v_proj: get_tensor(
                    &stream,
                    &tensors,
                    &format!("{}self_attn.v_proj.weight", layer_prefix),
                )?,
                o_proj: get_tensor(
                    &stream,
                    &tensors,
                    &format!("{}self_attn.o_proj.weight", layer_prefix),
                )?,
                post_attention_layernorm: get_tensor(
                    &stream,
                    &tensors,
                    &format!("{}post_attention_layernorm.weight", layer_prefix),
                )?,
                gate_proj: get_tensor(
                    &stream,
                    &tensors,
                    &format!("{}mlp.gate_proj.weight", layer_prefix),
                )?,
                up_proj: get_tensor(
                    &stream,
                    &tensors,
                    &format!("{}mlp.up_proj.weight", layer_prefix),
                )?,
                down_proj: get_tensor(
                    &stream,
                    &tensors,
                    &format!("{}mlp.down_proj.weight", layer_prefix),
                )?,
            };
            layers.push(layer_weights);
        }
        let final_norm = get_tensor(&stream, &tensors, "model.norm.weight")?;

        // Qwen2.5 head_dim = hidden_size / num_attention_heads
        let head_dim = config.hidden_size / config.num_attention_heads;
        let rope = RopeCache::new(
            &stream,
            config.max_position_embeddings,
            head_dim,
            config.rope_theta,
        )?;
        let kv_cache = KVCache::new(device, &stream, &config)?;
        let cuda_functions = CudaFunctions::load(device)?;

        Ok(Qwen2Model {
            embed_tokens,
            layers,
            final_norm,
            rope,
            kv_cache,
            cuda_functions,
        })
    }
}

fn get_tensor(
    stream: &Arc<CudaStream>,
    tensors: &SafeTensors,
    name: &str,
) -> Result<CudaSlice<bf16>> {
    let view = tensors
        .tensor(name)
        .with_context(|| format!("Tensor {} not found", name))?;
    let data = view.data();
    let bf16_data: &[bf16] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const bf16, data.len() / 2) };
    Ok(stream.clone_htod(bf16_data)?)
}

impl RopeCache {
    fn new(
        stream: &Arc<CudaStream>,
        max_position_embeddings: usize,
        head_dim: usize,
        base: f32,
    ) -> Result<Self> {
        let mut cos_h = vec![0.0f32; max_position_embeddings * (head_dim / 2)];
        let mut sin_h = vec![0.0f32; max_position_embeddings * (head_dim / 2)];

        for pos in 0..max_position_embeddings {
            for i in 0..(head_dim / 2) {
                let theta = (pos as f32) / base.powf((2 * i) as f32 / head_dim as f32);
                cos_h[pos * (head_dim / 2) + i] = theta.cos();
                sin_h[pos * (head_dim / 2) + i] = theta.sin();
            }
        }

        let cos_dev = stream.clone_htod(&cos_h)?;
        let sin_dev = stream.clone_htod(&sin_h)?;

        Ok(RopeCache {
            cos: cos_dev,
            sin: sin_dev,
        })
    }
}

impl KVCache {
    pub fn new(
        context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        config: &ModelConfig,
    ) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let size = config.num_hidden_layers
            * config.num_key_value_heads
            * config.max_position_embeddings
            * head_dim;
        let k = stream.alloc_zeros::<bf16>(size)?;
        let v = stream.alloc_zeros::<bf16>(size)?;
        let cuda_function = load_cuda_funtion(context, KV_CACHE_PTX, "update_kv_cache_kernel")?;

        Ok(KVCache {
            k,
            v,
            max_seq_len: config.max_position_embeddings,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            num_layers: config.num_hidden_layers,
            cuda_function,
        })
    }

    pub fn update(
        &self,
        stream: &Arc<CudaStream>,
        layer_idx: usize,
        pos: usize,
        k_input: &CudaSlice<bf16>,
        v_input: &CudaSlice<bf16>,
    ) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems(self.num_kv_heads as u32);
        let mut builder = stream.launch_builder(&self.cuda_function);
        let layer_idx = layer_idx as i32;
        let pos = pos as i32;
        let num_layers = self.num_layers as i32;
        let num_kv_heads = self.num_kv_heads as i32;
        let max_seq_len = self.max_seq_len as i32;
        let head_dim = self.head_dim as i32;
        builder
            .arg(&self.k)
            .arg(&self.v)
            .arg(k_input)
            .arg(v_input)
            .arg(&layer_idx)
            .arg(&pos)
            .arg(&num_layers)
            .arg(&num_kv_heads)
            .arg(&max_seq_len)
            .arg(&head_dim);
        unsafe { builder.launch(cfg) }?;
        Ok(())
    }
}

impl Qwen2Model {
    unsafe fn matmul_bf16(
        &self,
        stream: &Arc<CudaStream>,
        blas: &CudaBlas,
        m: usize,
        n: usize,
        k: usize,
        a: &CudaSlice<bf16>,
        b: &CudaSlice<bf16>,
        c: &mut CudaSlice<bf16>,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        let m_blas = n as i32;
        let n_blas = m as i32;
        let k_blas = k as i32;

        unsafe {
            cublasGemmEx(
                *blas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
                m_blas,
                n_blas,
                k_blas,
                &alpha as *const f32 as *const _,
                b.device_ptr(stream).0 as *const _,
                cudaDataType::CUDA_R_16BF,
                k_blas,
                a.device_ptr(stream).0 as *const _,
                cudaDataType::CUDA_R_16BF,
                k_blas,
                &beta as *const f32 as *const _,
                c.device_ptr(stream).0 as *mut _,
                cudaDataType::CUDA_R_16BF,
                m_blas,
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            );
        }
        Ok(())
    }
}

impl Qwen2Model {
    fn apply_rmsnorm(
        &self,
        stream: &Arc<CudaStream>,
        out: &mut CudaSlice<bf16>,
        input: &CudaSlice<bf16>,
        weight: &CudaSlice<bf16>,
        epsilon: f32,
    ) -> Result<()> {
        let num_elements = input.len();
        let num_cols = weight.len();
        let num_rows = num_elements / num_cols;
        let cfg = LaunchConfig {
            grid_dim: (num_rows as u32, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        let num_cols = num_cols as i32;
        let mut build = stream.launch_builder(&self.cuda_functions.rmsnorm);
        build
            .arg(out)
            .arg(input)
            .arg(weight)
            .arg(&epsilon)
            .arg(&num_cols);
        unsafe {
            build.launch(cfg)?;
        }
        Ok(())
    }

    fn apply_activation(
        &self,
        stream: &Arc<CudaStream>,
        out: &mut CudaSlice<bf16>,
        gate: &CudaSlice<bf16>,
        up: &CudaSlice<bf16>,
    ) -> Result<()> {
        let n = out.len() as i32;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut build = stream.launch_builder(&self.cuda_functions.activation);
        build.arg(out).arg(gate).arg(up).arg(&n);
        unsafe {
            build.launch(cfg)?;
        }
        Ok(())
    }

    fn apply_rope(
        &self,
        stream: &Arc<CudaStream>,
        q: &mut CudaSlice<bf16>,
        k: &mut CudaSlice<bf16>,
        pos: usize,
        head_dim: usize,
        num_q_heads: usize,
        num_k_heads: usize,
    ) -> Result<()> {
        let total_threads = num_q_heads * (head_dim / 2);
        let cfg = LaunchConfig::for_num_elems(total_threads as u32);

        let pos = pos as i32;
        let head_dim = head_dim as i32;
        let num_q_heads = num_q_heads as i32;
        let num_k_heads = num_k_heads as i32;

        let mut builder = stream.launch_builder(&self.cuda_functions.rope);
        builder
            .arg(q)
            .arg(k)
            .arg(&self.rope.cos)
            .arg(&self.rope.sin)
            .arg(&pos)
            .arg(&head_dim)
            .arg(&num_q_heads)
            .arg(&num_k_heads);
        unsafe {
            builder.launch(cfg)?;
        }
        Ok(())
    }

    fn apply_softmax(
        &self,
        stream: &Arc<CudaStream>,
        logits: &mut CudaSlice<bf16>,
        seq_len: usize,
        num_heads: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (num_heads as u32, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        let seq_len = seq_len as u32;
        let mut builder = stream.launch_builder(&self.cuda_functions.softmax);
        builder.arg(logits).arg(&seq_len);
        unsafe {
            builder.launch(cfg)?;
        }
        Ok(())
    }
}
