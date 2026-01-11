use anyhow::{Context, Result};
use cudarc::cublas::CudaBlas;
use cudarc::cublas::sys::{
    cublasComputeType_t, cublasGemmAlgo_t, cublasGemmEx, cublasOperation_t, cudaDataType,
};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig,
    PushKernelArg,
};
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
    pub intermediate_size: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
}

fn default_rope_theta() -> f32 {
    1000000.0
}

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

pub struct Qwen2Model {
    pub embed_tokens: CudaSlice<bf16>,
    pub lm_head: CudaSlice<bf16>,
    pub layers: Vec<LayerWeights>,
    pub final_norm: CudaSlice<bf16>,
    pub rope: RopeCache,
    pub kv_cache: KVCache,
    pub buffers: InferenceBuffers,
    cuda_functions: CudaFunctions,
    config: ModelConfig,
}

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
    fn new(stream: &Arc<CudaStream>, config: &ModelConfig) -> Result<Self> {
        let hidden_dim = config.hidden_size;
        let head_dim = hidden_dim / config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let intermediate_size = config.intermediate_size; // Or calculate it if not available

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
        let lm_head = get_tensor(&stream, &tensors, "lm_head.weight").or_else(|_| {
            println!("lm_head.weight not found, using embed_tokens (tied weights)");
            Ok::<_, anyhow::Error>(embed_tokens.clone())
        })?;

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
                q_bias: get_tensor(
                    &stream,
                    &tensors,
                    &format!("{}self_attn.q_proj.bias", layer_prefix),
                )?,
                k_proj: get_tensor(
                    &stream,
                    &tensors,
                    &format!("{}self_attn.k_proj.weight", layer_prefix),
                )?,
                k_bias: get_tensor(
                    &stream,
                    &tensors,
                    &format!("{}self_attn.k_proj.bias", layer_prefix),
                )?,
                v_proj: get_tensor(
                    &stream,
                    &tensors,
                    &format!("{}self_attn.v_proj.weight", layer_prefix),
                )?,
                v_bias: get_tensor(
                    &stream,
                    &tensors,
                    &format!("{}self_attn.v_proj.bias", layer_prefix),
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
        let cuda_functions = CudaFunctions::load(device, head_dim)?;
        let buffers = InferenceBuffers::new(&stream, &config)?;

        Ok(Qwen2Model {
            embed_tokens,
            lm_head,
            layers,
            final_norm,
            rope,
            kv_cache,
            buffers,
            cuda_functions,
            config,
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
        k_input: &impl DevicePtr<bf16>,
        v_input: &impl DevicePtr<bf16>,
        v_bias: &CudaSlice<bf16>,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (self.num_kv_heads as u32, 1, 1),
            block_dim: (self.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let (k_ptr, _k_guard) = DevicePtr::device_ptr(k_input, stream);
        let (v_ptr, _v_guard) = DevicePtr::device_ptr(v_input, stream);

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
            .arg(&k_ptr)
            .arg(&v_ptr)
            .arg(v_bias)
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
        cuda_functions: &CudaFunctions,
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

        let out_ptr = out.device_ptr(stream).0;
        let input_ptr = if let Some(inp) = input {
            inp.device_ptr(stream).0
        } else {
            out_ptr
        };

        let mut build = stream.launch_builder(&cuda_functions.rmsnorm);
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

    fn apply_activation(
        cuda_functions: &CudaFunctions,
        stream: &Arc<CudaStream>,
        out: &mut CudaSlice<bf16>,
        gate: Option<&CudaSlice<bf16>>,
        up: &CudaSlice<bf16>,
    ) -> Result<()> {
        let n = out.len() as i32;
        let cfg = LaunchConfig::for_num_elems(n as u32);

        let out_ptr = out.device_ptr(stream).0;
        let gate_ptr = if let Some(g) = gate {
            g.device_ptr(stream).0
        } else {
            out_ptr
        };

        let mut build = stream.launch_builder(&cuda_functions.activation);
        build.arg(&out_ptr).arg(&gate_ptr).arg(up).arg(&n);

        unsafe {
            build.launch(cfg)?;
        }
        Ok(())
    }

    fn apply_rope(
        cuda_functions: &CudaFunctions,
        rope: &RopeCache,
        stream: &Arc<CudaStream>,
        q: &mut (impl DevicePtrMut<bf16> + DevicePtr<bf16>),
        q_bias: &CudaSlice<bf16>,
        k: &mut (impl DevicePtrMut<bf16> + DevicePtr<bf16>),
        k_bias: &CudaSlice<bf16>,
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

        let (q_ptr, _q_guard) = DevicePtr::device_ptr(q, stream);
        let (k_ptr, _k_guard) = DevicePtr::device_ptr(k, stream);

        let mut builder = stream.launch_builder(&cuda_functions.rope);
        builder
            .arg(&q_ptr)
            .arg(q_bias)
            .arg(&k_ptr)
            .arg(k_bias)
            .arg(&rope.cos)
            .arg(&rope.sin)
            .arg(&pos)
            .arg(&head_dim)
            .arg(&num_q_heads)
            .arg(&num_k_heads);
        unsafe {
            builder.launch(cfg)?;
        }
        Ok(())
    }

    fn apply_flash_decoding(
        cuda_functions: &CudaFunctions,
        stream: &Arc<CudaStream>,
        output: &mut (impl DevicePtrMut<bf16> + DevicePtr<bf16>),
        q: &impl DevicePtr<bf16>,
        cache: &KVCache,
        layer_idx: usize,
        pos: usize,
        head_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
    ) -> Result<()> {
        let max_seq_len = cache.max_seq_len as i32;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let cfg = LaunchConfig {
            grid_dim: (num_q_heads as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: (head_dim as u32) * 4,
        };

        let mut builder = stream.launch_builder(&cuda_functions.attention);

        let layer_idx = layer_idx as i32;
        let num_q_heads_i32 = num_q_heads as i32;
        let num_kv_heads_i32 = num_kv_heads as i32;
        let head_dim_i32 = head_dim as i32;
        let current_pos = pos as i32;

        let (out_ptr, _out_guard) = DevicePtr::device_ptr(output, stream);
        let (q_ptr, _q_guard) = DevicePtr::device_ptr(q, stream);

        builder
            .arg(&out_ptr)
            .arg(&q_ptr)
            .arg(&cache.k)
            .arg(&cache.v)
            .arg(&layer_idx)
            .arg(&num_q_heads_i32)
            .arg(&num_kv_heads_i32)
            .arg(&head_dim_i32)
            .arg(&max_seq_len)
            .arg(&current_pos)
            .arg(&scale);

        unsafe {
            builder.launch(cfg)?;
        }
        Ok(())
    }
}

impl Qwen2Model {
    pub fn sample(
        &self,
        _device: &CudaContext,
        stream: &Arc<CudaStream>,
        blas: &CudaBlas,
    ) -> Result<Vec<f32>> {
        let vocab_size = self.config.vocab_size;
        let hidden_dim = self.config.hidden_size;
        let logits = stream.alloc_zeros::<bf16>(vocab_size)?;

        // C = A * B
        // C [V, 1] = lm_head [V, H] * hidden_states [H, 1]
        // CuBLAS Col Major:
        // A_mem is [H, V] (because RowMajor [V, H]). We want [V, H] -> OP_T. lda=H.
        // B_mem is [H, 1]. We want [H, 1] -> OP_N. ldb=H.
        // C_mem is [V, 1]. ldc=V.

        let alpha = 1.0f32;
        let beta = 0.0f32;

        unsafe {
            cublasGemmEx(
                *blas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
                vocab_size as i32, // m
                1,                 // n
                hidden_dim as i32, // k
                &alpha as *const f32 as *const _,
                self.lm_head.device_ptr(stream).0 as *const _,
                cudaDataType::CUDA_R_16BF,
                hidden_dim as i32, // lda
                self.buffers.hidden_states.device_ptr(stream).0 as *const _,
                cudaDataType::CUDA_R_16BF,
                hidden_dim as i32, // ldb
                &beta as *const f32 as *const _,
                logits.device_ptr(stream).0 as *mut _,
                cudaDataType::CUDA_R_16BF,
                vocab_size as i32, // ldc
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            );
        }

        let mut host_data = vec![bf16::default(); vocab_size];
        stream.memcpy_dtoh(&logits, &mut host_data)?;
        stream.synchronize()?;
        Ok(host_data.iter().map(|x: &bf16| x.to_f32()).collect())
    }

    pub fn forward(
        &mut self,
        stream: &Arc<CudaStream>,
        blas: &CudaBlas,
        input_ids: &[u32],
        cache_pos: usize,
    ) -> Result<()> {
        let seq_len = input_ids.len();
        if seq_len == 0 {
            return Ok(());
        }

        let hidden_dim = self.config.hidden_size;
        let head_dim = hidden_dim / self.config.num_attention_heads;
        let num_q_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let intermediate_size = self.config.intermediate_size;

        let funcs = &self.cuda_functions;
        let rope = &self.rope;

        if seq_len == 1 {
            let token_id = input_ids[0] as usize;

            // 1. Embedding
            let embed_offset = token_id * hidden_dim;
            {
                let embed_view = self
                    .embed_tokens
                    .slice(embed_offset..embed_offset + hidden_dim);
                stream.memcpy_dtod(&embed_view, &mut self.buffers.hidden_states)?;
            }

            let buffers = &mut self.buffers;

            for (i, layer) in self.layers.iter().enumerate() {
                // --- Attention Block ---
                Self::apply_rmsnorm(
                    funcs,
                    stream,
                    &mut buffers.norm_buffer,
                    Some(&buffers.hidden_states),
                    &layer.input_layernorm,
                    1e-6,
                )?;

                unsafe {
                    Self::matmul_bf16(
                        stream,
                        blas,
                        1,
                        hidden_dim,
                        hidden_dim,
                        &buffers.norm_buffer,
                        &layer.q_proj,
                        &mut buffers.q_states,
                        1.0,
                        0.0,
                    )?;
                    Self::matmul_bf16(
                        stream,
                        blas,
                        1,
                        num_kv_heads * head_dim,
                        hidden_dim,
                        &buffers.norm_buffer,
                        &layer.k_proj,
                        &mut buffers.k_states,
                        1.0,
                        0.0,
                    )?;
                    Self::matmul_bf16(
                        stream,
                        blas,
                        1,
                        num_kv_heads * head_dim,
                        hidden_dim,
                        &buffers.norm_buffer,
                        &layer.v_proj,
                        &mut buffers.v_states,
                        1.0,
                        0.0,
                    )?;
                }

                Self::apply_rope(
                    funcs,
                    rope,
                    stream,
                    &mut buffers.q_states,
                    &layer.q_bias,
                    &mut buffers.k_states,
                    &layer.k_bias,
                    cache_pos,
                    head_dim,
                    num_q_heads,
                    num_kv_heads,
                )?;
                self.kv_cache.update(
                    stream,
                    i,
                    cache_pos,
                    &buffers.k_states,
                    &buffers.v_states,
                    &layer.v_bias,
                )?;
                Self::apply_flash_decoding(
                    funcs,
                    stream,
                    &mut buffers.att_output,
                    &buffers.q_states,
                    &self.kv_cache,
                    i,
                    cache_pos,
                    head_dim,
                    num_q_heads,
                    num_kv_heads,
                )?;

                unsafe {
                    Self::matmul_bf16(
                        stream,
                        blas,
                        1,
                        hidden_dim,
                        hidden_dim,
                        &buffers.att_output,
                        &layer.o_proj,
                        &mut buffers.hidden_states,
                        1.0,
                        1.0,
                    )?;
                }

                // --- MLP Block ---
                Self::apply_rmsnorm(
                    funcs,
                    stream,
                    &mut buffers.norm_buffer,
                    Some(&buffers.hidden_states),
                    &layer.post_attention_layernorm,
                    1e-6,
                )?;

                unsafe {
                    Self::matmul_bf16(
                        stream,
                        blas,
                        1,
                        intermediate_size,
                        hidden_dim,
                        &buffers.norm_buffer,
                        &layer.gate_proj,
                        &mut buffers.mlp_gate,
                        1.0,
                        0.0,
                    )?;
                    Self::matmul_bf16(
                        stream,
                        blas,
                        1,
                        intermediate_size,
                        hidden_dim,
                        &buffers.norm_buffer,
                        &layer.up_proj,
                        &mut buffers.mlp_up,
                        1.0,
                        0.0,
                    )?;
                }

                Self::apply_activation(
                    funcs,
                    stream,
                    &mut buffers.mlp_gate,
                    None,
                    &buffers.mlp_up,
                )?;

                unsafe {
                    Self::matmul_bf16(
                        stream,
                        blas,
                        1,
                        hidden_dim,
                        intermediate_size,
                        &buffers.mlp_gate,
                        &layer.down_proj,
                        &mut buffers.hidden_states,
                        1.0,
                        1.0,
                    )?;
                }
            }

            Self::apply_rmsnorm(
                funcs,
                stream,
                &mut buffers.hidden_states,
                None,
                &self.final_norm,
                1e-6,
            )?;
            stream.synchronize()?;
            return Ok(());
        }

        // --- Batched Path (SeqLen > 1) ---
        let mut hidden_states = stream.alloc_zeros::<bf16>(seq_len * hidden_dim)?;

        // 1. Batched Embedding
        for (t, &id) in input_ids.iter().enumerate() {
            let offset = t * hidden_dim; // Destination offset (row t)
            let embed_offset = (id as usize) * hidden_dim;
            let embed_view = self
                .embed_tokens
                .slice(embed_offset..embed_offset + hidden_dim);
            let mut hidden_sub = hidden_states.slice_mut(offset..offset + hidden_dim);
            stream.memcpy_dtod(&embed_view, &mut hidden_sub)?;
        }

        // Allocate Batched Buffers
        let mut norm_buffer = stream.alloc_zeros::<bf16>(seq_len * hidden_dim)?;
        let mut q_states = stream.alloc_zeros::<bf16>(seq_len * hidden_dim)?;
        let mut k_states = stream.alloc_zeros::<bf16>(seq_len * num_kv_heads * head_dim)?;
        let mut v_states = stream.alloc_zeros::<bf16>(seq_len * num_kv_heads * head_dim)?;
        let mut att_output = stream.alloc_zeros::<bf16>(seq_len * hidden_dim)?;
        let mut mlp_gate = stream.alloc_zeros::<bf16>(seq_len * intermediate_size)?;
        let mut mlp_up = stream.alloc_zeros::<bf16>(seq_len * intermediate_size)?;

        for (i, layer) in self.layers.iter().enumerate() {
            // 1. RMSNorm
            Self::apply_rmsnorm(
                funcs,
                stream,
                &mut norm_buffer,
                Some(&hidden_states),
                &layer.input_layernorm,
                1e-6,
            )?;

            // 2. Batched QKV Proj
            unsafe {
                Self::matmul_bf16(
                    stream,
                    blas,
                    seq_len,
                    hidden_dim,
                    hidden_dim,
                    &norm_buffer,
                    &layer.q_proj,
                    &mut q_states,
                    1.0,
                    0.0,
                )?;
                Self::matmul_bf16(
                    stream,
                    blas,
                    seq_len,
                    num_kv_heads * head_dim,
                    hidden_dim,
                    &norm_buffer,
                    &layer.k_proj,
                    &mut k_states,
                    1.0,
                    0.0,
                )?;
                Self::matmul_bf16(
                    stream,
                    blas,
                    seq_len,
                    num_kv_heads * head_dim,
                    hidden_dim,
                    &norm_buffer,
                    &layer.v_proj,
                    &mut v_states,
                    1.0,
                    0.0,
                )?;
            }

            // 3. Serial Loop for Attention/RoPE/Cache
            for t in 0..seq_len {
                let current_pos = cache_pos + t;
                let q_offset = t * hidden_dim;
                let k_offset = t * num_kv_heads * head_dim;
                let v_offset = t * num_kv_heads * head_dim;

                let mut q_sub = q_states.slice_mut(q_offset..q_offset + hidden_dim);
                let mut k_sub = k_states.slice_mut(k_offset..k_offset + num_kv_heads * head_dim);
                let v_sub = v_states.slice(v_offset..v_offset + num_kv_heads * head_dim);

                // RoPE & KV Update & Attention
                Self::apply_rope(
                    funcs,
                    rope,
                    stream,
                    &mut q_sub,
                    &layer.q_bias,
                    &mut k_sub,
                    &layer.k_bias,
                    current_pos,
                    head_dim,
                    num_q_heads,
                    num_kv_heads,
                )?;
                self.kv_cache
                    .update(stream, i, current_pos, &k_sub, &v_sub, &layer.v_bias)?;

                let mut att_sub = att_output.slice_mut(q_offset..q_offset + hidden_dim);
                Self::apply_flash_decoding(
                    funcs,
                    stream,
                    &mut att_sub,
                    &q_sub,
                    &self.kv_cache,
                    i,
                    current_pos,
                    head_dim,
                    num_q_heads,
                    num_kv_heads,
                )?;
            }

            // 4. Batched Output Proj
            unsafe {
                Self::matmul_bf16(
                    stream,
                    blas,
                    seq_len,
                    hidden_dim,
                    hidden_dim,
                    &att_output,
                    &layer.o_proj,
                    &mut hidden_states,
                    1.0,
                    1.0,
                )?;
            }

            // --- Batched MLP ---
            Self::apply_rmsnorm(
                funcs,
                stream,
                &mut norm_buffer,
                Some(&hidden_states),
                &layer.post_attention_layernorm,
                1e-6,
            )?;

            unsafe {
                Self::matmul_bf16(
                    stream,
                    blas,
                    seq_len,
                    intermediate_size,
                    hidden_dim,
                    &norm_buffer,
                    &layer.gate_proj,
                    &mut mlp_gate,
                    1.0,
                    0.0,
                )?;
                Self::matmul_bf16(
                    stream,
                    blas,
                    seq_len,
                    intermediate_size,
                    hidden_dim,
                    &norm_buffer,
                    &layer.up_proj,
                    &mut mlp_up,
                    1.0,
                    0.0,
                )?;
            }

            Self::apply_activation(funcs, stream, &mut mlp_gate, None, &mlp_up)?;

            unsafe {
                Self::matmul_bf16(
                    stream,
                    blas,
                    seq_len,
                    hidden_dim,
                    intermediate_size,
                    &mlp_gate,
                    &layer.down_proj,
                    &mut hidden_states,
                    1.0,
                    1.0,
                )?;
            }
        }

        // Final Norm
        Self::apply_rmsnorm(
            funcs,
            stream,
            &mut hidden_states,
            None,
            &self.final_norm,
            1e-6,
        )?;

        // Extract last token state
        let last_offset = (seq_len - 1) * hidden_dim;
        let last_token_state = hidden_states.slice(last_offset..last_offset + hidden_dim);
        stream.memcpy_dtod(&last_token_state, &mut self.buffers.hidden_states)?;

        stream.synchronize()?;
        Ok(())
    }
}
