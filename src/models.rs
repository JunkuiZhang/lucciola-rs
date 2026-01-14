use anyhow::Result;
use cudarc::cublas::CudaBlas;
use cudarc::cublas::sys::{
    cublasComputeType_t, cublasGemmAlgo_t, cublasGemmEx, cublasOperation_t, cudaDataType,
};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::{fs::File, path::Path, sync::Arc};
use tokenizers::Tokenizer;

use crate::config::ModelConfig;
use crate::kernels::CudaFunctions;
use crate::layers::buffers::InferenceBuffers;
use crate::layers::kv_cache::KVCache;
use crate::layers::rope::RopeCache;
use crate::layers::weights::LayerWeights;
use crate::streamer::Streamer;
use crate::utils::get_tensor;

struct ForwardPassBuffers<'a> {
    pub hidden_states: &'a mut CudaSlice<bf16>,
    pub norm_buffer: &'a mut CudaSlice<bf16>,
    pub q_states: &'a mut CudaSlice<bf16>,
    pub k_states: &'a mut CudaSlice<bf16>,
    pub v_states: &'a mut CudaSlice<bf16>,
    pub att_output: &'a mut CudaSlice<bf16>,
    pub mlp_gate: &'a mut CudaSlice<bf16>,
    pub mlp_up: &'a mut CudaSlice<bf16>,
}

pub struct Qwen2Model {
    pub device: Arc<CudaContext>,
    pub blas: Arc<CudaBlas>,
    pub embed_tokens: CudaSlice<bf16>,
    pub lm_head: CudaSlice<bf16>,
    pub layers: Vec<LayerWeights>,
    pub final_norm: CudaSlice<bf16>,
    pub rope: RopeCache,
    pub kv_cache: KVCache,
    pub buffers: InferenceBuffers,
    cuda_functions: CudaFunctions,
    pub config: ModelConfig,
}

impl Qwen2Model {
    pub fn load(gpu_id: usize, path: impl AsRef<Path>) -> Result<Self> {
        let device = CudaContext::new(gpu_id)?;
        let blas = Arc::new(CudaBlas::new(device.default_stream())?);

        let config_file = path.as_ref().join("config.json");
        let config: ModelConfig = serde_json::from_reader(std::fs::File::open(config_file)?)?;
        println!("Model Config: {:#?}", config);

        let tensors_file = File::open(path.as_ref().join("model.safetensors"))?;
        let mmap = unsafe { MmapOptions::new().map(&tensors_file)? };
        let tensors = SafeTensors::deserialize(&mmap)?;
        let stream = device.default_stream();

        let embed_tokens = get_tensor(&stream, &tensors, "model.embed_tokens.weight")?;
        let lm_head = if config.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            get_tensor(&stream, &tensors, "lm_head.weight")?
        };

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
        // Initialize KV Cache (using 80% of remaining free VRAM)
        let kv_cache = KVCache::new(&device, &stream, &config, 0.8)?;
        let cuda_functions = CudaFunctions::load(&device, head_dim)?;
        let buffers = InferenceBuffers::new(&stream, &config)?;

        Ok(Qwen2Model {
            device,
            blas,
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
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        sampler: &mut crate::sampler::Sampler,
        max_new_tokens: usize,
        tokenizer: &Tokenizer,
        mut token_callback: impl FnMut(&str) -> bool,
    ) -> Result<()> {
        let mut cache_pos = 0;
        let mut next_token_id = 0;
        let mut streamer = Streamer::new(tokenizer);

        let eos_token_id = self.config.eos_token_id;
        let bos_token_id = self.config.bos_token_id;

        // Prefill
        if !prompt_ids.is_empty() {
            self.forward(prompt_ids, cache_pos)?;
            cache_pos += prompt_ids.len();

            let mut logits = self.sample()?;
            next_token_id = sampler.sample(&mut logits)?;

            // Check stop conditions for first token
            if next_token_id == eos_token_id || next_token_id == bos_token_id {
                return Ok(());
            }

            if let Some(text) = streamer.put(next_token_id) {
                if !token_callback(&text) {
                    return Ok(());
                }
            }
        }

        // Decode Loop
        for _ in 0..max_new_tokens {
            self.forward(&[next_token_id], cache_pos)?;
            cache_pos += 1;

            let mut logits = self.sample()?;
            next_token_id = sampler.sample(&mut logits)?;

            if next_token_id == eos_token_id || next_token_id == bos_token_id {
                break;
            }

            if let Some(text) = streamer.put(next_token_id) {
                if !token_callback(&text) {
                    break;
                }
            }
        }

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
    pub fn sample(&self) -> Result<Vec<f32>> {
        let stream = self.device.default_stream();
        let vocab_size = self.config.vocab_size;
        let hidden_dim = self.config.hidden_size;
        let mut logits = stream.alloc_zeros::<bf16>(vocab_size)?;

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
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
                vocab_size as i32, // m
                1,                 // n
                hidden_dim as i32, // k
                &alpha as *const f32 as *const _,
                self.lm_head.device_ptr(&stream).0 as *const _,
                cudaDataType::CUDA_R_16BF,
                hidden_dim as i32, // lda
                self.buffers.hidden_states.device_ptr(&stream).0 as *const _,
                cudaDataType::CUDA_R_16BF,
                hidden_dim as i32, // ldb
                &beta as *const f32 as *const _,
                logits.device_ptr_mut(&stream).0 as _,
                cudaDataType::CUDA_R_16BF,
                vocab_size as i32, // ldc
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            );
        }

        let host_data = stream.clone_dtoh(&logits)?;
        Ok(host_data.into_iter().map(|x: bf16| x.to_f32()).collect())
    }

    fn forward_layer(
        stream: &Arc<CudaStream>,
        blas: &CudaBlas,
        config: &ModelConfig,
        funcs: &CudaFunctions,
        rope: &RopeCache,
        kv_cache: &mut KVCache,
        layer: &LayerWeights,
        i: usize,
        seq_len: usize,
        cache_pos: usize,
        bufs: &mut ForwardPassBuffers,
    ) -> Result<()> {
        let hidden_dim = config.hidden_size;
        let head_dim = hidden_dim / config.num_attention_heads;
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let intermediate_size = config.intermediate_size;
        let rms_norm_eps = config.rms_norm_eps;

        // 1. RMSNorm
        funcs.apply_rmsnorm(
            stream,
            bufs.norm_buffer,
            Some(bufs.hidden_states),
            &layer.input_layernorm,
            rms_norm_eps,
        )?;

        // 2. Batched QKV Proj
        unsafe {
            Self::matmul_bf16(
                stream,
                blas,
                seq_len,
                hidden_dim,
                hidden_dim,
                bufs.norm_buffer,
                &layer.q_proj,
                bufs.q_states,
                1.0,
                0.0,
            )?;
            Self::matmul_bf16(
                stream,
                blas,
                seq_len,
                num_kv_heads * head_dim,
                hidden_dim,
                bufs.norm_buffer,
                &layer.k_proj,
                bufs.k_states,
                1.0,
                0.0,
            )?;
            Self::matmul_bf16(
                stream,
                blas,
                seq_len,
                num_kv_heads * head_dim,
                hidden_dim,
                bufs.norm_buffer,
                &layer.v_proj,
                bufs.v_states,
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

            // RoPE & KV Update & Attention
            funcs.apply_rope(
                rope,
                stream,
                bufs.q_states,
                q_offset,
                &layer.q_bias,
                bufs.k_states,
                k_offset,
                &layer.k_bias,
                current_pos,
                head_dim,
                num_q_heads,
                num_kv_heads,
            )?;
            kv_cache.update(
                stream,
                i,
                current_pos,
                bufs.k_states,
                k_offset,
                bufs.v_states,
                v_offset,
                &layer.v_bias,
            )?;

            funcs.apply_flash_decoding(
                stream,
                bufs.att_output,
                t * hidden_dim,
                bufs.q_states,
                q_offset,
                kv_cache,
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
                bufs.att_output,
                &layer.o_proj,
                bufs.hidden_states,
                1.0,
                1.0,
            )?;
        }

        // --- Batched MLP ---
        funcs.apply_rmsnorm(
            stream,
            bufs.norm_buffer,
            Some(bufs.hidden_states),
            &layer.post_attention_layernorm,
            rms_norm_eps,
        )?;

        unsafe {
            Self::matmul_bf16(
                stream,
                blas,
                seq_len,
                intermediate_size,
                hidden_dim,
                bufs.norm_buffer,
                &layer.gate_proj,
                bufs.mlp_gate,
                1.0,
                0.0,
            )?;
            Self::matmul_bf16(
                stream,
                blas,
                seq_len,
                intermediate_size,
                hidden_dim,
                bufs.norm_buffer,
                &layer.up_proj,
                bufs.mlp_up,
                1.0,
                0.0,
            )?;
        }

        funcs.apply_activation(stream, bufs.mlp_gate, None, bufs.mlp_up)?;

        unsafe {
            Self::matmul_bf16(
                stream,
                blas,
                seq_len,
                hidden_dim,
                intermediate_size,
                bufs.mlp_gate,
                &layer.down_proj,
                bufs.hidden_states,
                1.0,
                1.0,
            )?;
        }

        Ok(())
    }

    pub fn forward(&mut self, input_ids: &[u32], cache_pos: usize) -> Result<()> {
        let stream = self.device.default_stream();
        let blas = &*self.blas;

        let seq_len = input_ids.len();
        if seq_len == 0 {
            return Ok(());
        }

        let hidden_dim = self.config.hidden_size;
        let head_dim = hidden_dim / self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let intermediate_size = self.config.intermediate_size;
        let rms_norm_eps = self.config.rms_norm_eps;

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

            let mut bufs = ForwardPassBuffers {
                hidden_states: &mut self.buffers.hidden_states,
                norm_buffer: &mut self.buffers.norm_buffer,
                q_states: &mut self.buffers.q_states,
                k_states: &mut self.buffers.k_states,
                v_states: &mut self.buffers.v_states,
                att_output: &mut self.buffers.att_output,
                mlp_gate: &mut self.buffers.mlp_gate,
                mlp_up: &mut self.buffers.mlp_up,
            };

            for (i, layer) in self.layers.iter().enumerate() {
                Self::forward_layer(
                    &stream,
                    blas,
                    &self.config,
                    funcs,
                    rope,
                    &mut self.kv_cache,
                    layer,
                    i,
                    1,
                    cache_pos,
                    &mut bufs,
                )?;
            }

            funcs.apply_rmsnorm(
                &stream,
                bufs.hidden_states,
                None,
                &self.final_norm,
                rms_norm_eps,
            )?;
            return Ok(());
        }

        // --- Batched Path (SeqLen > 1) ---
        let mut hidden_states = stream.alloc_zeros::<bf16>(seq_len * hidden_dim)?;

        // 1. Batched Embedding
        let input_ids_dev = stream.clone_htod(input_ids)?;

        funcs.apply_batched_embedding(
            &stream,
            &self.embed_tokens,
            &input_ids_dev,
            &mut hidden_states,
            hidden_dim,
            seq_len,
        )?;

        // Allocate Batched Buffers
        let mut norm_buffer = stream.alloc_zeros::<bf16>(seq_len * hidden_dim)?;
        let mut q_states = stream.alloc_zeros::<bf16>(seq_len * hidden_dim)?;
        let mut k_states = stream.alloc_zeros::<bf16>(seq_len * num_kv_heads * head_dim)?;
        let mut v_states = stream.alloc_zeros::<bf16>(seq_len * num_kv_heads * head_dim)?;
        let mut att_output = stream.alloc_zeros::<bf16>(seq_len * hidden_dim)?;
        let mut mlp_gate = stream.alloc_zeros::<bf16>(seq_len * intermediate_size)?;
        let mut mlp_up = stream.alloc_zeros::<bf16>(seq_len * intermediate_size)?;

        {
            let mut bufs = ForwardPassBuffers {
                hidden_states: &mut hidden_states,
                norm_buffer: &mut norm_buffer,
                q_states: &mut q_states,
                k_states: &mut k_states,
                v_states: &mut v_states,
                att_output: &mut att_output,
                mlp_gate: &mut mlp_gate,
                mlp_up: &mut mlp_up,
            };

            for (i, layer) in self.layers.iter().enumerate() {
                Self::forward_layer(
                    &stream,
                    blas,
                    &self.config,
                    funcs,
                    rope,
                    &mut self.kv_cache,
                    layer,
                    i,
                    seq_len,
                    cache_pos,
                    &mut bufs,
                )?;
            }

            // Final Norm
            funcs.apply_rmsnorm(
                &stream,
                bufs.hidden_states,
                None,
                &self.final_norm,
                rms_norm_eps,
            )?;
        }

        // Extract last token state
        let last_offset = (seq_len - 1) * hidden_dim;
        let last_token_state = hidden_states.slice(last_offset..last_offset + hidden_dim);
        stream.memcpy_dtod(&last_token_state, &mut self.buffers.hidden_states)?;

        Ok(())
    }
}
