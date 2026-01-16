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
use crate::sampler::Sampler;
use crate::streamer::Streamer;
use crate::utils::{concat_tensors, get_tensor};

struct ForwardPassBuffers<'a> {
    pub hidden_states: &'a mut CudaSlice<bf16>,
    pub norm_buffer: &'a mut CudaSlice<bf16>,
    pub qkv_states: &'a mut CudaSlice<bf16>,
    pub att_output: &'a mut CudaSlice<bf16>,
    pub gate_up_states: &'a mut CudaSlice<bf16>,
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
    pub sample_indices_buffer: CudaSlice<u32>,
    pub sort_buffer: CudaSlice<u8>, // Temporary buffer for sort
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
                qkv_proj: concat_tensors(
                    &stream,
                    &tensors,
                    &[
                        format!("{}self_attn.q_proj.weight", layer_prefix),
                        format!("{}self_attn.k_proj.weight", layer_prefix),
                        format!("{}self_attn.v_proj.weight", layer_prefix),
                    ],
                )?,
                qkv_bias: concat_tensors(
                    &stream,
                    &tensors,
                    &[
                        format!("{}self_attn.q_proj.bias", layer_prefix),
                        format!("{}self_attn.k_proj.bias", layer_prefix),
                        format!("{}self_attn.v_proj.bias", layer_prefix),
                    ],
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
                gate_up_proj: concat_tensors(
                    &stream,
                    &tensors,
                    &[
                        format!("{}mlp.gate_proj.weight", layer_prefix),
                        format!("{}mlp.up_proj.weight", layer_prefix),
                    ],
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
        let sample_indices_buffer = stream.alloc_zeros::<u32>(1)?;

        let vocab_size = config.vocab_size;
        let n: u32 = 1u32 << (32 - (vocab_size as u32 - 1).leading_zeros());
        // KeyValuePair is {int, float} = 8 bytes
        let sort_buffer = stream.alloc_zeros::<u8>(n as usize * 8)?;

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
            sample_indices_buffer,
            sort_buffer,
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

            next_token_id = self.sample_token(sampler)?;

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

            next_token_id = self.sample_token(sampler)?;

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
    unsafe fn matmul_bf16_strided(
        stream: &Arc<CudaStream>,
        blas: &CudaBlas,
        m: usize,
        n: usize,
        k: usize,
        a: &CudaSlice<bf16>,
        lda: usize,
        b: &CudaSlice<bf16>,
        ldb: usize,
        c: &mut CudaSlice<bf16>,
        ldc: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        let m_blas = n as i32;
        let n_blas = m as i32;
        let k_blas = k as i32;
        let lda_blas = lda as i32;
        let ldb_blas = ldb as i32;
        let ldc_blas = ldc as i32;

        unsafe {
            cublasGemmEx(
                *blas.handle(),
                cublasOperation_t::CUBLAS_OP_T, // Transpose of B (which is passed as A to GEMM because of swap)
                cublasOperation_t::CUBLAS_OP_N, // No Transpose of A (which is passed as B to GEMM)
                m_blas,
                n_blas,
                k_blas,
                &alpha as *const f32 as *const _,
                b.device_ptr(stream).0 as *const _, // B (Weight) passed first
                cudaDataType::CUDA_R_16BF,
                ldb_blas,                           // Stride for Weight
                a.device_ptr(stream).0 as *const _, // A (Input) passed second
                cudaDataType::CUDA_R_16BF,
                lda_blas, // Stride for Input
                &beta as *const f32 as *const _,
                c.device_ptr(stream).0 as *mut _,
                cudaDataType::CUDA_R_16BF,
                ldc_blas,
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            );
        }
        Ok(())
    }

    pub fn compute_logits(&mut self) -> Result<()> {
        let stream = self.device.default_stream();
        let vocab_size = self.config.vocab_size;
        let hidden_dim = self.config.hidden_size;

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
                self.buffers.logits.device_ptr_mut(&stream).0 as _,
                cudaDataType::CUDA_R_16BF,
                vocab_size as i32, // ldc
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            );
        }
        Ok(())
    }

    pub fn sample_token(&mut self, sampler: &mut Sampler) -> Result<u32> {
        self.compute_logits()?;

        // Access via public field since sampler is outside
        // But sampler doesn't expose temperature publicly?
        // Let's assume we can access it or use a method
        // Sampler struct definition in `sampler.rs`:
        // pub struct Sampler { rng, temperature, top_p, top_k }
        // Fields are private by default if not pub.
        // I need to check sampler.rs again.

        let stream = self.device.default_stream();

        if sampler.is_greedy() {
            self.cuda_functions.apply_argmax(
                &stream,
                &self.buffers.logits,
                &mut self.sample_indices_buffer,
                self.config.vocab_size,
            )?;
            let host_idx = stream.clone_dtoh(&self.sample_indices_buffer)?;
            return Ok(host_idx[0]);
        }

        // Top-P GPU Path
        // We use a random float for sampling
        use rand::Rng;
        let p_threshold = sampler.top_p();
        if p_threshold < 1.0 {
            let rand_val: f32 = rand::rng().random();
             self.cuda_functions.apply_sort_and_sample(
                &stream,
                &self.buffers.logits,
                &mut self.sample_indices_buffer,
                &mut self.sort_buffer,
                self.config.vocab_size,
                p_threshold,
                rand_val
            )?;
            let host_idx = stream.clone_dtoh(&self.sample_indices_buffer)?;
            return Ok(host_idx[0]);
        }

        // Fallback to CPU for other cases (Temp only or debug)
        let host_data = stream.clone_dtoh(&self.buffers.logits)?;
        let mut logits: Vec<f32> = host_data.into_iter().map(|x: bf16| x.to_f32()).collect();
        sampler.sample(&mut logits)
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
        let q_size_1d = hidden_dim;
        let k_size_1d = num_kv_heads * head_dim;
        let v_size_1d = num_kv_heads * head_dim;
        let qkv_output_dim = q_size_1d + k_size_1d + v_size_1d;

        unsafe {
            Self::matmul_bf16(
                stream,
                blas,
                seq_len,
                qkv_output_dim,
                hidden_dim,
                bufs.norm_buffer,
                &layer.qkv_proj,
                bufs.qkv_states,
                1.0,
                0.0,
            )?;
        }

        // 3. Serial Loop for Attention/RoPE/Cache
        for t in 0..seq_len {
            let current_pos = cache_pos + t;

            // For fused buffer: [seq_len, qkv_dim]
            // We need to slice Q, K, V parts for the current token (or sequence of tokens if we were doing flash attn, but here we loop t)
            // Wait, RoPE and logic below assume accessing data for ONE token at time 't' if seq_len > 1?
            // "bufs.qkv_states" holds [seq_len, Q+K+V]
            // RoPE expects: &mut q_states, q_offset

            // QKV layout in memory:
            // Row 0: [Q0... | K0... | V0...]
            // Row 1: [Q1... | K1... | V1...]

            let row_offset = t * qkv_output_dim;
            let q_offset = row_offset;
            let k_offset = row_offset + q_size_1d;
            let v_offset = row_offset + q_size_1d + k_size_1d;

            // We need to construct temporary slices to q_bias, k_bias, v_bias from qkv_bias
            // qkv_bias is [Q+K+V]
            let q_bias_view = layer.qkv_bias.slice(0..q_size_1d);
            let k_bias_view = layer.qkv_bias.slice(q_size_1d..q_size_1d + k_size_1d);
            let v_bias_view = layer.qkv_bias.slice(q_size_1d + k_size_1d..qkv_output_dim);

            // RoPE & KV Update & Attention
            // Note: RoPE modifies Q and K in place.
            {
                let mut qkv_view_mut = bufs.qkv_states.slice_mut(0..bufs.qkv_states.len());
                funcs.apply_rope(
                    rope,
                    stream,
                    &mut qkv_view_mut,
                    q_offset,
                    k_offset,
                    &q_bias_view,
                    &k_bias_view,
                    current_pos,
                    head_dim,
                    num_q_heads,
                    num_kv_heads,
                )?;
            }

            // KV Cache Update: needs to copy from K/V states to Cache
            let k_input_view = bufs.qkv_states.slice(k_offset..k_offset + k_size_1d);
            let v_input_view = bufs.qkv_states.slice(v_offset..v_offset + k_size_1d);

            kv_cache.update(
                stream,
                i,
                current_pos,
                &k_input_view,
                0,
                &v_input_view,
                0,
                &v_bias_view,
            )?;

            funcs.apply_flash_decoding(
                stream,
                bufs.att_output,
                t * hidden_dim,
                bufs.qkv_states,
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
                intermediate_size * 2,
                hidden_dim,
                bufs.norm_buffer,
                &layer.gate_up_proj,
                bufs.gate_up_states,
                1.0,
                0.0,
            )?;
        }

        // Activation: gate = silu(gate) * up
        // Currently apply_activation takes two tensors.
        // We can create views.
        // But stride? CudaSlice view is contiguous range.
        // Our 'gate_up_states' is [seq_len, 2*intermediate].
        // Row 0: [Gate0... | Up0...]
        // apply_activation is elementwise.
        // If we want to use the existing kernel which takes separate gate/up pointers,
        // passing `gate_up_states` with offset works IF the kernel respects stride/gap?
        // No, CudaSlice doesn't carry stride info usually in these kernels, it treats it as flat array.
        // IF seq_len > 1, gate_up_states memory is:
        // G0 U0 G1 U1 ...
        // We cannot get a contiguous slice of All Gs or All Us.
        // BUT, `matmul_bf16` produces result in RowMajor (conceptually)?
        // Wait, matmul output layout.
        // If we compute [Seq, Hidden] * [Hidden, 2*Inter]. Result is [Seq, 2*Inter].
        // Data: Row0(G0...U0...), Row1(G1...U1...).
        // So Gs and Us are interleaved per token.

        // Options:
        // 1. Write a fused "activation" kernel that takes [Seq, 2*Inter] and acts on it.
        // 2. Or, if seq_len=1 (Decode), they are contiguous.
        // 3. Current `apply_activation` kernel logic:
        //    It launches `silu_and_mul_kernel<<<grid, block>>>(gate, up, size)`.
        //    It assumes input is 1D array of size `size`.
        //    It computes gate[i] = silu(gate[i]) * up[i] for i in 0..size.
        //    This assumes gate[i] and up[i] are paired.
        //    If our memory is `G0[0]..G0[N-1] U0[0]..U0[N-1]`, then gate[0] pairs with up[0].
        //    This works perfectly for 1 token!

        // What about N tokens?
        // `G0...U0... | G1...U1...`
        // If we call activation on the whole buffer with `size = seq_len * intermediate_size`.
        // Thread i processes `buffer[i]` and `buffer[size + i]`? No.
        // The existing kernel expects separate arrays.

        // We MUST write a fused kernel or modify the existing one to handle interleaved structure?
        // OR better: Just treat the whole buffer as one array if the kernel supports `gate[i]` and `gate[i + offset]`.
        // But here `Up` is immediately after `Gate` for EACH token.
        // So `Up[i]` is at `Gate[i] + IntermediateSize`? NO.
        // If flattened:
        // Idx 0: G0_0.  Idx M: U0_0.
        // Idx 2M: G1_0.

        // So for token T, Gate starts at T*2*M, Up starts at T*2*M + M.
        // Distance is always M (intermediate_size).
        // BUT, Thread `i` (global index) usually maps to `i` in Gate and `i` in Up.
        // If we use standard kernel, it expects contiguous blocks.

        // Optimally, we need a kernel: `silu_and_mul_fused(ptr, mid_stride, total_elements)`?
        // Actually, easiest fix for now without writing new kernel:
        // Only works trivially for seq_len=1.
        // For seq_len > 1, we suffer from non-contiguous memory for Gate/Up separation if we want to treat them as independent large arrays.
        // However, `silu` is pointwise.
        // We can launch a kernel that processes `(gate, up)` pairs.
        // If we iterate over `seq_len * intermediate_size` elements.
        // For element `j` (0..total_elements), we need to find where `G_j` and `U_j` are.
        // `row = j / intermediate_size`. `col = j % intermediate_size`.
        // `G_ptr = row * (2 * intermediate) + col`.
        // `U_ptr = row * (2 * intermediate) + col + intermediate`.

        // This requires a custom kernel.
        // OR, we can split the matmul into 2 if we don't want to write kernel now.
        // But we want fusion.
        // Let's modify `src/kernels/activation.cu` to support interleaved (fused) buffer.
        // Actually, if we look at `apply_activation`.
        // We can just add `stride` parameter to it.

        // For now, to keep it simple and correct:
        // Use a new kernel entry point `silu_and_mul_fused`.
        // Input: `buffer` (contains Gate/Up interleaved).
        // Args: `rows`, `cols` (intermediate).
        // Thread x processes element `x` of Gate, and finds corresponding Up.
        // If 1D launch: idx `i`.
        // gate_idx = (i / cols) * 2 * cols + (i % cols).
        // up_idx = gate_idx + cols.

        // Activation: Fused Silu + Mul on interleaved buffer
        funcs.apply_activation_fused(stream, bufs.gate_up_states, seq_len, intermediate_size)?;

        unsafe {
            Self::matmul_bf16_strided(
                stream,
                blas,
                seq_len,
                hidden_dim,
                intermediate_size,
                bufs.gate_up_states,   // Input A
                2 * intermediate_size, // Lda (Stride of A is 2x Intermediate)
                &layer.down_proj,      // Input B
                intermediate_size,     // Ldb (Stride of B)
                bufs.hidden_states,
                hidden_dim, // Ldc
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
                qkv_states: &mut self.buffers.qkv_states,
                att_output: &mut self.buffers.att_output,
                gate_up_states: &mut self.buffers.gate_up_states,
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
        let mut qkv_states =
            stream.alloc_zeros::<bf16>(seq_len * (hidden_dim + 2 * num_kv_heads * head_dim))?;
        let mut att_output = stream.alloc_zeros::<bf16>(seq_len * hidden_dim)?;
        let mut gate_up_states = stream.alloc_zeros::<bf16>(seq_len * intermediate_size * 2)?;

        {
            let mut bufs = ForwardPassBuffers {
                hidden_states: &mut hidden_states,
                norm_buffer: &mut norm_buffer,
                qkv_states: &mut qkv_states,
                att_output: &mut att_output,
                gate_up_states: &mut gate_up_states,
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
