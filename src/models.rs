use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::bf16;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use serde::Deserialize;
use std::{fs::File, path::Path, sync::Arc};

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
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
    // TODO: Add other weights
}

pub struct Qwen2Model {
    pub embed_tokens: CudaSlice<bf16>,
    pub layers: Vec<LayerWeights>,
    pub final_norm: CudaSlice<bf16>,
    pub rope: RopeCache,
}

pub struct RopeCache {
    pub cos: CudaSlice<f32>,
    pub sin: CudaSlice<f32>,
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

        Ok(Qwen2Model {
            embed_tokens,
            layers,
            final_norm,
            rope,
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
