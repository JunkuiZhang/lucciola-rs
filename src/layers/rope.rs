use anyhow::Result;
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

pub struct RopeCache {
    pub cos: CudaSlice<f32>,
    pub sin: CudaSlice<f32>,
}

impl RopeCache {
    pub fn new(
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
