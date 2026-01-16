use anyhow::Result;
use rand::distr::Uniform;
use rand::{Rng, SeedableRng, rngs::StdRng};

pub struct Sampler {
    rng: StdRng,
    temperature: f32,
    top_p: f32,
    top_k: usize,
}

impl Sampler {
    pub fn new(seed: Option<u64>, temperature: f32, top_p: f32, top_k: usize) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };
        let top_k = if top_k == 0 { 1024 } else { top_k };
        Self {
            rng,
            temperature,
            top_p,
            top_k,
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }

    pub fn top_p(&self) -> f32 {
        self.top_p
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> Result<u32> {
        // 1. Apply Temperature
        if self.temperature == 0.0 {
            // Greedy
            return Ok(self.argmax(logits));
        }

        // Scale by temperature
        let inv_temp = 1.0 / self.temperature;
        for p in logits.iter_mut() {
            *p *= inv_temp;
        }

        // 2. Apply Softmax
        self.softmax(logits);

        // 3. Top-K Sampling
        // Filter out indices that are not in top-k, set their prob to 0
        // Then re-normalize? Or just sort and pick.
        // Efficient way: Partial sort indices by probability.

        let mut indices: Vec<usize> = (0..logits.len()).collect();

        // Sort indices by probability descending
        // For performance in production, we should use `select_nth_unstable` logic but `sort_by` is easier for now.
        indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());

        // 4. Top-P (Nucleus) Sampling
        let mut cumulative_prob = 0.0;
        let mut cutoff_index = logits.len();

        for (i, &idx) in indices.iter().enumerate() {
            cumulative_prob += logits[idx];
            if cumulative_prob > self.top_p {
                cutoff_index = i + 1;
                break;
            }
        }

        // Also apply Top-K restriction if set
        if self.top_k > 0 {
            cutoff_index = cutoff_index.min(self.top_k);
        }

        // 5. Select
        let forced_top_index = indices[0]; // Backup in case randomization fails or p=0

        let r: f32 = self.rng.sample(Uniform::new(0.0, 1.0).unwrap());
        let mut cdf = 0.0;

        // Re-normalize chosen chunk
        let mut chunk_prob_sum = 0.0;
        for i in 0..cutoff_index {
            chunk_prob_sum += logits[indices[i]];
        }

        for i in 0..cutoff_index {
            let idx = indices[i];
            let prob = logits[idx] / chunk_prob_sum; // renormalization
            cdf += prob;
            if r < cdf {
                return Ok(idx as u32);
            }
        }

        Ok(forced_top_index as u32)
    }

    fn argmax(&self, logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index as u32)
            .unwrap()
    }

    fn softmax(&self, logits: &mut [f32]) {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum_exp = 0.0;
        for p in logits.iter_mut() {
            *p = (*p - max_logit).exp();
            sum_exp += *p;
        }
        for p in logits.iter_mut() {
            *p /= sum_exp;
        }
    }
}
