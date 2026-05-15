//! MUCD (Multi-source Unsupervised Contrastive Decoding) Naive 实现
//!
//! 核心公式:
//!   score = (1 + α + β) * log_p_final - α * log_p_mid - β * log_p_aux
//!
//! 其中:
//!   α = H_mid / (H_final + H_mid + ε)
//!   β = H_aux / (H_final + H_aux + ε)

use anyhow::Result;
use tokenizers::Tokenizer;

use crate::models::Qwen2Model;
use crate::streamer::Streamer;

/// MUCD 解码器，持有主模型和辅助模型
pub struct MucdDecoder {
    pub main_model: Qwen2Model,
    pub aux_model: Qwen2Model,
    /// 候选 premature 层索引（低层偶数层）
    candidate_layers: Vec<usize>,
    /// Relative top 过滤参数
    relative_top: f32,
}

impl MucdDecoder {
    /// 创建 MUCD 解码器
    /// main_model: 主模型 (例如 deepseek-coder-6.7b-base)
    /// aux_model: 辅助模型 (例如 deepseek-coder-1.3b-base)
    /// relative_top: relative top 过滤参数 (默认 0.1)
    pub fn new(main_model: Qwen2Model, aux_model: Qwen2Model, relative_top: f32) -> Self {
        // 验证 vocab_size 匹配
        assert_eq!(
            main_model.config.vocab_size, aux_model.config.vocab_size,
            "主模型和辅助模型的 vocab_size 不匹配: {} vs {}",
            main_model.config.vocab_size, aux_model.config.vocab_size
        );

        // 计算候选 premature 层 (low 策略)
        let final_layer = main_model.config.num_hidden_layers;
        let start_layer = if !main_model.config.tie_word_embeddings {
            0
        } else if final_layer > 2 {
            2
        } else if final_layer == 2 {
            1
        } else {
            0
        };

        let half = final_layer / 2;
        let candidate_layers: Vec<usize> = if start_layer >= half {
            vec![start_layer]
        } else if final_layer <= 40 {
            (start_layer..half).step_by(2).collect()
        } else {
            (start_layer..20).step_by(2).collect()
        };

        println!("MUCD 候选 premature 层: {:?}", candidate_layers);
        println!("MUCD relative_top: {}", relative_top);

        MucdDecoder {
            main_model,
            aux_model,
            candidate_layers,
            relative_top,
        }
    }

    /// MUCD Naive 生成
    /// prompt_ids: 输入 token ids
    /// max_new_tokens: 最大生成 token 数
    /// tokenizer: 用于解码的 tokenizer
    /// token_callback: 每生成一个 token 后的回调函数，返回 false 停止生成
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        tokenizer: &Tokenizer,
        mut token_callback: impl FnMut(&str, &MucdStepDebugInfo) -> bool,
    ) -> Result<()> {
        let mut main_cache_pos = 0;
        let mut aux_cache_pos = 0;
        let mut next_token_id: u32;
        let mut streamer = Streamer::new(tokenizer);

        let eos_token_id = self.main_model.config.eos_token_id;
        let bos_token_id = self.main_model.config.bos_token_id;

        // ===== Prefill 阶段 =====
        if prompt_ids.is_empty() {
            return Ok(());
        }

        // 主模型 prefill
        self.main_model.forward(prompt_ids, main_cache_pos)?;
        main_cache_pos += prompt_ids.len();

        // 辅助模型 prefill
        self.aux_model.forward(prompt_ids, aux_cache_pos)?;
        aux_cache_pos += prompt_ids.len();

        // 第一个 token: 用 MUCD 算法
        let final_logits = self.main_model.get_logits_f32()?;
        let _aux_logits = self.aux_model.get_logits_f32()?;

        // Prefill 阶段的 premature logits 需要特殊处理
        // 由于 prefill 时 seq_len > 1，forward_with_hidden_states 不支持
        // 第一个 token 直接使用 final logits 做 argmax
        next_token_id = argmax(&final_logits);

        let debug_info = MucdStepDebugInfo {
            step: 0,
            alpha: 0.0,
            beta: 0.0,
            h_final: 0.0,
            h_mid: 0.0,
            h_aux: 0.0,
            js_mid: 0.0,
            js_aux: 0.0,
            selected_layer: 0,
        };

        // 检查停止条件
        if next_token_id == eos_token_id || next_token_id == bos_token_id {
            return Ok(());
        }

        if let Some(text) = streamer.put(next_token_id) {
            if !token_callback(&text, &debug_info) {
                return Ok(());
            }
        }

        // ===== Decode 循环 =====
        for step in 0..max_new_tokens {
            // 主模型：带 hidden states 提取的 forward
            let hidden_states_collection = self.main_model.forward_with_hidden_states(
                &[next_token_id],
                main_cache_pos,
                &self.candidate_layers,
            )?;
            main_cache_pos += 1;

            // 辅助模型：普通 forward
            self.aux_model.forward(&[next_token_id], aux_cache_pos)?;
            aux_cache_pos += 1;

            // 获取各模型的 logits (f32, CPU)
            let final_logits = self.main_model.get_logits_f32()?;
            let aux_logits = self.aux_model.get_logits_f32()?;

            // 计算 premature layers 的 logits
            let premature_logits_all = self.main_model.compute_premature_logits_batched(
                &hidden_states_collection,
                self.candidate_layers.len(),
            )?;

            // MUCD Naive 算法
            let (score, debug_info) = mucd_naive_score(
                &self.candidate_layers,
                &premature_logits_all,
                &aux_logits,
                &final_logits,
                self.relative_top,
                step + 1,
            );

            // Argmax 采样
            next_token_id = argmax(&score);

            // 检查停止条件
            if next_token_id == eos_token_id || next_token_id == bos_token_id {
                break;
            }

            if let Some(text) = streamer.put(next_token_id) {
                if !token_callback(&text, &debug_info) {
                    break;
                }
            }
        }

        Ok(())
    }

    /// 重置两个模型的 KV Cache
    pub fn reset(&mut self) {
        self.main_model.reset_kv_cache();
        self.aux_model.reset_kv_cache();
    }
}

/// MUCD 每步的调试信息
#[derive(Debug, Clone)]
pub struct MucdStepDebugInfo {
    pub step: usize,
    pub alpha: f32,
    pub beta: f32,
    pub h_final: f32,
    pub h_mid: f32,
    pub h_aux: f32,
    pub js_mid: f32,
    pub js_aux: f32,
    pub selected_layer: usize,
}

// ==================== MUCD 核心算法 (CPU f32) ====================

/// MUCD Naive 评分
/// 返回 (contrastive_scores, debug_info)
fn mucd_naive_score(
    candidate_layers: &[usize],
    premature_logits_all: &[Vec<f32>],
    aux_logits: &[f32],
    final_logits: &[f32],
    relative_top: f32,
    step: usize,
) -> (Vec<f32>, MucdStepDebugInfo) {
    // 1. 选择 JS 散度最大的 premature 层
    let (mid_logits, js_mid, selected_layer_idx) =
        select_premature_layer(candidate_layers, premature_logits_all, final_logits);

    // 2. 计算 JS_aux
    let js_aux = calc_js_divergence(final_logits, aux_logits);

    // 3. Relative Top 过滤
    let (final_log_probs, mid_log_probs, aux_log_probs, mask) =
        relative_top_filter(final_logits, &mid_logits, aux_logits, relative_top);

    // 4. 计算熵
    let h_final = calc_entropy(&final_log_probs, &mask);
    let h_mid = calc_entropy(&mid_log_probs, &mask);
    let h_aux = calc_entropy(&aux_log_probs, &mask);

    // 5. 计算权重
    let alpha = h_mid / (h_final + h_mid + 1e-8);
    let beta = h_aux / (h_final + h_aux + 1e-8);

    // 6. 计算对比分数
    let vocab_size = final_logits.len();
    let mut score = vec![0.0f32; vocab_size];
    for i in 0..vocab_size {
        score[i] = (1.0 + alpha + beta) * final_log_probs[i]
            - alpha * mid_log_probs[i]
            - beta * aux_log_probs[i];
    }

    let debug_info = MucdStepDebugInfo {
        step,
        alpha,
        beta,
        h_final,
        h_mid,
        h_aux,
        js_mid,
        js_aux,
        selected_layer: candidate_layers[selected_layer_idx],
    };

    (score, debug_info)
}

/// 选择 JS 散度最大的 premature 层
/// 返回 (选中层的 logits, JS 散度值, 层在候选列表中的索引)
fn select_premature_layer(
    candidate_layers: &[usize],
    premature_logits_all: &[Vec<f32>],
    final_logits: &[f32],
) -> (Vec<f32>, f32, usize) {
    if candidate_layers.len() == 1 {
        let js = calc_js_divergence(final_logits, &premature_logits_all[0]);
        return (premature_logits_all[0].clone(), js, 0);
    }

    let mut max_js = f32::NEG_INFINITY;
    let mut max_idx = 0;

    for (idx, premature_logits) in premature_logits_all.iter().enumerate() {
        let js = calc_js_divergence(final_logits, premature_logits);
        if js > max_js {
            max_js = js;
            max_idx = idx;
        }
    }

    (premature_logits_all[max_idx].clone(), max_js, max_idx)
}

/// 计算两个 logits 分布之间的 JS 散度
/// 使用 mean(-1) 归一化（与 Python 实现一致）
fn calc_js_divergence(p_logits: &[f32], q_logits: &[f32]) -> f32 {
    let vocab_size = p_logits.len();

    // softmax(p) 和 softmax(q)
    let p_softmax = softmax(p_logits);
    let q_softmax = softmax(q_logits);

    // 平均分布 M = 0.5 * (P + Q)
    let mut avg: Vec<f32> = vec![0.0; vocab_size];
    for i in 0..vocab_size {
        avg[i] = 0.5 * (p_softmax[i] + q_softmax[i]);
    }

    // log_softmax(p) 和 log_softmax(q)
    let p_log_softmax = log_softmax(p_logits);
    let q_log_softmax = log_softmax(q_logits);

    // KL(P || M) = sum(M * (log(M) - log_softmax(P)))
    // 使用与 PyTorch kl_div 一致的方式: kl_div(log_input, target) = target * (log(target) - log_input)
    // mean(-1) 表示对 vocab 维度取平均
    let mut kl1 = 0.0f32;
    let mut kl2 = 0.0f32;
    for i in 0..vocab_size {
        if avg[i] > 0.0 {
            let log_avg = avg[i].ln();
            kl1 += avg[i] * (log_avg - p_log_softmax[i]);
            kl2 += avg[i] * (log_avg - q_log_softmax[i]);
        }
    }

    // mean over vocab dimension
    kl1 /= vocab_size as f32;
    kl2 /= vocab_size as f32;

    0.5 * (kl1 + kl2)
}

/// Relative Top 过滤
/// 返回 (final_log_probs, mid_log_probs, aux_log_probs, mask)
/// mask[i] = true 表示该位置被过滤掉了
fn relative_top_filter(
    final_logits: &[f32],
    mid_logits: &[f32],
    aux_logits: &[f32],
    relative_top: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<bool>) {
    let vocab_size = final_logits.len();
    let filter_value = f32::NEG_INFINITY;
    let base_filter_value = -1e-3_f32;

    // log_softmax
    let mut final_log_probs = log_softmax(final_logits);
    let mut mid_log_probs = log_softmax(mid_logits);
    let mut aux_log_probs = log_softmax(aux_logits);

    // 找到 final_log_probs 的最大值
    let max_log_prob = final_log_probs
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    // 阈值 = max + log(relative_top)
    let threshold = max_log_prob + relative_top.ln();

    // 确保至少保留 1 个 token (min_tokens_to_keep = 1)
    // 找到排序后第一个 token 的 log_prob
    let mut sorted_log_probs = final_log_probs.clone();
    sorted_log_probs.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let min_thresh = sorted_log_probs[0]; // 至少保留最大的
    let effective_threshold = threshold.min(min_thresh);

    // 过滤
    let mut mask = vec![false; vocab_size];
    for i in 0..vocab_size {
        if final_log_probs[i] < effective_threshold {
            mid_log_probs[i] = base_filter_value;
            aux_log_probs[i] = base_filter_value;
            final_log_probs[i] = filter_value;
            mask[i] = true;
        }
    }

    (final_log_probs, mid_log_probs, aux_log_probs, mask)
}

/// 计算过滤后分布的熵
fn calc_entropy(log_probs: &[f32], mask: &[bool]) -> f32 {
    let vocab_size = log_probs.len();

    // 将 log_probs 转为 probs，被 mask 的位置为 0
    let mut probs: Vec<f32> = vec![0.0; vocab_size];
    let mut sum = 0.0f32;
    for i in 0..vocab_size {
        if !mask[i] {
            probs[i] = log_probs[i].exp();
            sum += probs[i];
        }
    }

    // 重新归一化
    if sum > 1e-10 {
        for i in 0..vocab_size {
            probs[i] /= sum;
        }
    }

    // 计算熵 H = -sum(p * log(p))
    let mut entropy = 0.0f32;
    for i in 0..vocab_size {
        if probs[i] > 1e-10 {
            entropy -= probs[i] * probs[i].ln();
        }
    }

    entropy
}

// ==================== 辅助函数 ====================

/// Softmax
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut result: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = result.iter().sum();
    for x in result.iter_mut() {
        *x /= sum;
    }
    result
}

/// Log-softmax
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&x| (x - max_val).exp()).sum();
    let log_sum_exp = max_val + sum_exp.ln();
    logits.iter().map(|&x| x - log_sum_exp).collect()
}

/// Argmax
fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap()
}

// ==================== 单元测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // softmax 应该是单调递增的
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_log_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let log_probs = log_softmax(&logits);
        let probs = softmax(&logits);
        for i in 0..3 {
            assert!((log_probs[i] - probs[i].ln()).abs() < 1e-5);
        }
    }

    #[test]
    fn test_js_divergence_same_distribution() {
        // 相同分布的 JS 散度应该接近 0
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let js = calc_js_divergence(&logits, &logits);
        assert!(
            js < 1e-6,
            "相同分布的 JS 散度应该接近 0，但得到 {}",
            js
        );
    }

    #[test]
    fn test_js_divergence_different_distributions() {
        // 不同分布的 JS 散度应该大于 0
        let p = vec![1.0, 0.0, 0.0, 0.0];
        let q = vec![0.0, 0.0, 0.0, 1.0];
        let js = calc_js_divergence(&p, &q);
        assert!(js > 0.0, "不同分布的 JS 散度应该大于 0");
    }

    #[test]
    fn test_js_divergence_symmetry() {
        // JS 散度应该是对称的
        let p = vec![1.0, 2.0, 3.0];
        let q = vec![3.0, 1.0, 2.0];
        let js_pq = calc_js_divergence(&p, &q);
        let js_qp = calc_js_divergence(&q, &p);
        assert!(
            (js_pq - js_qp).abs() < 1e-6,
            "JS 散度应该是对称的: {} vs {}",
            js_pq,
            js_qp
        );
    }

    #[test]
    fn test_calc_entropy_uniform() {
        // 均匀分布的熵应该是 ln(n)
        let n = 4;
        let log_prob = -(n as f32).ln();
        let log_probs = vec![log_prob; n];
        let mask = vec![false; n];
        let entropy = calc_entropy(&log_probs, &mask);
        let expected = (n as f32).ln();
        assert!(
            (entropy - expected).abs() < 1e-4,
            "均匀分布的熵应该是 ln({}), 得到 {}",
            n,
            entropy
        );
    }

    #[test]
    fn test_calc_entropy_peaked() {
        // 高度集中分布的熵应该很低
        let mut log_probs = vec![-100.0; 100];
        log_probs[0] = 0.0; // 几乎所有概率集中在第一个 token
        let mask = vec![false; 100];
        let entropy = calc_entropy(&log_probs, &mask);
        assert!(
            entropy < 0.1,
            "高度集中分布的熵应该很低，得到 {}",
            entropy
        );
    }

    #[test]
    fn test_relative_top_filter() {
        // 构造简单 logits
        let final_logits = vec![10.0, 5.0, 0.0, -5.0, -10.0];
        let mid_logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let aux_logits = vec![5.0, 4.0, 3.0, 2.0, 1.0];

        let (f_lp, m_lp, a_lp, mask) =
            relative_top_filter(&final_logits, &mid_logits, &aux_logits, 0.1);

        // 最大的 token 不应该被过滤
        assert!(!mask[0], "概率最高的 token 不应被过滤");

        // 被过滤的 final log_probs 应该是 -inf
        for i in 0..5 {
            if mask[i] {
                assert!(f_lp[i] == f32::NEG_INFINITY);
                assert!((m_lp[i] - (-1e-3)).abs() < 1e-6);
                assert!((a_lp[i] - (-1e-3)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_argmax() {
        let logits = vec![1.0, 3.0, 2.0, 0.0];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn test_select_premature_layer_single() {
        let candidate_layers = vec![2];
        let premature_logits = vec![vec![1.0, 2.0, 3.0]];
        let final_logits = vec![3.0, 2.0, 1.0];
        let (_, js, idx) =
            select_premature_layer(&candidate_layers, &premature_logits, &final_logits);
        assert_eq!(idx, 0);
        assert!(js > 0.0);
    }

    #[test]
    fn test_select_premature_layer_multiple() {
        // 第二个候选层与 final 差异更大，应该被选中
        let candidate_layers = vec![2, 4, 6];
        let final_logits = vec![10.0, 0.0, 0.0, 0.0];
        let premature_logits = vec![
            vec![9.0, 0.1, 0.1, 0.1],  // 与 final 很接近
            vec![0.0, 0.0, 0.0, 10.0], // 与 final 差异最大
            vec![8.0, 0.5, 0.5, 0.5],  // 中等差异
        ];
        let (_, _, idx) =
            select_premature_layer(&candidate_layers, &premature_logits, &final_logits);
        assert_eq!(idx, 1, "应该选择与 final 差异最大的层");
    }

    #[test]
    fn test_mucd_naive_score_basic() {
        // 基本测试: 确保不 panic 且输出合理
        let candidate_layers = vec![2, 4];
        let final_logits = vec![5.0, 3.0, 1.0, 0.5];
        let premature_logits = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4.0, 3.0, 2.0, 1.0],
        ];
        let aux_logits = vec![2.0, 2.0, 2.0, 2.0];

        let (score, debug) =
            mucd_naive_score(&candidate_layers, &premature_logits, &aux_logits, &final_logits, 0.1, 1);

        assert_eq!(score.len(), 4);
        assert!(debug.alpha >= 0.0 && debug.alpha <= 1.0);
        assert!(debug.beta >= 0.0 && debug.beta <= 1.0);

        // argmax 应该能正常工作
        let token = argmax(&score);
        assert!(token < 4);
    }
}
