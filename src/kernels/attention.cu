#include <cuda_bf16.h>

// Online Softmax 算法需要维护的统计量
struct SoftmaxState {
    float m;     // max score
    float l;     // sum of exp
    float o[64]; // accumulated output (head_dim=64 for Qwen2.5-0.5B)
};

__device__ __forceinline__ float rope_correction(int pos, int dim_idx,
                                                 int head_dim) {
    // 这里如果还没做 RoPE，可以在这里做。
    // 但我们的模型已经在前面做了 Fused RoPE，所以这里接收到的 Q 和 K
    // 已经是旋转过的了。 保持简单，假设输入已经 RoPE 过。
    return 1.0f;
}

extern "C" __global__ void flash_decoding_kernel(
    __nv_bfloat16 *output,  // [num_q_heads, head_dim]
    const __nv_bfloat16 *q, // [num_q_heads, head_dim]
    const __nv_bfloat16
        *k_cache, // [num_layers, num_kv_heads, max_seq_len, head_dim]
    const __nv_bfloat16
        *v_cache, // [num_layers, num_kv_heads, max_seq_len, head_dim]
    int layer_idx, int num_q_heads, int num_kv_heads, int head_dim,
    int max_seq_len,
    int current_pos, // valid sequence length - 1 (index of current token)
    float sm_scale) {
    // 1. Grid Configuration
    // Grid: (num_q_heads, 1, 1) -> 每个 Block 处理一个 Q Head
    // Block: (128, 1, 1) -> Warp Reduction 需要

    int q_head_idx = blockIdx.x;
    int kv_head_idx =
        q_head_idx / (num_q_heads / num_kv_heads); // GQA Group Map
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // 2. 初始化局部状态
    float local_m = -1e20f;
    float local_l = 0.0f;
    float local_o[64]; // 寄存器存 Output 行
#pragma unroll
    for (int i = 0; i < 64; ++i)
        local_o[i] = 0.0f;

    // Load Q into shared memory?
    // Head Dim = 64 is small.
    // We can just load Q into registers or shared mem. Shared is safer for
    // broadcast.
    extern __shared__ float s_q[]; // Size: head_dim

    // 协作加载 Q 到 shared memory
    // Q: [num_q_heads, head_dim] stored linearly
    if (tid < head_dim) {
        s_q[tid] = __bfloat162float(q[q_head_idx * head_dim + tid]);
    }
    __syncthreads();

    // 3. 遍历 KV Cache
    // 每个 Block 遍历整个 Sequence
    // 这里的循环可以用 blockDim.x 进行跨步，也可以按 Block 分块加载
    // 简单的写法：每个线程负责计算一部分 Token 的 Score

    // K cache offset setup
    // K cache shape: [num_layers, num_kv_heads, max_seq_len, head_dim]
    // 我们的 index 计算必须和 update kernel 保持一致
    // 假设是 RowMajor，则 offset = l * (NH * S * D) + h * (S * D) + s * D
    long long layer_offset =
        (long long)layer_idx * num_kv_heads * max_seq_len * head_dim;
    long long kv_head_offset = (long long)kv_head_idx * max_seq_len * head_dim;
    long long base_offset = layer_offset + kv_head_offset;

    // 还有优化的空间：Tiling K Loaded.
    // 这里用最简单的逻辑：每个线程独立计算它负责的时间步 (time steps)

    for (int t = tid; t <= current_pos; t += blockDim.x) {
        // --- Step A: Calculate Score S_t = Q . K_t ---
        float score = 0.0f;
        long long k_ptr = base_offset + (long long)t * head_dim;

#pragma unroll
        for (int i = 0; i < 64; ++i) { // head_dim = 64
            float ki = __bfloat162float(k_cache[k_ptr + i]);
            score += s_q[i] * ki;
        }
        score *= sm_scale;

        // --- Step B: Online Softmax Update ---
        // m_new = max(m_prev, score)
        // l_new = l_prev * exp(m_prev - m_new) + exp(score - m_new)
        // o_new = o_prev * exp(m_prev - m_new) + v_t * exp(score - m_new)

        float m_prev = local_m;
        local_m = fmaxf(m_prev, score);
        float exp_m = expf(m_prev - local_m);
        float exp_s = expf(score - local_m);

        local_l = local_l * exp_m + exp_s;

        long long v_ptr = base_offset + (long long)t * head_dim;
#pragma unroll
        for (int i = 0; i < 64; ++i) {
            float vi = __bfloat162float(v_cache[v_ptr + i]);
            local_o[i] = local_o[i] * exp_m + vi * exp_s;
        }
    }

    // 4. Block Reduction (reduce across threads)
    // 现在的状态：每个线程都有它那部分 tokens 的局部聚合结果 (local_m, local_l,
    // local_o) 我们需要把它们合并成全局的。

    // 这是一个略微复杂的 Online Softmax Reduction
    // Merge (m1, l1, o1) and (m2, l2, o2):
    // m_new = max(m1, m2)
    // l_new = l1 * exp(m1 - m_new) + l2 * exp(m2 - m_new)
    // o_new = o1 * ... + o2 * ...

    // Use Shared Memory for reduction?
    // Just warp reduce then block reduce.

    // Try simple approach: Shuffle Down
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_m = __shfl_down_sync(0xffffffff, local_m, offset);
        float other_l = __shfl_down_sync(0xffffffff, local_l, offset);

        float new_m = fmaxf(local_m, other_m);
        float scale_self = expf(local_m - new_m);
        float scale_other = expf(other_m - new_m);

        local_m = new_m;
        local_l = local_l * scale_self + other_l * scale_other;

        for (int i = 0; i < 64; i++) {
            float other_o = __shfl_down_sync(0xffffffff, local_o[i], offset);
            local_o[i] = local_o[i] * scale_self + other_o * scale_other;
        }
    }

    // Now lane 0 of each warp has the warp-result.
    // Store to shared memory to let Warp 0 collect them.
    static __shared__ float s_m[4]; // 128 threads = 4 warps
    static __shared__ float s_l[4];
    static __shared__ float s_o[64][4]; // [dim][warp]

    if (lane_id == 0) {
        s_m[warp_id] = local_m;
        s_l[warp_id] = local_l;
        for (int i = 0; i < 64; ++i)
            s_o[i][warp_id] = local_o[i];
    }
    __syncthreads();

    // Warp 0 does final reduction
    if (warp_id == 0) {
        // Reload local state from Warp 0's slot
        local_m = s_m[0];
        local_l = s_l[0];
        for (int i = 0; i < 64; ++i)
            local_o[i] = s_o[i][0];

        // Merge Warp 1, 2, 3
        for (int w = 1; w < 4; ++w) { // Assuming 128 threads -> 4 warps
            float other_m = s_m[w];
            float other_l = s_l[w];
            float new_m = fmaxf(local_m, other_m);
            float scale_self = expf(local_m - new_m);
            float scale_other = expf(other_m - new_m);

            local_m = new_m;
            local_l = local_l * scale_self + other_l * scale_other;

            for (int i = 0; i < 64; ++i) {
                local_o[i] = local_o[i] * scale_self + s_o[i][w] * scale_other;
            }
        }

        // Final normalization: O = O / L
        for (int i = 0; i < 64; ++i) {
            local_o[i] /= local_l;
        }

        // 5. Write to Global Memory (Lane 0 ~ 31 write output indices 0~31 and
        // 32~63?) No, current logic: only Lane 0 of Warp 0 has the full array
        // local_o. We can broadcast it or just let Lane 0 write (slow but ok
        // for 64 floats). Better: store to shared memory and write coalesced.
        for (int i = 0; i < 64; ++i)
            s_q[i] = local_o[i]; // reuse s_q buffer
    }

    __syncthreads();

    // Coalesced write logic:
    // We have 64 elements to write.
    // Thread 0..63 write one element each.
    if (tid < 64) {
        output[q_head_idx * head_dim + tid] = __float2bfloat16(s_q[tid]);
    }
}
