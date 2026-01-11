#include <cuda_bf16.h>

template <int HEAD_DIM>
__device__ void flash_decoding_impl(
    __nv_bfloat16 *output,  // [num_q_heads, head_dim]
    const __nv_bfloat16 *q, // [num_q_heads, head_dim]
    const __nv_bfloat16
        *k_cache, // [num_layers, num_kv_heads, max_seq_len, head_dim]
    const __nv_bfloat16
        *v_cache, // [num_layers, num_kv_heads, max_seq_len, head_dim]
    int layer_idx, int num_q_heads, int num_kv_heads,
    int head_dim_rt, // runtime head_dim (should match HEAD_DIM)
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
    float local_o[HEAD_DIM]; // 寄存器存 Output 行
#pragma unroll
    for (int i = 0; i < HEAD_DIM; ++i)
        local_o[i] = 0.0f;

    // Load Q into shared memory?
    // We can just load Q into registers or shared mem. Shared is safer for
    // broadcast.
    extern __shared__ float s_q[]; // Size: HEAD_DIM

    // 协作加载 Q 到 shared memory
    // Q: [num_q_heads, head_dim] stored linearly
    if (tid < HEAD_DIM) {
        s_q[tid] = __bfloat162float(q[q_head_idx * HEAD_DIM + tid]);
    }
    __syncthreads();

    // 3. 遍历 KV Cache
    // K cache offset setup
    // K cache shape: [num_layers, num_kv_heads, max_seq_len, head_dim]
    // 我们的 index 计算必须和 update kernel 保持一致
    // 假设是 RowMajor，则 offset = l * (NH * S * D) + h * (S * D) + s * D
    long long layer_offset =
        (long long)layer_idx * num_kv_heads * max_seq_len * HEAD_DIM;
    long long kv_head_offset = (long long)kv_head_idx * max_seq_len * HEAD_DIM;
    long long base_offset = layer_offset + kv_head_offset;

    // 每个线程独立计算它负责的时间步 (time steps)
    for (int t = tid; t <= current_pos; t += blockDim.x) {
        // --- Step A: Calculate Score S_t = Q . K_t ---
        float score = 0.0f;
        long long k_ptr = base_offset + (long long)t * HEAD_DIM;

#pragma unroll
        for (int i = 0; i < HEAD_DIM; ++i) {
            float ki = __bfloat162float(k_cache[k_ptr + i]);
            score += s_q[i] * ki;
        }
        score *= sm_scale;

        // --- Step B: Online Softmax Update ---
        float m_prev = local_m;
        local_m = fmaxf(m_prev, score);
        float exp_m = expf(m_prev - local_m);
        float exp_s = expf(score - local_m);

        local_l = local_l * exp_m + exp_s;

        long long v_ptr = base_offset + (long long)t * HEAD_DIM;
#pragma unroll
        for (int i = 0; i < HEAD_DIM; ++i) {
            float vi = __bfloat162float(v_cache[v_ptr + i]);
            local_o[i] = local_o[i] * exp_m + vi * exp_s;
        }
    }

    // 4. Block Reduction (reduce across threads)

    // Try simple approach: Shuffle Down
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_m = __shfl_down_sync(0xffffffff, local_m, offset);
        float other_l = __shfl_down_sync(0xffffffff, local_l, offset);

        float new_m = fmaxf(local_m, other_m);
        float scale_self = expf(local_m - new_m);
        float scale_other = expf(other_m - new_m);

        local_m = new_m;
        local_l = local_l * scale_self + other_l * scale_other;

        for (int i = 0; i < HEAD_DIM; i++) {
            float other_o = __shfl_down_sync(0xffffffff, local_o[i], offset);
            local_o[i] = local_o[i] * scale_self + other_o * scale_other;
        }
    }

    // Now lane 0 of each warp has the warp-result.
    // Store to shared memory to let Warp 0 collect them.
    static __shared__ float s_m[4]; // 128 threads = 4 warps
    static __shared__ float s_l[4];
    static __shared__ float s_o[HEAD_DIM][4]; // [dim][warp]

    if (lane_id == 0) {
        s_m[warp_id] = local_m;
        s_l[warp_id] = local_l;
        for (int i = 0; i < HEAD_DIM; ++i)
            s_o[i][warp_id] = local_o[i];
    }
    __syncthreads();

    // Warp 0 does final reduction
    if (warp_id == 0) {
        // Reload local state from Warp 0's slot
        local_m = s_m[0];
        local_l = s_l[0];
        for (int i = 0; i < HEAD_DIM; ++i)
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

            for (int i = 0; i < HEAD_DIM; ++i) {
                local_o[i] = local_o[i] * scale_self + s_o[i][w] * scale_other;
            }
        }

        // Final normalization: O = O / L
        for (int i = 0; i < HEAD_DIM; ++i) {
            local_o[i] /= local_l;
        }

        // 5. Write to Global Memory
        for (int i = 0; i < HEAD_DIM; ++i)
            s_q[i] = local_o[i]; // reuse s_q buffer
    }

    __syncthreads();

    // Coalesced write logic:
    if (tid < HEAD_DIM) {
        output[q_head_idx * HEAD_DIM + tid] = __float2bfloat16(s_q[tid]);
    }
}

extern "C" __global__ void
flash_decoding_kernel_64(__nv_bfloat16 *output, const __nv_bfloat16 *q,
                         const __nv_bfloat16 *k_cache,
                         const __nv_bfloat16 *v_cache, int layer_idx,
                         int num_q_heads, int num_kv_heads, int head_dim,
                         int max_seq_len, int current_pos, float sm_scale) {
    flash_decoding_impl<64>(output, q, k_cache, v_cache, layer_idx, num_q_heads,
                            num_kv_heads, head_dim, max_seq_len, current_pos,
                            sm_scale);
}

extern "C" __global__ void
flash_decoding_kernel_128(__nv_bfloat16 *output, const __nv_bfloat16 *q,
                          const __nv_bfloat16 *k_cache,
                          const __nv_bfloat16 *v_cache, int layer_idx,
                          int num_q_heads, int num_kv_heads, int head_dim,
                          int max_seq_len, int current_pos, float sm_scale) {
    flash_decoding_impl<128>(output, q, k_cache, v_cache, layer_idx,
                             num_q_heads, num_kv_heads, head_dim, max_seq_len,
                             current_pos, sm_scale);
}
