#include <cuda_bf16.h>

// 辅助函数：BF16 点积
__device__ __inline__ float dot_product(const __nv_bfloat16 *a,
                                        const __nv_bfloat16 *b, int dim) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        sum += __bfloat162float(a[i]) * __bfloat162float(b[i]);
    }
    return sum;
}

extern "C" __global__ void
gqa_attention_kernel(__nv_bfloat16 *out,           // 输出 [14, 64]
                     const __nv_bfloat16 *q,       // 当前 Q [14, 64]
                     const __nv_bfloat16 *k_cache, // K 缓存 [24, 2, 32768, 64]
                     const __nv_bfloat16 *v_cache, // V 缓存 [24, 2, 32768, 64]
                     int layer_id, int pos_id,
                     int num_q_heads,  // 14
                     int num_kv_heads, // 2
                     int head_dim,     // 64
                     int max_seq_len,  // 32768
                     float scale       // 1.0 / sqrt(64)
) {
    int q_head_idx = blockIdx.x;      // 0..13
    int kv_head_idx = q_head_idx / 7; // GQA 映射：每 7 个 Q 共享 1 个 KV
    int tid = threadIdx.x;

    // 共享内存用于存储所有历史位置的得分 (Scores)
    // 注意：如果 pos_id 非常大，建议用全局内存或更高级的 FlashDecoding
    extern __shared__ float shared_scores[];

    // 1. 计算当前 Q Head 与所有历史 K 的点积得分
    const __nv_bfloat16 *my_q = q + q_head_idx * head_dim;

    for (int p = tid; p <= pos_id; p += blockDim.x) {
        // 定位 K 缓存中的位置
        long long k_offset =
            (long long)layer_id * num_kv_heads * max_seq_len * head_dim +
            (long long)kv_head_idx * max_seq_len * head_dim +
            (long long)p * head_dim;
        const __nv_bfloat16 *my_k = k_cache + k_offset;

        // 计算点积
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += __bfloat162float(my_q[d]) * __bfloat162float(my_k[d]);
        }
        shared_scores[p] = score * scale;
    }
    __syncthreads();

    // 2. Softmax (简易版：寻找最大值并求和)
    float max_s = -1e20f;
    if (tid == 0) {
        for (int p = 0; p <= pos_id; ++p)
            max_s = fmaxf(max_s, shared_scores[p]);
        float sum_e = 0.0f;
        for (int p = 0; p <= pos_id; ++p) {
            shared_scores[p] = expf(shared_scores[p] - max_s);
            sum_e += shared_scores[p];
        }
        for (int p = 0; p <= pos_id; ++p)
            shared_scores[p] /= sum_e;
    }
    __syncthreads();

    // 3. 根据得分对 V 向量进行加权求和
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float v_out = 0.0f;
        for (int p = 0; p <= pos_id; ++p) {
            long long v_offset =
                (long long)layer_id * num_kv_heads * max_seq_len * head_dim +
                (long long)kv_head_idx * max_seq_len * head_dim +
                (long long)p * head_dim;
            v_out += shared_scores[p] * __bfloat162float(v_cache[v_offset + d]);
        }
        out[q_head_idx * head_dim + d] = __float2bfloat16(v_out);
    }
}