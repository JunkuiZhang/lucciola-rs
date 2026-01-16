#include <cuda_bf16.h>

extern "C" __global__ void rope(
    __nv_bfloat16 *q, // [num_heads, head_dim]
    const __nv_bfloat16 *q_bias,
    __nv_bfloat16 *k, // [num_kv_heads, head_dim]
    const __nv_bfloat16 *k_bias,
    const float *cos_cache, // [max_seq_len, head_dim / 2]
    const float *sin_cache, // [max_seq_len, head_dim / 2]
    const int *pos_ptr,
    int head_dim,
    int num_q_heads,
    int num_k_heads) {
    int pos_idx = *pos_ptr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程处理一对 (2个) 元素
    int half_dim = head_dim / 2;
    if (i >= num_q_heads * half_dim)
        return;

    int head_idx = i / half_dim;
    int dim_idx = i % half_dim;

    float cos_val = cos_cache[pos_idx * half_dim + dim_idx];
    float sin_val = sin_cache[pos_idx * half_dim + dim_idx];

    // Q 向量的旋转
    int idx1 = head_idx * head_dim + dim_idx;
    int idx2 = head_idx * head_dim + dim_idx + half_dim;

    float q_b1 = __bfloat162float(q_bias[idx1]);
    float q_b2 = __bfloat162float(q_bias[idx2]);
    float v1 = __bfloat162float(q[idx1]) + q_b1;
    float v2 = __bfloat162float(q[idx2]) + q_b2;

    // 旋转矩阵计算
    q[idx1] = __float2bfloat16(v1 * cos_val - v2 * sin_val);
    q[idx2] = __float2bfloat16(v1 * sin_val + v2 * cos_val);

    // 对 K 进行同样操作 (如果 i 在 K 的范围内)
    if (head_idx < num_k_heads) {
        int k_idx1 = head_idx * head_dim + dim_idx;
        int k_idx2 = head_idx * head_dim + dim_idx + half_dim;

        float k_b1 = __bfloat162float(k_bias[k_idx1]);
        float k_b2 = __bfloat162float(k_bias[k_idx2]);
        float kv1 = __bfloat162float(k[k_idx1]) + k_b1;
        float kv2 = __bfloat162float(k[k_idx2]) + k_b2;

        k[k_idx1] = __float2bfloat16(kv1 * cos_val - kv2 * sin_val);
        k[k_idx2] = __float2bfloat16(kv1 * sin_val + kv2 * cos_val);
    }
}
