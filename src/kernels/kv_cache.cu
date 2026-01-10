#include <cuda_bf16.h>

extern "C" __global__ void update_kv_cache_kernel(
    __nv_bfloat16 *k_cache, // [num_layers, num_kv_heads, max_seq_len, head_dim]
    __nv_bfloat16 *v_cache, // 同上
    const __nv_bfloat16 *new_k, // 当前层新算的 k [num_kv_heads, head_dim]
    const __nv_bfloat16 *new_v, // 当前层新算的 v [num_kv_heads, head_dim]
    int layer_id, int pos_id, int num_layers, int num_kv_heads, int max_seq_len,
    int head_dim) {
    int head_idx = blockIdx.x; // 对应哪个 KV 头
    int dim_idx = threadIdx.x; // 对应维度

    if (head_idx < num_kv_heads && dim_idx < head_dim) {
        // 计算在大缓存中的偏移量
        // 索引公式: layer * (heads * max_len * dim) + head * (max_len * dim) +
        // pos * dim + dim_idx
        long long layer_offset =
            (long long)layer_id * num_kv_heads * max_seq_len * head_dim;
        long long head_offset = (long long)head_idx * max_seq_len * head_dim;
        long long pos_offset = (long long)pos_id * head_dim;

        long long cache_idx = layer_offset + head_offset + pos_offset + dim_idx;
        int new_idx = head_idx * head_dim + dim_idx;

        k_cache[cache_idx] = new_k[new_idx];
        v_cache[cache_idx] = new_v[new_idx];
    }
}