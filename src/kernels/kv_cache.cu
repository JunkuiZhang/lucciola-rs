#include <cuda_bf16.h>

extern "C" __global__ void update_kv_cache_kernel(
    __nv_bfloat16 *k_pool, // Base Pointer to Paged Pool
    __nv_bfloat16 *v_pool, const __nv_bfloat16 *new_k,
    const __nv_bfloat16 *new_v, const __nv_bfloat16 *v_bias,
    const int *block_table, // [max_num_blocks] mapping logical->physical
    int layer_id, const int* pos_ptr, int num_layers, int num_kv_heads,
    int max_seq_len, // Not strictly needed for addressing anymore, but maybe
                     // for bound checking
    int head_dim, int block_size) {
    int pos_id = *pos_ptr;
    int head_idx = blockIdx.x; // kv_head
    int dim_idx = threadIdx.x; // dim

    if (head_idx < num_kv_heads && dim_idx < head_dim) {
        // 1. Calculate Logical Block and Offset
        int logical_block_idx = pos_id / block_size;
        int block_offset = pos_id % block_size;

        // 2. Look up Physical Block
        int physical_block_idx = block_table[logical_block_idx];

        // 3. Calculate Physical Address in Pool
        // Layout: [PhysicalBlock, NumLayers, NumKVHeads, BlockSize, HeadDim]
        // Stride breakdown:
        //  Block: NumLayers * NumKVHeads * BlockSize * HeadDim
        //  Layer: NumKVHeads * BlockSize * HeadDim
        //  Head:  BlockSize * HeadDim
        //  Token: HeadDim
        //  Dim:   1

        long long stride_block =
            (long long)num_layers * num_kv_heads * block_size * head_dim;
        long long stride_layer =
            (long long)num_kv_heads * block_size * head_dim;
        long long stride_head = (long long)block_size * head_dim;

        long long final_offset = (long long)physical_block_idx * stride_block +
                                 (long long)layer_id * stride_layer +
                                 (long long)head_idx * stride_head +
                                 (long long)block_offset * head_dim + dim_idx;

        int new_idx = head_idx * head_dim + dim_idx;

        // Write K
        k_pool[final_offset] = new_k[new_idx];

        // Write V (with Bias)
        float val_v = __bfloat162float(new_v[new_idx]);
        float bias = __bfloat162float(v_bias[new_idx]);
        v_pool[final_offset] = __float2bfloat16(val_v + bias);
    }
}