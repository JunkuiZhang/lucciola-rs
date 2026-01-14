#include <cuda_bf16.h>

extern "C" __global__ void
batched_embedding_kernel(const __nv_bfloat16 *__restrict__ embedding_table,
                         const unsigned int *__restrict__ input_ids,
                         __nv_bfloat16 *__restrict__ output, int hidden_dim) {
    // blockIdx.x corresponds to the token index in the batch/sequence
    int token_idx = blockIdx.x;
    unsigned int token_id = input_ids[token_idx];

    // Pointers for this specific token
    const __nv_bfloat16 *src_row = embedding_table + (token_id * hidden_dim);
    __nv_bfloat16 *dst_row = output + (token_idx * hidden_dim);

    // Vectorized copy using float4 (128 bits = 8 bf16 elements)
    // Check alignment. hidden_dim usually divisible by 8 for LLMs.
    // If hidden_dim is not divisible by 8, we fallback or handle tail,
    // but for simplicity and typical LLM sizes (e.g. 1024, 4096), vectorization
    // is safe.

    int num_vec_elements = hidden_dim / 8;
    int tail_elements = hidden_dim % 8;

    const float4 *src_vec = reinterpret_cast<const float4 *>(src_row);
    float4 *dst_vec = reinterpret_cast<float4 *>(dst_row);

    for (int i = threadIdx.x; i < num_vec_elements; i += blockDim.x) {
        dst_vec[i] = src_vec[i];
    }

    // Handle any remaining elements (if hidden_dim is not multiple of 8)
    if (tail_elements > 0) {
        int start_idx = num_vec_elements * 8;
        for (int i = threadIdx.x; i < tail_elements; i += blockDim.x) {
            dst_row[start_idx + i] = src_row[start_idx + i];
        }
    }
}
