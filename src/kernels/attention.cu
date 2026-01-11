#include <cuda_runtime.h>
#include <cuda_bf16.h>

template <int HEAD_DIM>
__device__ void flash_decoding_impl(
    __nv_bfloat16 *output,  
    const __nv_bfloat16 *q, 
    const __nv_bfloat16 *k_pool, 
    const __nv_bfloat16 *v_pool, 
    const int* block_table,     
    int layer_idx, 
    int num_q_heads, 
    int num_kv_heads,
    int head_dim_rt, 
    int max_seq_len,
    int current_pos, 
    float sm_scale,
    int block_size,
    int num_layers) 
{
    int q_head_idx = blockIdx.x;
    int kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads); 
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    float local_m = -1e20f;
    float local_l = 0.0f;
    float local_o[HEAD_DIM]; 
#pragma unroll
    for (int i = 0; i < HEAD_DIM; ++i) local_o[i] = 0.0f;

    extern __shared__ char smem[];
    float* s_q = (float*)smem;
    int* s_block_table = (int*)(s_q + HEAD_DIM);

    int num_blocks = (current_pos + block_size) / block_size;

    if (tid < HEAD_DIM) {
        s_q[tid] = __bfloat162float(q[q_head_idx * HEAD_DIM + tid]);
    }

    // Load block table to shared memory
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        s_block_table[i] = block_table[i];
    }
    
    __syncthreads();

    // Strides
    long long stride_block = (long long)num_layers * num_kv_heads * block_size * HEAD_DIM;
    long long stride_layer = (long long)num_kv_heads * block_size * HEAD_DIM;
    long long stride_head  = (long long)block_size * HEAD_DIM;

    // Base pointer for this Layer & Head (but dependent on Block P)
    // Offset_in_block = (layer * stride_layer) + (head * stride_head) + (token * head_dim)
    long long layer_head_offset = (long long)layer_idx * stride_layer + (long long)kv_head_idx * stride_head;

    for (int t = tid; t <= current_pos; t += blockDim.x) {
        // Paged Address Calculation
        int log_blk = t / block_size;
        int phy_blk = s_block_table[log_blk];
        int tok_off = t % block_size;

        long long final_offset = 
            (long long)phy_blk * stride_block + 
            layer_head_offset + 
            (long long)tok_off * HEAD_DIM;

        // --- Attention Logic ---
        float score = 0.0f;
        
#pragma unroll
        for (int i = 0; i < HEAD_DIM; ++i) {
            float ki = __bfloat162float(k_pool[final_offset + i]);
            score += s_q[i] * ki;
        }
        score *= sm_scale;

        float m_prev = local_m;
        local_m = fmaxf(m_prev, score);
        float exp_m = expf(m_prev - local_m);
        float exp_s = expf(score - local_m);

        local_l = local_l * exp_m + exp_s;

#pragma unroll
        for (int i = 0; i < HEAD_DIM; ++i) {
            float vi = __bfloat162float(v_pool[final_offset + i]);
            local_o[i] = local_o[i] * exp_m + vi * exp_s;
        }
    }

    // Reduction (Same as before)
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_m = __shfl_down_sync(0xffffffff, local_m, offset);
        float other_l = __shfl_down_sync(0xffffffff, local_l, offset);

        float new_m = fmaxf(local_m, other_m);
        float scale_self = expf(local_m - new_m);
        float scale_other = expf(other_m - new_m);

        local_m = new_m;
        local_l = local_l * scale_self + other_l * scale_other;

        for (int i = 0; i < HEAD_DIM; i++) {
            float val = local_o[i];
            float other_val = __shfl_down_sync(0xffffffff, val, offset);
            local_o[i] = val * scale_self + other_val * scale_other;
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
            s_q[i] = local_o[i]; 
    }

    __syncthreads();

    if (tid < HEAD_DIM) {
        output[q_head_idx * HEAD_DIM + tid] = __float2bfloat16(s_q[tid]);
    }
}

extern "C" __global__ void flash_decoding_kernel_64(
    __nv_bfloat16 *output, const __nv_bfloat16 *q, const __nv_bfloat16 *k,
    const __nv_bfloat16 *v, const int* block_table,
    int layer_idx, int num_q_heads, int num_kv_heads,
    int head_dim, int max_seq_len, int current_pos, float sm_scale, int block_size, int num_layers) {
    flash_decoding_impl<64>(output, q, k, v, block_table, layer_idx, num_q_heads, num_kv_heads,
                            head_dim, max_seq_len, current_pos, sm_scale, block_size, num_layers);
}

extern "C" __global__ void flash_decoding_kernel_128(
    __nv_bfloat16 *output, const __nv_bfloat16 *q, const __nv_bfloat16 *k,
    const __nv_bfloat16 *v, const int* block_table,
    int layer_idx, int num_q_heads, int num_kv_heads,
    int head_dim, int max_seq_len, int current_pos, float sm_scale, int block_size, int num_layers) {
    flash_decoding_impl<128>(output, q, k, v, block_table, layer_idx, num_q_heads, num_kv_heads,
                             head_dim, max_seq_len, current_pos, sm_scale, block_size, num_layers);
}
