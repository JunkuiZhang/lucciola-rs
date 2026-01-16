#include <cuda_bf16.h>
#include <cuda_runtime.h>

template <int HEAD_DIM>
__device__ void flash_decoding_impl(
    __nv_bfloat16 *output, const __nv_bfloat16 *q, const __nv_bfloat16 *k_pool,
    const __nv_bfloat16 *v_pool, const int *block_table, int layer_idx,
    int num_q_heads, int num_kv_heads, int head_dim_rt, int max_seq_len,
    const int* current_pos_ptr, float sm_scale, int block_size, int num_layers) {
    int current_pos = *current_pos_ptr;
    // --- Indexing ---
    int q_head_idx = blockIdx.x;
    int kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads);
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;

    // --- Shared Memory Layout ---
    extern __shared__ char smem[];
    float *s_q = (float *)smem; // Size: HEAD_DIM
    // Align to 4 bytes for int*
    int *s_block_table = (int *)(s_q + HEAD_DIM); // Size: num_blocks

    int num_blocks = (current_pos + block_size) / block_size;

    // Pointers for Inter-Warp Reduction (placed after block table)
    // We expect Rust to allocate enough SMEM:
    // HEAD_DIM*4 + num_blocks*4 + WARPS*HEAD_DIM*4 + WARPS*4 + WARPS*4
    float *s_reduce_m = (float *)(s_block_table + num_blocks);
    // Align if num_blocks was odd? int* -> float* address is compatible (4
    // bytes).

    float *s_reduce_l = s_reduce_m + num_warps;
    float *s_reduce_o = s_reduce_l + num_warps; // Size: num_warps * HEAD_DIM

    // --- Load Q Phase ---
    // Warp-cooperative load
    for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
        s_q[i] = __bfloat162float(q[q_head_idx * HEAD_DIM + i]);
    }

    // --- Load Block Table Phase ---
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        s_block_table[i] = block_table[i];
    }

    __syncthreads();

    // --- Local Accumulators ---
    float local_m = -1e20f;
    float local_l = 0.0f;
    // Distributed Output: Each thread handles specific dimensions
    // Max HEAD_DIM 128 (approx) => 128 / 32 = 4 elements per thread
    float local_o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        local_o[i] = 0.0f;

    // --- Strides ---
    long long stride_block =
        (long long)num_layers * num_kv_heads * block_size * HEAD_DIM;
    long long stride_layer = (long long)num_kv_heads * block_size * HEAD_DIM;
    long long stride_head = (long long)block_size * HEAD_DIM;
    long long layer_head_offset = (long long)layer_idx * stride_layer +
                                  (long long)kv_head_idx * stride_head;

    // --- Loop Over Tokens (Warp-Strided) ---
    // Each Warp processes tokens stepping by num_warps
    for (int t = warp_id; t <= current_pos; t += num_warps) {
        int log_blk = t / block_size;
        int phy_blk = s_block_table[log_blk];
        int tok_off = t % block_size;

        long long final_offset = (long long)phy_blk * stride_block +
                                 layer_head_offset +
                                 (long long)tok_off * HEAD_DIM;

        // 1. Compute Dot Product for Token t
        float score = 0.0f;

        // Coalesced Load K and Dot
        for (int i = lane_id; i < HEAD_DIM; i += 32) {
            float val_k = __bfloat162float(k_pool[final_offset + i]);
            score += s_q[i] * val_k;
        }

        // Warp Reduction for Score
        for (int offset = 16; offset > 0; offset /= 2) {
            score += __shfl_down_sync(0xffffffff, score, offset);
        }
        score = __shfl_sync(0xffffffff, score, 0); // Broadcast scalar score
        score *= sm_scale;

        // 2. Online Softmax Update
        float m_prev = local_m;
        local_m = fmaxf(m_prev, score);
        float exp_m = expf(m_prev - local_m);
        float exp_s = expf(score - local_m);

        local_l = local_l * exp_m + exp_s;

        // 3. Accumulate V (Distributed)
        int idx_local = 0;
        for (int i = lane_id; i < HEAD_DIM; i += 32) {
            float val_v = __bfloat162float(v_pool[final_offset + i]);
            local_o[idx_local] = local_o[idx_local] * exp_m + val_v * exp_s;
            idx_local++;
        }
    }

    // --- Inter-Warp Reduction ---
    // Store local state to Shared Mem
    if (lane_id == 0) {
        s_reduce_m[warp_id] = local_m;
        s_reduce_l[warp_id] = local_l;
    }

    // Store distributed O to Shared Mem
    int idx_local = 0;
    for (int i = lane_id; i < HEAD_DIM; i += 32) {
        // Flattened indexing: [warp * Dim + dim]
        s_reduce_o[warp_id * HEAD_DIM + i] = local_o[idx_local];
        idx_local++;
    }

    __syncthreads();

    // Warp 0 Aggregates and Writes
    if (warp_id == 0) {
        // 1. Compute Global M
        float global_m = -1e20f;
        for (int w = 0; w < num_warps; ++w) {
            global_m = fmaxf(global_m, s_reduce_m[w]);
        }

        // 2. Compute Global L
        float global_l = 0.0f;
        for (int w = 0; w < num_warps; ++w) {
            float diff = s_reduce_m[w] - global_m;
            // Prevent underflow/NaN for unused warps (init -1e20)
            if (s_reduce_m[w] <= -1e19f)
                continue;
            global_l += s_reduce_l[w] * expf(diff);
        }

        // 3. Aggregate O and Write
        // Each thread in Warp 0 handles specific dimensions `i`
        for (int i = lane_id; i < HEAD_DIM; i += 32) {
            float sum_o = 0.0f;
            for (int w = 0; w < num_warps; ++w) {
                if (s_reduce_m[w] <= -1e19f)
                    continue;

                float val = s_reduce_o[w * HEAD_DIM + i];
                float scale = expf(s_reduce_m[w] - global_m);
                sum_o += val * scale;
            }
            // Normalize
            sum_o /= global_l;

            // Global Write (Coalesced)
            output[q_head_idx * HEAD_DIM + i] = __float2bfloat16(sum_o);
        }
    }
}

extern "C" __global__ void flash_decoding_kernel_64(
    __nv_bfloat16 *output, const __nv_bfloat16 *q, const __nv_bfloat16 *k,
    const __nv_bfloat16 *v, const int *block_table, int layer_idx,
    int num_q_heads, int num_kv_heads, int head_dim, int max_seq_len,
    const int* current_pos, float sm_scale, int block_size, int num_layers) {
    flash_decoding_impl<64>(output, q, k, v, block_table, layer_idx,
                            num_q_heads, num_kv_heads, head_dim, max_seq_len,
                            current_pos, sm_scale, block_size, num_layers);
}

extern "C" __global__ void flash_decoding_kernel_128(
    __nv_bfloat16 *output, const __nv_bfloat16 *q, const __nv_bfloat16 *k,
    const __nv_bfloat16 *v, const int *block_table, int layer_idx,
    int num_q_heads, int num_kv_heads, int head_dim, int max_seq_len,
    const int* current_pos, float sm_scale, int block_size, int num_layers) {
    flash_decoding_impl<128>(output, q, k, v, block_table, layer_idx,
                             num_q_heads, num_kv_heads, head_dim, max_seq_len,
                             current_pos, sm_scale, block_size, num_layers);
}
