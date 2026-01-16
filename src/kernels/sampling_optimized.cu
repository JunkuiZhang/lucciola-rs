#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda_bf16.h>

struct KeyValuePair {
    int key;
    float value;
};

// --- Part 1: Top-K Filter (Warp Optimized) ---

#define FINAL_MASK 0xffffffff

__inline__ __device__ KeyValuePair warpReduceMax(KeyValuePair val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other_v = __shfl_down_sync(FINAL_MASK, val.value, offset);
        int other_k = __shfl_down_sync(FINAL_MASK, val.key, offset);
        if (other_v > val.value) {
            val.value = other_v;
            val.key = other_k;
        }
    }
    return val;
}

extern "C" __global__ void
top_k_filter(const __nv_bfloat16 *__restrict__ logits,
             KeyValuePair *candidates, // Output [GridDim * K_per_block]
             int vocab_size,
             int k_per_block // typically 32
) {
    int partition_count = gridDim.x;
    int partition_idx = blockIdx.x;

    // Use long long to prevent overflow
    long long start_idx_ll =
        (long long)partition_idx * vocab_size / partition_count;
    long long end_idx_ll =
        (long long)(partition_idx + 1) * vocab_size / partition_count;
    int start_idx = (int)start_idx_ll;
    int end_idx = (int)end_idx_ll;
    int num_items = end_idx - start_idx;

    if (num_items <= 0)
        return;

    extern __shared__ float s_logits[];
    volatile float *v_s_logits = s_logits;

    for (int i = threadIdx.x; i < num_items; i += blockDim.x) {
        if (start_idx + i < vocab_size) {
            s_logits[i] = __bfloat162float(logits[start_idx + i]);
        } else {
            s_logits[i] = -1e30f;
        }
    }
    __syncthreads();

    __shared__ KeyValuePair s_warp_max[32];

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    for (int k = 0; k < k_per_block; ++k) {
        KeyValuePair thread_max;
        thread_max.key = -1;
        thread_max.value = -1e30f;

        for (int i = threadIdx.x; i < num_items; i += blockDim.x) {
            float val = v_s_logits[i];
            if (val > thread_max.value) {
                thread_max.value = val;
                thread_max.key = i;
            }
        }

        thread_max = warpReduceMax(thread_max);

        if (lane_id == 0) {
            s_warp_max[warp_id] = thread_max;
        }
        __syncthreads();

        KeyValuePair block_max = {-1, -1e30f};
        if (warp_id == 0) {
            int num_warps = (blockDim.x + 31) / 32;
            if (lane_id < num_warps) {
                block_max = s_warp_max[lane_id];
            }
            block_max = warpReduceMax(block_max);
        }

        if (threadIdx.x == 0) {
            s_warp_max[0] = block_max;
        }
        __syncthreads();
        block_max = s_warp_max[0];

        int found_key = block_max.key;

        if (threadIdx.x == 0) {
            int global_out_idx = partition_idx * k_per_block + k;
            candidates[global_out_idx].key = start_idx + found_key;
            candidates[global_out_idx].value = block_max.value;

            if (found_key >= 0 && found_key < num_items)
                v_s_logits[found_key] = -1e30f;
        }
        __syncthreads();
    }
}

// --- Part 2: Fused Sort and Sample (Templated) ---

template <int MAX_CANDIDATES>
__device__ void fused_top_p_sample_impl(KeyValuePair *candidates, float top_p,
                                        float rand_val,
                                        unsigned int *output_idx) {
    int tid = threadIdx.x;

    __shared__ KeyValuePair s_data[MAX_CANDIDATES];
    if (tid < MAX_CANDIDATES) {
        s_data[tid] = candidates[tid];
    }
    __syncthreads();

#pragma unroll
    for (int k = 2; k <= MAX_CANDIDATES; k <<= 1) {
#pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                if ((tid & k) == 0) {
                    if (s_data[tid].value < s_data[ixj].value) {
                        KeyValuePair tmp = s_data[tid];
                        s_data[tid] = s_data[ixj];
                        s_data[ixj] = tmp;
                    }
                } else {
                    if (s_data[tid].value > s_data[ixj].value) {
                        KeyValuePair tmp = s_data[tid];
                        s_data[tid] = s_data[ixj];
                        s_data[ixj] = tmp;
                    }
                }
            }
            __syncthreads();
        }
    }

    float max_val = s_data[0].value;
    float local_val = expf(s_data[tid].value - max_val);

    typedef cub::BlockReduce<float, MAX_CANDIDATES> BlockReduceFloat;
    __shared__ typename BlockReduceFloat::TempStorage temp_storage;
    float total_sum = BlockReduceFloat(temp_storage).Sum(local_val);
    __shared__ float s_total_sum;
    if (tid == 0)
        s_total_sum = total_sum;
    __syncthreads();

    float prob = local_val / s_total_sum;
    typedef cub::BlockScan<float, MAX_CANDIDATES> BlockScan;
    __shared__ typename BlockScan::TempStorage scan_storage;
    float cdf, aggregate;
    BlockScan(scan_storage).InclusiveSum(prob, cdf, aggregate);

    __shared__ float cutoff_cdf;
    if (tid == 0)
        cutoff_cdf = 1.0f;
    __syncthreads();

    float prev_cdf = cdf - prob;
    if (prev_cdf < top_p && cdf >= top_p) {
        cutoff_cdf = cdf;
    }
    __syncthreads();

    float target = rand_val * cutoff_cdf;
    bool wins = (cdf >= target) && (prev_cdf < target);

    if (wins) {
        *output_idx = s_data[tid].key;
    }
}

extern "C" __global__ void fused_top_p_sample_32(KeyValuePair *candidates,
                                                 float top_p, float rand_val,
                                                 unsigned int *output_idx) {
    fused_top_p_sample_impl<32>(candidates, top_p, rand_val, output_idx);
}

extern "C" __global__ void fused_top_p_sample_64(KeyValuePair *candidates,
                                                 float top_p, float rand_val,
                                                 unsigned int *output_idx) {
    fused_top_p_sample_impl<64>(candidates, top_p, rand_val, output_idx);
}

extern "C" __global__ void fused_top_p_sample_128(KeyValuePair *candidates,
                                                  float top_p, float rand_val,
                                                  unsigned int *output_idx) {
    fused_top_p_sample_impl<128>(candidates, top_p, rand_val, output_idx);
}

extern "C" __global__ void fused_top_p_sample_256(KeyValuePair *candidates,
                                                  float top_p, float rand_val,
                                                  unsigned int *output_idx) {
    fused_top_p_sample_impl<256>(candidates, top_p, rand_val, output_idx);
}

extern "C" __global__ void fused_top_p_sample_512(KeyValuePair *candidates,
                                                  float top_p, float rand_val,
                                                  unsigned int *output_idx) {
    fused_top_p_sample_impl<512>(candidates, top_p, rand_val, output_idx);
}

extern "C" __global__ void fused_top_p_sample_1024(KeyValuePair *candidates,
                                                   float top_p, float rand_val,
                                                   unsigned int *output_idx) {
    fused_top_p_sample_impl<1024>(candidates, top_p, rand_val, output_idx);
}
