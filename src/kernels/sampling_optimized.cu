#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda_bf16.h>

struct KeyValuePair {
    int key;
    float value;
};

struct ArgMaxPairOps {
    __device__ __forceinline__ KeyValuePair
    operator()(const KeyValuePair &a, const KeyValuePair &b) const {
        return (b.value > a.value) ? b : a;
    }
};

// Part 1: Top-K Filter
// Grid: 32 blocks. Each block finds Top-32 items from its partition.
// Total Output: 1024 items.
// Assumption: (VocabSize / 32) fits in Shared Memory (Float).
// 152000 / 32 ~= 4750. 4750 * 4B ~= 19KB. Safe.
extern "C" __global__ void
top_k_filter(const __nv_bfloat16 *__restrict__ logits,
             KeyValuePair *candidates, // Output [1024]
             int vocab_size,
             int k_per_block // 32
) {
    // 1. Setup Partition
    int partition_count = gridDim.x; // 32
    int partition_idx = blockIdx.x;
    int items_per_partition =
        (vocab_size + partition_count - 1) / partition_count;

    int start_idx = partition_idx * items_per_partition;
    int end_idx = min(start_idx + items_per_partition, vocab_size);
    int num_items = end_idx - start_idx;

    if (num_items <= 0)
        return;

    // 2. Load into Shared Memory (as float)
    // Dynamic Shared Memory: declared in launch (items_per_partition *
    // sizeof(float))
    extern __shared__ float s_logits[];

    for (int i = threadIdx.x; i < items_per_partition; i += blockDim.x) {
        if (start_idx + i < vocab_size) // boundary check inside partition logic
            s_logits[i] = __bfloat162float(logits[start_idx + i]);
        else
            s_logits[i] = -1e30f; // Padding
    }
    __syncthreads();

    // 3. Iteratively find Max
    typedef cub::BlockReduce<KeyValuePair, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    for (int k = 0; k < k_per_block; ++k) {
        KeyValuePair thread_max;
        thread_max.key = -1;
        thread_max.value = -1e30f;

        // Grid stide loop over shared memory
        for (int i = threadIdx.x; i < items_per_partition; i += blockDim.x) {
            float val = s_logits[i];
            if (val > thread_max.value) {
                thread_max.value = val;
                thread_max.key = i;
            }
        }

        // Reduce
        KeyValuePair block_max =
            BlockReduce(temp_storage).Reduce(thread_max, ArgMaxPairOps());
        __syncthreads();

        // Write output and mask
        if (threadIdx.x == 0) {
            int global_out_idx = partition_idx * k_per_block + k;
            candidates[global_out_idx].key = start_idx + block_max.key;
            candidates[global_out_idx].value = block_max.value;

            // Mask out found max in shared memory to prevent re-selection
            // We need to signal this index to the specific thread?
            // Or just do it here if share-mem is accessible by tid 0 (it is).
            s_logits[block_max.key] = -1e30f;
        }
        __syncthreads();
    }
}

// Part 2: Fused Sort and Sample
// Input: 1024 candidates.
// BlockDim: 1024. GridDim: 1.
extern "C" __global__ void
fused_top_p_sample(KeyValuePair *candidates, // Input/Output (Sorted buffer)
                   int n,                    // 1024
                   float top_p, float rand_val, unsigned int *output_idx) {
    int tid = threadIdx.x;

    // 1. Load Data
    __shared__ KeyValuePair s_data[1024];
    if (tid < n) {
        s_data[tid] = candidates[tid];
    } else {
        s_data[tid] = {-1, -1e30f};
    }
    __syncthreads();

    // 2. Bitonic Sort (Descending)
    // Since N=1024 is power of 2, standard bitonic sort
    for (int k = 2; k <= 1024; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                if ((tid & k) == 0) {
                    // Descending Check
                    if (s_data[tid].value < s_data[ixj].value) {
                        KeyValuePair tmp = s_data[tid];
                        s_data[tid] = s_data[ixj];
                        s_data[ixj] = tmp;
                    }
                } else {
                    // Ascending Check (for lower half of bitonic merge)
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

    // 3. Softmax (Max Sub + Exp)
    // Compute Max
    float max_val = s_data[0].value;

    // Compute Exp and Sum
    float local_val = expf(s_data[tid].value - max_val);

    typedef cub::BlockReduce<float, 1024> BlockReduceFloat;
    __shared__ typename BlockReduceFloat::TempStorage temp_storage;
    float total_sum = BlockReduceFloat(temp_storage).Sum(local_val);
    __shared__ float s_total_sum;
    if (tid == 0)
        s_total_sum = total_sum;
    __syncthreads();

    // 4. Scan (CDF)
    float prob = local_val / s_total_sum;

    typedef cub::BlockScan<float, 1024> BlockScan;
    __shared__ typename BlockScan::TempStorage scan_storage;
    float cdf, aggregate;
    BlockScan(scan_storage).InclusiveSum(prob, cdf, aggregate);

    // 5. Select
    // We want smallest idx where cdf >= rand_val * cutoff? No.
    // Standard Top-P:
    // We accumulated probs until sum >= top_p.
    // We only keep those.
    // Normalized sample: r ~ U[0, 1].
    // Target = r * sum_probs_in_cutoff.
    // Actually, simplifying:
    // Standard sampling: Find first idx where CDF >= rand_val.
    // But we need to clamp distribution to Top-P mass?

    // For now, let's implement standard Sampling on the sorted set.
    // If strict Top-P is required:
    //   Find cutoff index K where cdf[K] >= top_p.
    //   Rescale probs 0..K by (1/cdf[K]).
    //   Target = rand_val * 1.0 (since we rescaled).
    //   New Target = rand_val * cdf[K] (in original scale).
    //   So find token where cdf >= rand_val * cdf[K].

    // Find Cutoff CDF
    __shared__ float cutoff_cdf;
    if (tid == 0)
        cutoff_cdf = 1.0f; // Default
    __syncthreads();

    // Find first thread that exceeds top_p
    float prev_cdf = cdf - prob;
    if (prev_cdf < top_p && cdf >= top_p) {
        // This is the cutoff token
        // We write the cutoff CDF to shared mem
        // Atomic not needed if strictly one thread (monotonic scan)
        cutoff_cdf = cdf;
        // Or better: cutoff_cdf = min(cdf, 1.0f);
    }
    __syncthreads();

    // Adjusted Target
    float target = rand_val * cutoff_cdf;

    // Find Winner
    // First token with cdf >= target
    __shared__ int winner_idx;
    if (tid == 0)
        winner_idx = s_data[0].key; // Fallback
    __syncthreads();

    if (prev_cdf < target && cdf >= target) {
        winner_idx = s_data[tid].key;
    }
    __syncthreads();

    if (tid == 0) {
        *output_idx = winner_idx;
    }
}
