#include <cuda_bf16.h>

struct KeyValuePair {
    int key;
    float value;
};

extern "C" __global__ void bitonic_sort_step(KeyValuePair *items, int j, int k,
                                             int n) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = tid ^ j;

    if (ixj > tid) {
        if ((tid & k) == 0) {
            // Sort ascending
            if (tid < n && ixj < n) {
                if (items[tid].value < items[ixj].value) {
                    KeyValuePair temp = items[tid];
                    items[tid] = items[ixj];
                    items[ixj] = temp;
                }
            }
        } else {
            // Sort descending
            if (tid < n && ixj < n) {
                if (items[tid].value > items[ixj].value) {
                    KeyValuePair temp = items[tid];
                    items[tid] = items[ixj];
                    items[ixj] = temp;
                }
            }
        }
    }
}

extern "C" __global__ void init_pairs(KeyValuePair *items,
                                      const __nv_bfloat16 *logits, int n,
                                      int vocab_size) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < vocab_size) {
        items[tid].key = tid;
        items[tid].value = __bfloat162float(logits[tid]);
    } else if (tid < n) {
        items[tid].key = -1;
        items[tid].value = -1e30f; // Padding
    }
}
