#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>

struct KeyValuePair {
    int key;
    float value;
};

struct ArgMaxOp {
    __device__ __forceinline__ KeyValuePair
    operator()(const KeyValuePair &a, const KeyValuePair &b) const {
        return (b.value > a.value) ? b : a;
    }
};

extern "C" __global__ void
argmax_kernel(const __nv_bfloat16 *__restrict__ input, const int size,
              unsigned int *__restrict__ output_idx) {
    int tid = threadIdx.x;

    // Initialize with lowest possible value
    KeyValuePair local_max;
    local_max.key = -1;
    local_max.value = -1e30f; // Sufficiently small for logits

    // Grid-stride loop (though we likely use 1 block)
    for (int i = tid; i < size; i += blockDim.x) {
        float val = __bfloat162float(input[i]);
        if (val > local_max.value) {
            local_max.value = val;
            local_max.key = i;
        }
    }

    // Block Reduce
    typedef cub::BlockReduce<KeyValuePair, 1024> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    KeyValuePair block_max =
        BlockReduce(temp_storage).Reduce(local_max, ArgMaxOp());

    if (tid == 0) {
        *output_idx = (unsigned int)block_max.key;
    }
}
