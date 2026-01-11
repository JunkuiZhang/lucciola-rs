#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>

extern "C" __global__ void
softmax_kernel(__nv_bfloat16 *logits, // 输入输出: [num_heads, seq_len]
               int seq_len) {
    // 1. 设置索引
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;
    int offset = head_idx * seq_len;

    // 2. 局部求最大值 (每个线程处理这一行中的一部分数据)
    float local_max = -1e20f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float val = __bfloat162float(logits[offset + i]);
        local_max = fmaxf(local_max, val);
    }

    // 3. Block 级规约：找到全行的全局最大值
    // Specialize BlockReduce for a 1D block of 1024 threads type float
    typedef cub::BlockReduce<float, 1024> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // 注意：所有线程必须参与 BlockReduce，即使它们在上面的循环中没做工作
    // CUB 会处理 synchronization
    float row_max = BlockReduce(temp_storage).Reduce(local_max, cub::Max());

    // 广播 Max 到所有线程
    __shared__ float shared_max;
    if (tid == 0)
        shared_max = row_max;
    __syncthreads();
    row_max = shared_max;

    // 4. 计算局部指数和
    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float val = __bfloat162float(logits[offset + i]);
        // 使用减去 max 的 trick 防止溢出
        local_sum += expf(val - row_max);
    }

    // 5. Block 级规约：计算全行的指数和
    // 这里的 temp_storage 可以复用，因为之前的规约已经结束
    float row_sum = BlockReduce(temp_storage).Sum(local_sum);

    // 广播 Sum 到所有线程
    __shared__ float shared_sum;
    if (tid == 0)
        shared_sum = row_sum;
    __syncthreads();
    row_sum = shared_sum;

    // 6. 最终归一化并写回
    float inv_sum = 1.0f / (row_sum + 1e-6f);
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float val = __bfloat162float(logits[offset + i]);
        float prob = expf(val - row_max) * inv_sum;
        logits[offset + i] = __float2bfloat16(prob);
    }
}