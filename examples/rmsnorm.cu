#include <cub/block/block_reduce.cuh>

// 并行归约：在一个 Warp 内求和
__device__ __inline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// 并行归约：在一个 Block 内求和
__device__ __inline__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // 假设最多 1024 线程 (32 warps)
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0)
        val = warpReduceSum(val);

    return val;
}

extern "C" __global__ void rmsnorm_optimized(float *out, const float *input,
                                             const float *weight, float epsilon,
                                             int num_cols) {
    int row = blockIdx.x;
    float sq_sum = 0.0f;

    // 1. 使用 float4 向量化读取并计算平方和
    // 每个线程处理多个 float4 块
    const float4 *x_ptr =
        reinterpret_cast<const float4 *>(input + row * num_cols);
    for (int i = threadIdx.x; i < num_cols / 4; i += blockDim.x) {
        float4 tmp = x_ptr[i];
        sq_sum += tmp.x * tmp.x;
        sq_sum += tmp.y * tmp.y;
        sq_sum += tmp.z * tmp.z;
        sq_sum += tmp.w * tmp.w;
    }

    // 2. Block 级并行归约得到最终平方和
    sq_sum = blockReduceSum(sq_sum);

    // 3. 计算 RMS 因子（仅在主线程计算一次并广播）
    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(sq_sum / num_cols + epsilon);
    }
    __syncthreads();

    // 4. 向量化写回结果
    float4 *out_ptr = reinterpret_cast<float4 *>(out + row * num_cols);
    const float4 *w_ptr = reinterpret_cast<const float4 *>(weight);
    for (int i = threadIdx.x; i < num_cols / 4; i += blockDim.x) {
        float4 in_val = x_ptr[i];
        float4 weight_val = w_ptr[i];
        float4 result;
        result.x = in_val.x * inv_rms * weight_val.x;
        result.y = in_val.y * inv_rms * weight_val.y;
        result.z = in_val.z * inv_rms * weight_val.z;
        result.w = in_val.w * inv_rms * weight_val.w;
        out_ptr[i] = result;
    }
}

extern "C" __global__ void rmsnorm_nvidia(float *out, const float *input,
                                          const float *weight, float epsilon,
                                          int num_cols) {
    int row = blockIdx.x;
    float sq_sum = 0.0f;

    // 1. 使用 float4 向量化读取并计算平方和
    // 每个线程处理多个 float4 块
    const float4 *x_ptr =
        reinterpret_cast<const float4 *>(input + row * num_cols);
    for (int i = threadIdx.x; i < num_cols / 4; i += blockDim.x) {
        float4 tmp = x_ptr[i];
        sq_sum += tmp.x * tmp.x;
        sq_sum += tmp.y * tmp.y;
        sq_sum += tmp.z * tmp.z;
        sq_sum += tmp.w * tmp.w;
    }

    // 2. Block 级并行归约得到最终平方和
    typedef cub::BlockReduce<float, 256> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    sq_sum = BlockReduceT(temp_storage).Sum(sq_sum);

    // 3. 计算 RMS 因子（仅在主线程计算一次并广播）
    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(sq_sum / num_cols + epsilon);
    }
    __syncthreads();

    // 4. 向量化写回结果
    float4 *out_ptr = reinterpret_cast<float4 *>(out + row * num_cols);
    const float4 *w_ptr = reinterpret_cast<const float4 *>(weight);
    for (int i = threadIdx.x; i < num_cols / 4; i += blockDim.x) {
        float4 in_val = x_ptr[i];
        float4 weight_val = w_ptr[i];
        float4 result;
        result.x = in_val.x * inv_rms * weight_val.x;
        result.y = in_val.y * inv_rms * weight_val.y;
        result.z = in_val.z * inv_rms * weight_val.z;
        result.w = in_val.w * inv_rms * weight_val.w;
        out_ptr[i] = result;
    }
}

extern "C" __global__ void
rmsnorm_kernel(float *out,          // 输出
               const float *input,  // 输入
               const float *weight, // 权重 (gamma)
               float epsilon,       // 稳定项
               int num_cols         // 向量维度 (例如 Llama-7B 是 4096)
) {
    // 每一行由一个 Block 处理
    int row = blockIdx.x;
    const float *x = input + row * num_cols;
    float *y = out + row * num_cols;

    // 1. 计算平方和 (Sum of Squares)
    // 为了简单，这里使用单线程循环。在生产环境会使用“归约算法(Reduction)”优化
    float sum = 0.0f;
    for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
        sum += x[i] * x[i];
    }

    // 使用原子加或 Shuffle 指令汇总所有线程的 sum (此处为简化示意)
    // 我们先用一个简单的逻辑：假设 blockDim.x
    // 覆盖了所有维度，或者只用一个线程算
    __shared__ float shared_ss;
    if (threadIdx.x == 0)
        shared_ss = 0;
    __syncthreads();

    atomicAdd(&shared_ss, sum);
    __syncthreads();

    // 2. 计算 RMS 因子
    float inv_rms = rsqrtf(shared_ss / num_cols + epsilon);

    // 3. 标准化并应用权重
    for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
        y[i] = x[i] * inv_rms * weight[i];
    }
}
