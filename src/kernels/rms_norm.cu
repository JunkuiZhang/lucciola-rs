#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>

extern "C" __global__ void rmsnorm_nvidia(__nv_bfloat16 *out,
                                          const __nv_bfloat16 *input,
                                          const __nv_bfloat16 *weight,
                                          float epsilon, int num_cols) {
    int row = blockIdx.x;
    float sq_sum = 0.0f;

    // 16 bytes = 8 elements of bf16
    int num_vec_elements = num_cols / 8;

    const float4 *x_ptr =
        reinterpret_cast<const float4 *>(input + row * num_cols);

    for (int i = threadIdx.x; i < num_vec_elements; i += blockDim.x) {
        float4 packed_val = x_ptr[i];
        __nv_bfloat162 *packed_val_bf16 =
            reinterpret_cast<__nv_bfloat162 *>(&packed_val);

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 val = __bfloat1622float2(packed_val_bf16[j]);
            sq_sum += val.x * val.x + val.y * val.y;
        }
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

    for (int i = threadIdx.x; i < num_vec_elements; i += blockDim.x) {
        float4 in_val = x_ptr[i];
        float4 weight_val = w_ptr[i];
        float4 result;

        __nv_bfloat162 *in_val_bf16 =
            reinterpret_cast<__nv_bfloat162 *>(&in_val);
        __nv_bfloat162 *weight_val_bf16 =
            reinterpret_cast<__nv_bfloat162 *>(&weight_val);
        __nv_bfloat162 *result_bf16 =
            reinterpret_cast<__nv_bfloat162 *>(&result);

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 in_f2 = __bfloat1622float2(in_val_bf16[j]);
            float2 w_f2 = __bfloat1622float2(weight_val_bf16[j]);
            float2 res_f2;
            res_f2.x = in_f2.x * inv_rms * w_f2.x;
            res_f2.y = in_f2.y * inv_rms * w_f2.y;
            result_bf16[j] = __float22bfloat162_rn(res_f2);
        }
        out_ptr[i] = result;
    }
}