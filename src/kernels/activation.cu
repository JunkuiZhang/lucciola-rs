#include <cuda_bf16.h>

// 基础的 SiLU 函数
__device__ __inline__ float silu(float x) { return x / (1.0f + expf(-x)); }

extern "C" __global__ void
silu_and_mul_kernel(__nv_bfloat16 *out,        // 输出: [num_elements]
                    const __nv_bfloat16 *gate, // 输入1: gate_proj 的结果
                    const __nv_bfloat16 *up,   // 输入2: up_proj 的结果
                    int n) {
    // 使用 float4 向量化 (128位 = 8个 bf16) 以最大化显存带宽利用率
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_vec = n / 8;

    if (idx < num_vec) {
        // 强制转换为 float4 指针进行 128 位读取
        const float4 *g_vec_ptr = reinterpret_cast<const float4 *>(gate);
        const float4 *u_vec_ptr = reinterpret_cast<const float4 *>(up);
        float4 *o_vec_ptr = reinterpret_cast<float4 *>(out);

        float4 g_val = g_vec_ptr[idx];
        float4 u_val = u_vec_ptr[idx];
        float4 res_val;

        // 将读取到的 float4 数据视为 4 个 __nv_bfloat162 包
        __nv_bfloat162 *g_bf2 = reinterpret_cast<__nv_bfloat162 *>(&g_val);
        __nv_bfloat162 *u_bf2 = reinterpret_cast<__nv_bfloat162 *>(&u_val);
        __nv_bfloat162 *r_bf2 = reinterpret_cast<__nv_bfloat162 *>(&res_val);

#pragma unroll
        for (int k = 0; k < 4; ++k) {
            // 将 bf16x2 转为 float2 计算
            float2 g_f2 = __bfloat1622float2(g_bf2[k]);
            float2 u_f2 = __bfloat1622float2(u_bf2[k]);
            float2 r_f2;

            r_f2.x = silu(g_f2.x) * u_f2.x;
            r_f2.y = silu(g_f2.y) * u_f2.y;

            // 转回 bf16x2
            r_bf2[k] = __float22bfloat162_rn(r_f2);
        }

        // 写回 128 位结果
        o_vec_ptr[idx] = res_val;
    }
}