#include <cuda_bf16.h>

// 基础的 SiLU 函数
__device__ __inline__ float silu(float x) { return x / (1.0f + expf(-x)); }

extern "C" __global__ void silu_and_mul_fused_kernel(
    __nv_bfloat16 *buffer, // interleaved memory: G0...U0...G1...
    int mid_stride, // intermediate_size (number of columns for one gate ref)
    int n // total number of elements to process (seq_len * intermediate_size)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Find which row we are in
        int row = idx / mid_stride;
        int col = idx % mid_stride;

        // Base pointer for this row
        // Each row has stride = 2 * mid_stride (Gate + Up)
        long long row_start = (long long)row * 2 * mid_stride;
        int gate_idx = row_start + col;
        int up_idx = gate_idx + mid_stride;

        float g = __bfloat162float(buffer[gate_idx]);
        float u = __bfloat162float(buffer[up_idx]);

        float val = silu(g) * u;

        buffer[gate_idx] = __float2bfloat16(val);
    }
}
