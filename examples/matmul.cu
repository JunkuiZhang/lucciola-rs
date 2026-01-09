extern "C" __global__ void matmul(float *A, float *B, float *C, int N) {
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    // printf("pos, (%d, %d) - N %d - value %f\n", ROW, COL, N, tmpSum);
    C[ROW * N + COL] = tmpSum;
}

#define TILE 16

extern "C" __global__ void matmul_tiled(const float *A, const float *B,
                                        float *C, int N) {

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    // 遍历 K 维度的所有 tile
    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {

        // 每个线程加载一个元素到 shared memory
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        if (row < N && a_col < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (b_row < N && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // 使用 shared memory 中的数据
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}
