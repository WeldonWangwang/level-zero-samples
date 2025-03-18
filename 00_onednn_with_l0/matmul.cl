__kernel void matmul(
    __global float* A, 
    __global float* B, 
    __global float* C, 
    const int M, const int N, const int K) 
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

