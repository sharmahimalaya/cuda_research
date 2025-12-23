#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <time.h>

// #define M 1024
// #define K 1024
// #define N 1024

#define M 6144
#define K 6144
#define N 6144


// double get_time() {
//     struct timespec ts;
//     clock_gettime(CLOCK_MONOTONIC, &ts);
//     return ts.tv_sec*1e3 + ts.tv_nsec * 1e-6;
// }

__global__ void blocktilingmulti(float *A, float *B, float *C) {
    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // printf("BM*BK : (%d), BN*BK : (%d), blockDim.x: (%d)\n", BM * BK, BN * BK, blockDim.x);
    assert(BM * BK == blockDim.x);
    assert(BN * BK == blockDim.x);
    const uint innerColA = threadIdx.x % BK; 
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN; 
    const uint innerRowB = threadIdx.x / BN;

    float threadResults[TM] = {0.0};
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    A += BK;
    B += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        float tmpB = Bs[dotIdx * BN + threadCol];
        for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
        }
    }
    __syncthreads();
    }

    // write out the results
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
    }
}

int main(){
    float *d_a, *d_b, *d_c;
    float *A, *B;   

    srand(time(NULL));

    A = (float*)malloc(M * K * sizeof(float));
    B = (float*)malloc(K * N * sizeof(float));
    for(int i = 0; i < M*K; i++) A[i] = static_cast<float>(rand()) / RAND_MAX;
    for(int i = 0; i < K*N; i++) B[i] = static_cast<float>(rand()) / RAND_MAX;
    
    cudaMalloc((void**)&d_a, M * K * sizeof(float));
    cudaMalloc((void**)&d_b, K * N * sizeof(float));
    cudaMalloc((void**)&d_c, M * N * sizeof(float));

    cudaMemcpy(d_a, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // cudaMemset(d_c, 0, M * N * sizeof(float));

    dim3 blockSize(512);
    dim3 gridSize(96, 96);

    blocktilingmulti<<<gridSize, blockSize>>>(d_a, d_b, d_c);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    blocktilingmulti<<<gridSize, blockSize>>>(d_a, d_b, d_c);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    double num_ops = 2.0 * M * N * K;
    double gflops = num_ops / (elapsed_ms * 1e6);

    printf("Time: %.3f ms\n", elapsed_ms);
    printf("Performance: %.2f GFLOPS\n", gflops);

    free(A);
    free(B);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}