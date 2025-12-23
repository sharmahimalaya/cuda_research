#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define N 1024
#define M 1024
#define K 1024

int main(){
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    A = (float*)malloc(M * K * sizeof(float));
    B = (float*)malloc(K * N * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < M * K; ++i) A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) B[i] = static_cast<float>(rand()) / RAND_MAX;

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    //creating cublas handler
    cublasHandle_t handle;
    cublasCreate(&handle);

    
}