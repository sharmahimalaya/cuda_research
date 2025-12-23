// // #include <iostream>
// // #include <cuda_runtime.h>
// // #include <cublas_v2.h>
// // #include <curand_kernel.h>

// // #define N 4096  // Large enough to hit peak Tensor Core performance
// // #define THREADS_PER_BLOCK 256
// // #define GRID_X 16
// // #define GRID_Y 16

// // #define CHECK_CUDA(call) \
// //     do { \
// //         cudaError_t err = call; \
// //         if (err != cudaSuccess) { \
// //             std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n"; \
// //             exit(EXIT_FAILURE); \
// //         } \
// //     } while (0)

// // #define CHECK_CUBLAS(call) \
// //     do { \
// //         cublasStatus_t status = call; \
// //         if (status != CUBLAS_STATUS_SUCCESS) { \
// //             std::cerr << "cuBLAS error code: " << status << "\n"; \
// //             exit(EXIT_FAILURE); \
// //         } \
// //     } while (0)

// // // Kernel to fill a matrix with uniform random floats
// // __global__ void init_random(float* mat, int size, unsigned long seed) {
// //     int idx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
// //     if (idx < size) {
// //         curandState state;
// //         curand_init(seed, idx, 0, &state);
// //         mat[idx] = curand_uniform(&state);  // [0, 1)
// //     }
// // }

// // int main() {
// //     float *d_A, *d_B, *d_C;
// //     size_t bytes = N * N * sizeof(float);

// //     CHECK_CUDA(cudaMalloc(&d_A, bytes));
// //     CHECK_CUDA(cudaMalloc(&d_B, bytes));
// //     CHECK_CUDA(cudaMalloc(&d_C, bytes));

// //     // Initialize A and B with random values using GPU
// //     dim3 block(THREADS_PER_BLOCK);
// //     dim3 grid(GRID_X, GRID_Y);

// //     init_random<<<grid, block>>>(d_A, N * N, 1234);
// //     init_random<<<grid, block>>>(d_B, N * N, 5678);
// //     CHECK_CUDA(cudaDeviceSynchronize());

// //     // cuBLAS setup
// //     cublasHandle_t handle;
// //     CHECK_CUBLAS(cublasCreate(&handle));
// //     CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

// //     float alpha = 1.0f, beta = 0.0f;


// //     // Timing only the GEMM operation
// //     cudaEvent_t start, stop;
// //     CHECK_CUDA(cudaEventCreate(&start));
// //     CHECK_CUDA(cudaEventCreate(&stop));
// //     CHECK_CUDA(cudaEventRecord(start));

// //     CHECK_CUBLAS(cublasGemmEx(
// //         handle,
// //         CUBLAS_OP_N, CUBLAS_OP_N,
// //         N, N, N,
// //         &alpha,
// //         d_A, CUDA_R_32F, N,
// //         d_B, CUDA_R_32F, N,
// //         &beta,
// //         d_C, CUDA_R_32F, N,
// //         CUBLAS_COMPUTE_32F_FAST_TF32,
// //         CUBLAS_GEMM_DEFAULT_TENSOR_OP));

// //     CHECK_CUDA(cudaEventRecord(stop));
// //     CHECK_CUDA(cudaEventSynchronize(stop));

// //     float ms = 0.0f;
// //     CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

// //     double gflops = (2.0 * N * N * N) / (ms / 1000.0) / 1e9;

// //     std::cout << "Matrix multiplication completed in " << ms << " ms\n";
// //     std::cout << "Performance: " << gflops << " GFLOPS\n";

// //     // Cleanup
// //     cublasDestroy(handle);
// //     cudaFree(d_A);
// //     cudaFree(d_B);
// //     cudaFree(d_C);
// //     cudaEventDestroy(start);
// //     cudaEventDestroy(stop);

// //     return 0;
// // }


// #include <cstdio>
// #include <cstdlib>
// #include <cuda_runtime.h>
// #include <cublas_v2.h>
// #include <ctime>

// #define M 1024
// #define N 1024
// #define K 1024

// int main() {
//     float *h_A, *h_B;
//     float *d_A, *d_B, *d_C;
//     float alpha = 1.0f, beta = 0.0f;

//     cudaMalloc(&d_A, M * K * sizeof(float));
//     cudaMalloc(&d_B, K * N * sizeof(float));
//     cudaMalloc(&d_C, M * N * sizeof(float));

//     h_A = (float*)malloc(M * K * sizeof(float));
//     h_B = (float*)malloc(K * N * sizeof(float));

//     srand(time(NULL));
//     for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
//     for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

//     cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

//     // Create cuBLAS handle
//     cublasHandle_t handle;
//     cublasCreate(&handle);

//     // Optional: use Tensor Cores via TF32
//     // cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

//     // Warm up
//     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
//                 N, M, K,
//                 &alpha,
//                 d_B, N,
//                 d_A, K,
//                 &beta,
//                 d_C, N);
//     cudaDeviceSynchronize();

//     // Reset output memory
//     cudaMemset(d_C, 0, M * N * sizeof(float));

//     // Timing
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     cudaEventRecord(start);
//     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
//                 N, M, K,
//                 &alpha,
//                 d_B, N,
//                 d_A, K,
//                 &beta,
//                 d_C, N);
//     cudaEventRecord(stop);

//     cudaEventSynchronize(stop);
//     float elapsed_ms = 0;
//     cudaEventElapsedTime(&elapsed_ms, start, stop);

//     double num_ops = 2.0 * M * N * K;
//     double gflops = num_ops / (elapsed_ms * 1e6);
//     printf("cuBLAS Time: %.3f ms\n", elapsed_ms);
//     printf("cuBLAS Performance: %.2f GFLOPS\n", gflops);

//     // Cleanup
//     cublasDestroy(handle);
//     free(h_A); free(h_B);
//     cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);
//     return 0;
// }




// #include <cstdio>
// #include <cstdlib>
// #include <cuda_runtime.h>
// #include <cublasLt.h>
// #include <ctime>

// #define M 1024
// #define N 1024
// #define K 1024

// int main() {
//     float *A, *B, *C;
//     float *d_A, *d_B, *d_C;
//     cudaMalloc(&d_A, M * K * sizeof(float));
//     cudaMalloc(&d_B, K * N * sizeof(float));
//     cudaMalloc(&d_C, M * N * sizeof(float));

//     A = (float*)malloc(M * K * sizeof(float));
//     B = (float*)malloc(K * N * sizeof(float));

//     srand(time(NULL));
//     for (int i = 0; i < M * K; ++i) A[i] = static_cast<float>(rand()) / RAND_MAX;
//     for (int i = 0; i < K * N; ++i) B[i] = static_cast<float>(rand()) / RAND_MAX;

//     cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

//     // Create cuBLASLt handle
//     cublasLtHandle_t ltHandle;
//     cublasLtCreate(&ltHandle);

//     cublasLtMatmulDesc_t operationDesc = nullptr;
//     cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;

//     // Set compute type to allow TensorFloat-32 (TF32)
//     cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
//     cudaDataType_t scaleType = CUDA_R_32F;
//     cudaDataType_t dataType = CUDA_R_32F;

//     cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
//     cublasOperation_t transa = CUBLAS_OP_N;
//     cublasOperation_t transb = CUBLAS_OP_N;
//     cublasLtMatmulDescSetAttribute(operationDesc,
//                                     CUBLASLT_MATMUL_DESC_TRANSA,
//                                     &transa,
//                                     sizeof(transa));
//     cublasLtMatmulDescSetAttribute(operationDesc,
//                                     CUBLASLT_MATMUL_DESC_TRANSB,
//                                     &transb,
//                                     sizeof(transb));

//     cublasLtMatrixLayoutCreate(&Adesc, dataType, M, K, K);
//     cublasLtMatrixLayoutCreate(&Bdesc, dataType, K, N, N);
//     cublasLtMatrixLayoutCreate(&Cdesc, dataType, M, N, N);

//     float alpha = 1.0f, beta = 0.0f;

//     // Timing setup
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     // Warm-up
//     cublasLtMatmul(ltHandle,
//                    operationDesc,
//                    &alpha,
//                    d_A, Adesc,
//                    d_B, Bdesc,
//                    &beta,
//                    d_C, Cdesc,
//                    d_C, Cdesc,
//                    nullptr, nullptr, 0, 0);
//     cudaDeviceSynchronize();

//     cudaMemset(d_C, 0, M * N * sizeof(float));
//     cudaEventRecord(start);

//     // Main GEMM
//     cublasLtMatmul(ltHandle,
//                    operationDesc,
//                    &alpha,
//                    d_A, Adesc,
//                    d_B, Bdesc,
//                    &beta,
//                    d_C, Cdesc,
//                    d_C, Cdesc,
//                    nullptr, nullptr, 0, 0);

//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);

//     float elapsed_ms = 0;
//     cudaEventElapsedTime(&elapsed_ms, start, stop);
//     double num_ops = 2.0 * M * N * K;
//     double gflops = num_ops / (elapsed_ms * 1e6);

//     printf("cuBLASLt Time: %.3f ms\n", elapsed_ms);
//     printf("cuBLASLt Performance: %.2f GFLOPS\n", gflops);

//     // Clean up
//     cublasLtMatmulDescDestroy(operationDesc);
//     cublasLtMatrixLayoutDestroy(Adesc);
//     cublasLtMatrixLayoutDestroy(Bdesc);
//     cublasLtMatrixLayoutDestroy(Cdesc);
//     cublasLtDestroy(ltHandle);
//     cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
//     free(A); free(B);
//     cudaEventDestroy(start); cudaEventDestroy(stop);
//     return 0;
// }


#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <ctime>

// #define M 1024
// #define N 1024
// #define K 1024
// #define WORKSPACE_SIZE (8 * 1024 * 1024) // 4MB workspace
// #define NUM_WARMUP 5
// #define NUM_ITER 1

#define M 6144
#define N 6144
#define K 6144
#define NUM_WARMUP 10
#define NUM_ITER 10
#define WORKSPACE_SIZE (64 * 1024 * 1024)

int main() {
    float *A, *B;
    float *d_A, *d_B, *d_C;
    void *workspace = nullptr;

    // Allocate device memory
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMalloc(&workspace, WORKSPACE_SIZE);

    // Allocate pinned host memory
    cudaMallocHost(&A, M * K * sizeof(float));
    cudaMallocHost(&B, K * N * sizeof(float));

    // Initialize host data
    srand(time(NULL));
    for (int i = 0; i < M * K; ++i) A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Copy to device
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Create cuBLASLt handle
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    // Create descriptors
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
    cudaDataType_t scaleType = CUDA_R_32F;
    cudaDataType_t dataType = CUDA_R_32F;

    cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));

    cublasLtMatrixLayoutCreate(&Adesc, dataType, M, K, K);
    cublasLtMatrixLayoutCreate(&Bdesc, dataType, K, N, N);
    cublasLtMatrixLayoutCreate(&Cdesc, dataType, M, N, N);

    // Algorithm selection via heuristics
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);
    size_t workspace_size = WORKSPACE_SIZE;
    cublasLtMatmulPreferenceSetAttribute(preference,
                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &workspace_size,
                                         sizeof(workspace_size));

    cublasLtMatmulHeuristicResult_t heuristicResult;
    int returnedResults = 0;
    cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc,
                                   Adesc, Bdesc, Cdesc, Cdesc,
                                   preference, 1, &heuristicResult,
                                   &returnedResults);
    if (returnedResults == 0) {
        printf("No suitable algorithm found.\n");
        return EXIT_FAILURE;
    }

    float alpha = 1.0f, beta = 0.0f;

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    for (int i = 0; i < NUM_WARMUP; ++i) {
        cublasLtMatmul(ltHandle,
                       operationDesc,
                       &alpha,
                       d_A, Adesc,
                       d_B, Bdesc,
                       &beta,
                       d_C, Cdesc,
                       d_C, Cdesc,
                       &heuristicResult.algo,
                       workspace,
                       WORKSPACE_SIZE,
                       0);
    }
    cudaDeviceSynchronize();

    // Timing main GEMM
    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITER; ++i) {
        cublasLtMatmul(ltHandle,
                       operationDesc,
                       &alpha,
                       d_A, Adesc,
                       d_B, Bdesc,
                       &beta,
                       d_C, Cdesc,
                       d_C, Cdesc,
                       &heuristicResult.algo,
                       workspace,
                       WORKSPACE_SIZE,
                       0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    elapsed_ms /= NUM_ITER;

    double num_ops = 2.0 * M * N * K;
    double gflops = num_ops / (elapsed_ms * 1e6);

    printf("cuBLASLt Avg Time: %.3f ms\n", elapsed_ms);
    printf("cuBLASLt Performance: %.2f GFLOPS\n", gflops);

    // Clean up
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtDestroy(ltHandle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(workspace);
    cudaFreeHost(A); cudaFreeHost(B);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
