// #include <iostream>
// #include <cuda_runtime.h>
// #include <cublas_v2.h>
// #include <curand_kernel.h>

// #define N 1024
// #define THREADS_PER_BLOCK 256
// #define GRID_X 8
// #define GRID_Y 8

// // CUDA error check
// #define CHECK_CUDA(call) \
//     do { \
//         cudaError_t err = call; \
//         if (err != cudaSuccess) { \
//             std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
//             exit(EXIT_FAILURE); \
//         } \
//     } while (0)

// #define CHECK_CUBLAS(call) \
//     do { \
//         cublasStatus_t status = call; \
//         if (status != CUBLAS_STATUS_SUCCESS) { \
//             std::cerr << "cuBLAS Error: " << status << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
//             exit(EXIT_FAILURE); \
//         } \
//     } while (0)

// // Random fill kernel
// __global__ void init_random(float* mat, int size, unsigned long seed) {
//     int idx = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < size) {
//         curandState state;
//         curand_init(seed, idx, 0, &state);
//         mat[idx] = curand_uniform(&state);
//     }
// }

// int main() {
//     float *d_A, *d_B, *d_C;
//     size_t bytes = N * N * sizeof(float);

//     CHECK_CUDA(cudaMalloc(&d_A, bytes));
//     CHECK_CUDA(cudaMalloc(&d_B, bytes));
//     CHECK_CUDA(cudaMalloc(&d_C, bytes));

//     // Launch kernels to init A and B with random floats
//     dim3 block(THREADS_PERa_Y);

//     init_random<<<grid, block>>>(d_A, N * N, 1234UL);
//     init_random<<<grid, block>>>(d_B, N * N, 5678UL);
//     CHECK_CUDA(cudaMemset(d_C, 0, bytes));
//     CHECK_CUDA(cudaDeviceSynchronize());

//     // Create cuBLAS handle
//     cublasHandle_t handle;
//     CHECK_CUBLAS(cublasCreate(&handle));
//     CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

//     float alpha = 1.0f, beta = 0.0f;

//     // Benchmark setup
//     cudaEvent_t start, stop;
//     CHECK_CUDA(cudaEventCreate(&start));
//     CHECK_CUDA(cudaEventCreate(&stop));

//     CHECK_CUDA(cudaEventRecord(start));

//     // GEMM: C = A * B
//     CHECK_CUBLAS(cublasGemmEx(
//         handle,
//         CUBLAS_OP_N, CUBLAS_OP_N,
//         N, N, N,
//         &alpha,
//         d_A, CUDA_R_32F, N,
//         d_B, CUDA_R_32F, N,
//         &beta,
//         d_C, CUDA_R_32F, N,
//         CUDA_R_32F,
//         CUBLAS_GEMM_DEFAULT_TENSOR_OP));

//     CHECK_CUDA(cudaEventRecord(stop));
//     CHECK_CUDA(cudaEventSynchronize(stop));

//     float ms = 0.0f;
//     CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

//     // Calculate GFLOPS: 2*N^3 ops / time
//     double gflops = (2.0 * N * N * N) / (ms / 1000.0) / 1e9;

//     std::cout << "Matrix multiplication completed in " << ms << " ms\n";
//     std::cout << "Performance: " << gflops << " GFLOPS\n";

//     // Cleanup
//     cublasDestroy(handle);
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);

//     return 0;
// }




