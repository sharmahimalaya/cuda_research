// #include<stdio.h>
// #include<stdlib.h>
// #include<assert.h>
// #include<cuda_runtime.h>
// #include<time.h>
// #define M 1024
// #define K 1024
// #define N 1024

// double get_time() {
//     struct timespec ts;
//     clock_gettime(CLOCK_MONOTONIC, &ts);
//     return ts.tv_sec*1e3 + ts.tv_nsec * 1e-6;
// }

// __global__ void blocktilingmulti(float *mat1p, float *mat2p, float *resMatp){
//     const uint comTileDim = 4; // size of the common dimension, number of columns in blocktile A and rows in blocktile B
//     const uint rowsPerThread = 4; // number of rows per thread
//     const uint colsPerThread = 4; // number of columns per thread
//     const uint tileRow = 64; // number of rows in the blocktile
//     const uint tileCols = 64; // number of columns in the blocktile

//     const uint cRow = blockIdx.y; // current row
//     const uint cCol = blockIdx.x; // current column

//     // const uint totalResultsBlocktile = tileRow * tileCols;
//     // Each thread is responsible for calculating rowsPerThread*colsPerThread elements in the blocktile
//     const uint numThreadsBlocktile = (tileRow * tileCols) / (rowsPerThread * colsPerThread);

//     // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
//     assert(numThreadsBlocktile == blockDim.x);
 
//     // BN/colsPerThread are the number of threads to span a column
//     const int threadCol = threadIdx.x % (tileCols / colsPerThread);
//     const int threadRow = threadIdx.x / (tileCols / colsPerThread);

//     // allocate space for the current blocktile in smem
//     __shared__ float tileA[tileRow * comTileDim];
//     __shared__ float tileB[comTileDim * tileCols];

//     // Move blocktile to beginning of A's row and B's column that is now being processed
//     mat1p += cRow * tileRow * K;
//     mat2p += cCol * tileCols;
//     resMatp += cRow * tileRow * N + cCol * tileCols;

//     // calculating the indices that this thread will load into the tiles, and which correspond to the inner tiles
//     const uint innerRowA = threadIdx.x / comTileDim;
//     const uint innerColA = threadIdx.x % comTileDim;
//     // calculates the number of rows of tileA that are being loaded in a single step
//     // by a single block
//     const uint strideA = numThreadsBlocktile / comTileDim;
//     const uint innerRowB = threadIdx.x / tileCols;
//     const uint innerColB = threadIdx.x % tileCols;
//     const uint strideB = numThreadsBlocktile / tileCols;

//     // allocate thread-local cache for results in registerfile
//     float threadResults[rowsPerThread * colsPerThread] = {0.0};
//     // register caches for tileA and tileB
//     float regM[rowsPerThread] = {0.0};
//     float regN[colsPerThread] = {0.0};

//     // outer-most loop over block tiles
//     for (uint bkIdx = 0; bkIdx < K; bkIdx += comTileDim) {
//         // populate the tiles with respective values
//         for (uint loadOffset = 0; loadOffset < tileRow; loadOffset += strideA) {
//             tileA[(innerRowA + loadOffset) * comTileDim + innerColA] = mat1p[(innerRowA + loadOffset) * K + innerColA];
//         }
//         for (uint loadOffset = 0; loadOffset < comTileDim; loadOffset += strideB) {
//             tileB[(innerRowB + loadOffset) * tileCols + innerColB] = mat2p[(innerRowB + loadOffset) * N + innerColB];
//         }
//         __syncthreads();

//         // advance blocktile
//         mat1p += comTileDim;     // move BK columns to right
//         mat2p += comTileDim * N; // move BK rows down

//         // calculate per-thread results
//         for (uint dotIdx = 0; dotIdx < comTileDim; ++dotIdx) {
//             // block into registers
//             for (uint i = 0; i < rowsPerThread; ++i) {
//                 regM[i] = tileA[(threadRow * rowsPerThread + i) * comTileDim + dotIdx];
//             }
//             for (uint i = 0; i < colsPerThread; ++i) {
//                 regN[i] = tileB[dotIdx * tileCols + threadCol * colsPerThread + i];
//             }
//             for (uint resIdxM = 0; resIdxM < rowsPerThread; ++resIdxM) {
//                 for (uint resIdxN = 0; resIdxN < colsPerThread; ++resIdxN) {
//                     threadResults[resIdxM * colsPerThread + resIdxN] += regM[resIdxM] * regN[resIdxN];
//                 }
//             }
//         }
//         __syncthreads();
//     }

//     // write out the results
//     for (uint resIdxM = 0; resIdxM < rowsPerThread; ++resIdxM) {
//         for (uint resIdxN = 0; resIdxN < colsPerThread; ++resIdxN) {
//             resMatp[(threadRow * rowsPerThread + resIdxM) * N + threadCol * colsPerThread + resIdxN] =threadResults[resIdxM * colsPerThread + resIdxN];
//         }
//     }
// }

// int main(){
//     float *d_a, *d_b, *d_c;
//     float *A, *B;   

//     srand(time(NULL));

//     A = (float*)malloc(M * K * sizeof(float));
//     B = (float*)malloc(K * N * sizeof(float));
//     // adding random values to A and B
//     for(int i = 0; i < M*K; i++) A[i] = static_cast<float>(rand()) / RAND_MAX;
//     for(int i = 0; i < K*N; i++) B[i] = static_cast<float>(rand()) / RAND_MAX;
//     // allocating device memory
//     cudaMalloc((void**)&d_a, M * K * sizeof(float));
//     cudaMalloc((void**)&d_b, K * N * sizeof(float));
//     cudaMalloc((void**)&d_c, M * N * sizeof(float));
//     // copying data from host to device
//     cudaMemcpy(d_a, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
//     // declaring dimensions for the grid and block
//     dim3 blockSize(256);
//     dim3 gridSize(8,8);
//     //warm up
//     blocktilingmulti<<<gridSize, blockSize>>>(d_a, d_b, d_c);
//     cudaDeviceSynchronize();
//     cudaMemset(d_c, 0, M * N * sizeof(float));
//     double start_time = get_time();
//     blocktilingmulti<<<gridSize, blockSize>>>(d_a, d_b, d_c);
//     cudaDeviceSynchronize();
//     double end_time = get_time();
//     double elapsed_ms = end_time - start_time;
//     double num_ops = 2.0 * M * N * K;
//     double gflops = num_ops / (elapsed_ms * 1e6);
//     printf("Time taken: %f ms\n", end_time - start_time);
//     printf("Performance: %.2f GFLOPS\n", gflops);
//     free(A);
//     free(B);
//     cudaFree(d_a);
//     cudaFree(d_b);
//     cudaFree(d_c);
//     return 0;
// }



// #include <stdio.h>
// #include <stdlib.h>
// #include <cuda_runtime.h>
// #include <assert.h>
// #include <time.h>

// #define M 1024
// #define K 1024
// #define N 1024

// __global__ void blocktilingmulti(float *mat1p, float *mat2p, float *resMatp) {
//     const uint comTileDim = 4;
//     const uint rowsPerThread = 4;
//     const uint colsPerThread = 4;
//     const uint tileRow = 64;
//     const uint tileCols = 64;

//     const uint cRow = blockIdx.y;
//     const uint cCol = blockIdx.x;

//     const uint numThreadsBlocktile = (tileRow * tileCols) / (rowsPerThread * colsPerThread);
//     assert(numThreadsBlocktile == blockDim.x);

//     const int threadCol = threadIdx.x % (tileCols / colsPerThread);
//     const int threadRow = threadIdx.x / (tileCols / colsPerThread);

//     __shared__ float tileA[tileRow * comTileDim];
//     __shared__ float tileB[comTileDim * tileCols];

//     mat1p += cRow * tileRow * K;
//     mat2p += cCol * tileCols;
//     resMatp += cRow * tileRow * N + cCol * tileCols;

//     const uint innerRowA = threadIdx.x / comTileDim;
//     const uint innerColA = threadIdx.x % comTileDim;
//     const uint strideA = numThreadsBlocktile / comTileDim;

//     const uint innerRowB = threadIdx.x / tileCols;
//     const uint innerColB = threadIdx.x % tileCols;
//     const uint strideB = numThreadsBlocktile / tileCols;

//     float threadResults[rowsPerThread * colsPerThread] = {0.0f};
//     float regM[rowsPerThread], regN[colsPerThread];

//     for (uint bkIdx = 0; bkIdx < K; bkIdx += comTileDim) {
//         for (uint loadOffset = 0; loadOffset < tileRow; loadOffset += strideA) {
//             tileA[(innerRowA + loadOffset) * comTileDim + innerColA] =
//                 mat1p[(innerRowA + loadOffset) * K + innerColA];
//         }
//         for (uint loadOffset = 0; loadOffset < comTileDim; loadOffset += strideB) {
//             tileB[(innerRowB + loadOffset) * tileCols + innerColB] =
//                 mat2p[(innerRowB + loadOffset) * N + innerColB];
//         }

//         __syncthreads();

//         mat1p += comTileDim;
//         mat2p += comTileDim * N;

//         for (uint dotIdx = 0; dotIdx < comTileDim; ++dotIdx) {
//             for (uint i = 0; i < rowsPerThread; ++i) {
//                 regM[i] = tileA[(threadRow * rowsPerThread + i) * comTileDim + dotIdx];
//             }
//             for (uint i = 0; i < colsPerThread; ++i) {
//                 regN[i] = tileB[dotIdx * tileCols + threadCol * colsPerThread + i];
//             }
//             for (uint resIdxM = 0; resIdxM < rowsPerThread; ++resIdxM) {
//                 for (uint resIdxN = 0; resIdxN < colsPerThread; ++resIdxN) {
//                     threadResults[resIdxM * colsPerThread + resIdxN] +=
//                         regM[resIdxM] * regN[resIdxN];
//                 }
//             }
//         }

//         __syncthreads();
//     }

//     for (uint resIdxM = 0; resIdxM < rowsPerThread; ++resIdxM) {
//         for (uint resIdxN = 0; resIdxN < colsPerThread; ++resIdxN) {
//             resMatp[(threadRow * rowsPerThread + resIdxM) * N + threadCol * colsPerThread + resIdxN] =
//                 threadResults[resIdxM * colsPerThread + resIdxN];
//         }
//     }
// }

// int main() {
//     float *A, *B, *d_A, *d_B, *d_C;
//     A = (float*)malloc(M * K * sizeof(float));
//     B = (float*)malloc(K * N * sizeof(float));

//     // Seed and initialize
//     srand(42);
//     for (int i = 0; i < M * K; ++i) A[i] = (float)rand() / RAND_MAX;
//     for (int i = 0; i < K * N; ++i) B[i] = (float)rand() / RAND_MAX;

//     // Allocate device memory
//     cudaMalloc(&d_A, M * K * sizeof(float));
//     cudaMalloc(&d_B, K * N * sizeof(float));
//     cudaMalloc(&d_C, M * N * sizeof(float));

//     cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

//     dim3 blockSize(256);
//     dim3 gridSize(8, 8);

//     // Warmup
//     blocktilingmulti<<<gridSize, blockSize>>>(d_A, d_B, d_C);
//     cudaDeviceSynchronize();

//     cudaMemset(d_C, 0, M * N * sizeof(float));

//     // Accurate timing with CUDA events
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);

//     blocktilingmulti<<<gridSize, blockSize>>>(d_A, d_B, d_C);

//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);

//     float elapsed_ms = 0.0f;
//     cudaEventElapsedTime(&elapsed_ms, start, stop);

//     double num_ops = 2.0 * M * N * K;
//     double gflops = num_ops / (elapsed_ms * 1e6);

//     printf("Time: %.3f ms\n", elapsed_ms);
//     printf("Performance: %.2f GFLOPS\n", gflops);

//     // Optional: verify output with CPU (disabled by default)
//     /*
//     float *C_host = (float*)malloc(M * N * sizeof(float));
//     cudaMemcpy(C_host, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
//     // verify correctness here...
//     free(C_host);
//     */

//     // Cleanup
//     free(A);
//     free(B);
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);

//     return 0;
// }


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <time.h>

#define M 6144
#define K 6144
#define N 6144

__global__ void blocktilingmulti(float *mat1p, float *mat2p, float *resMatp) {
    const uint comTileDim = 4;
    const uint rowsPerThread = 4;
    const uint colsPerThread = 4;
    const uint tileRow = 64;
    const uint tileCols = 64;

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint numThreadsBlocktile = (tileRow * tileCols) / (rowsPerThread * colsPerThread);
    assert(numThreadsBlocktile == blockDim.x);

    const int threadCol = threadIdx.x % (tileCols / colsPerThread);
    const int threadRow = threadIdx.x / (tileCols / colsPerThread);

    __shared__ float tileA[tileRow * comTileDim];
    __shared__ float tileB[comTileDim * tileCols];

    mat1p += cRow * tileRow * K;
    mat2p += cCol * tileCols;
    resMatp += cRow * tileRow * N + cCol * tileCols;

    const uint innerRowA = threadIdx.x / comTileDim;
    const uint innerColA = threadIdx.x % comTileDim;
    const uint strideA = numThreadsBlocktile / comTileDim;

    const uint innerRowB = threadIdx.x / tileCols;
    const uint innerColB = threadIdx.x % tileCols;
    const uint strideB = numThreadsBlocktile / tileCols;

    float threadResults[rowsPerThread * colsPerThread] = {0.0f};
    float regM[rowsPerThread], regN[colsPerThread];

    for (uint bkIdx = 0; bkIdx < K; bkIdx += comTileDim) {
        for (uint loadOffset = 0; loadOffset < tileRow; loadOffset += strideA) {
            tileA[(innerRowA + loadOffset) * comTileDim + innerColA] =
                mat1p[(innerRowA + loadOffset) * K + innerColA];
        }
        for (uint loadOffset = 0; loadOffset < comTileDim; loadOffset += strideB) {
            tileB[(innerRowB + loadOffset) * tileCols + innerColB] =
                mat2p[(innerRowB + loadOffset) * N + innerColB];
        }

        __syncthreads();

        mat1p += comTileDim;
        mat2p += comTileDim * N;

        for (uint dotIdx = 0; dotIdx < comTileDim; ++dotIdx) {
            for (uint i = 0; i < rowsPerThread; ++i) {
                regM[i] = tileA[(threadRow * rowsPerThread + i) * comTileDim + dotIdx];
            }
            for (uint i = 0; i < colsPerThread; ++i) {
                regN[i] = tileB[dotIdx * tileCols + threadCol * colsPerThread + i];
            }
            for (uint resIdxM = 0; resIdxM < rowsPerThread; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < colsPerThread; ++resIdxN) {
                    threadResults[resIdxM * colsPerThread + resIdxN] +=
                        regM[resIdxM] * regN[resIdxN];
                }
            }
        }

        __syncthreads();
    }

    for (uint resIdxM = 0; resIdxM < rowsPerThread; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < colsPerThread; ++resIdxN) {
            resMatp[(threadRow * rowsPerThread + resIdxM) * N + threadCol * colsPerThread + resIdxN] =
                threadResults[resIdxM * colsPerThread + resIdxN];
        }
    }
}

int main() {
    float *A, *B, *d_A, *d_B, *d_C;
    A = (float*)malloc(M * K * sizeof(float));
    B = (float*)malloc(K * N * sizeof(float));

    // Seed and initialize
    srand(time(NULL));
    for (int i = 0; i < M * K; ++i) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i) B[i] = (float)rand() / RAND_MAX;

    // Allocate device memory
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize(96, 96);  // 6144 / 64 = 96 tiles in each dimension

    // Warmup
    blocktilingmulti<<<gridSize, blockSize>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemset(d_C, 0, M * N * sizeof(float));

    // Accurate timing with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    blocktilingmulti<<<gridSize, blockSize>>>(d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    double num_ops = 2.0 * M * N * K;
    double gflops = num_ops / (elapsed_ms * 1e6);

    printf("Time: %.3f ms\n", elapsed_ms);
    printf("Performance: %.2f GFLOPS\n", gflops);

    // Cleanup
    free(A);
    free(B);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
