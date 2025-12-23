#include<stdio.h>
#include<cuda_runtime.h>
#include<time.h>
#include<stdlib.h>

using namespace std;
#define BLOCK_SIZE 16
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1e3 + ts.tv_nsec * 1e-6;
}

__global__ void coalescingmulti(float *A, float *B, float *C){
    int cRow = blockIdx.x * BLOCK_SIZE + (threadIdx.x/BLOCK_SIZE);
    int cCol = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);
    __syncthreads();

    printf("cRow: %d, cCol: %d, threadIdx.x: %d, blockIdx.x: %d, blockIdx.y: %d\n", cRow, cCol, threadIdx.x, blockIdx.x, blockIdx.y);
}

int main(){ 
    dim3 grid(2,2);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    coalescingmulti<<<grid,block>>>(nullptr, nullptr, nullptr);
    cudaDeviceSynchronize();
}