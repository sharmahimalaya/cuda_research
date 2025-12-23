#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>

__global__ void whoami(void){
    int block_offset = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int thread_offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int id = thread_offset + block_offset;
    printf("Hello from thread %d in block %d\n", id, block_offset);
}

int main(){
    
    whoami<<<24,64>>>();
    cudaDeviceSynchronize();
}