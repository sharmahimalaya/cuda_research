#include<stdio.h>
#include<cuda_runtime.h>
#include<time.h>
#include<stdlib.h>

#define M 1024 // rows in A and C
#define K 1024 // columns in A and rows in B
#define N 1024 // columns in B and C
#define TILE_SIZE 16

using namespace std;

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1e3 + ts.tv_nsec * 1e-6;
}

// tile multiplication using global coalesced memory access but turned out to be non-coalesced and hence slower 
__global__ void globalCoalescedMultiplication(float *A, float *B, float *C){
    // this time we will use flattened tiles as well
    __shared__ float tileA[TILE_SIZE*TILE_SIZE];
    __shared__ float tileB[TILE_SIZE*TILE_SIZE];

    int cRow = blockIdx.y* TILE_SIZE + threadIdx.x/TILE_SIZE;
    int cCol = blockIdx.x* TILE_SIZE + threadIdx.x%TILE_SIZE;
    float sum = 0.0f;
    for(int i=0;i<K; i+= TILE_SIZE){
        if(cRow < M && i + threadIdx.x < K){
            tileA[threadIdx.x + threadIdx.x%TILE_SIZE] = A[cRow*K + i + threadIdx.x/TILE_SIZE];
        } else {
            tileA[threadIdx.x + threadIdx.x%TILE_SIZE] = 0.0f;
        }
        if(i + threadIdx.x/TILE_SIZE < K && cCol < N){
            tileB[threadIdx.x + threadIdx.x%TILE_SIZE] = B[(i + threadIdx.x/TILE_SIZE)*N + cCol];
        } else {
            tileB[threadIdx.x + threadIdx.x%TILE_SIZE] = 0.0f;
        }
        __syncthreads();
        for(int j=0;j<TILE_SIZE;j++){
            sum += tileA[threadIdx.x + j*TILE_SIZE] * tileB[j + threadIdx.x%TILE_SIZE];
        }
        __syncthreads();
    }
    if(cRow < M && cCol < N){
        C[cRow*N + cCol] = sum;
    }
}


//  partial tile dotproduct with shared memory and coalesced memory access
__global__ void tileMultiplication(float *A, float *B, float *C){
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for(int i=0;i<(K + TILE_SIZE - 1)/TILE_SIZE;i++){
        if(row < M && i*TILE_SIZE + threadIdx.x < K){
            tileA[threadIdx.y][threadIdx.x] = A[row*K + i*TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if(i*TILE_SIZE + threadIdx.y < (K + TILE_SIZE - 1)/TILE_SIZE && col < N){
            tileB[threadIdx.y][threadIdx.x] = B[(i*TILE_SIZE + threadIdx.y)*N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        for (int j = 0; j < TILE_SIZE; j++) {
            sum += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
    
}

int main(){
    float *h_a, *h_b;
    float *d_a, *d_b, *d_c;

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);    

    h_a = (float*)malloc(size_a);
    h_b = (float*)malloc(size_b);
    cudaMalloc((void**)&d_a, size_a);
    cudaMalloc((void**)&d_b, size_b);
    cudaMalloc((void**)&d_c, size_c);
    for(int i = 0; i < M*K; i++) h_a[i] = static_cast<float>(rand()) / RAND_MAX;
    for(int i = 0; i < K*N; i++) h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // double start_time = get_time();
    // globalCoalescedMultiplication<<<gridSize, blockSize>>>(d_a, d_b, d_c);
    // cudaDeviceSynchronize();
    // double end_time = get_time();

    // printf("Time taken: %f milliseconds\n", end_time - start_time);

    tileMultiplication<<<gridSize, blockSize>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    double start_time = get_time();
    tileMultiplication<<<gridSize, blockSize>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    double end_time = get_time();
    printf("Time taken: %f milliseconds\n", end_time - start_time);
    // cudaDeviceSynchronize();
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    return 0;
}