#include <cuda_runtime.h>
#include <stdio.h>

// Dummy compute-heavy kernel
__global__ void heavyKernel(float *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        float val = 0.0f;
        for (int i = 0; i < 100000; ++i) {
            val += 0.0001f * i;
        }
        data[idx] = val;
    }
}

int main() {
    const int N = 1 << 14; // Number of elements (16K)
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    float *d_data1, *d_data2;
    cudaMalloc(&d_data1, N * sizeof(float));
    cudaMalloc(&d_data2, N * sizeof(float));

    // Create streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Events for timing
    cudaEvent_t start, stop, s1_start, s1_stop, s2_start, s2_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&s1_start);
    cudaEventCreate(&s1_stop);
    cudaEventCreate(&s2_start);
    cudaEventCreate(&s2_stop);

    cudaEventRecord(start);

    // Stream 1
    cudaEventRecord(s1_start, stream1);
    heavyKernel<<<blocks, threads, 0, stream1>>>(d_data1, N);
    cudaEventRecord(s1_stop, stream1);

    // Stream 2
    cudaEventRecord(s2_start, stream2);
    heavyKernel<<<blocks, threads, 0, stream2>>>(d_data2, N);
    cudaEventRecord(s2_stop, stream2);

    cudaEventRecord(stop);

    // Wait for all to finish
    cudaEventSynchronize(stop);

    float time_total = 0.0f, time1 = 0.0f, time2 = 0.0f;
    cudaEventElapsedTime(&time_total, start, stop);
    cudaEventElapsedTime(&time1, s1_start, s1_stop);
    cudaEventElapsedTime(&time2, s2_start, s2_stop);

    printf("Kernel 1 time: %.2f ms\n", time1);
    printf("Kernel 2 time: %.2f ms\n", time2);
    printf("Total (overlapped) time: %.2f ms\n", time_total);

    // Clean up
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(s1_start);
    cudaEventDestroy(s1_stop);
    cudaEventDestroy(s2_start);
    cudaEventDestroy(s2_stop);

    return 0;
}
