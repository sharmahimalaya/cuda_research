//adding two vectors using CUDA
#include<cuda.h>
#include<cuda_runtime.h>
#include<iostream>
#include<vector>
#include<memory>
#include<time.h>

using namespace std;

// Returns current time in seconds as a double
double get_time() {
    return static_cast<double>(clock()) / CLOCKS_PER_SEC;
}

void vectoraddCPU(vector<float> &h_a, vector<float> &h_b, vector<float> &h_c, int N){
    for(int i = 0; i < N; i++){
        h_c[i] = h_a[i] + h_b[i];
    }
}

__global__ void vectoraddGPU(float *d_a, float *d_b, float *d_c, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N){
        d_c[i] = d_a[i] + d_b[i];
    }
}

int main(){
    int N = 1024*1024;
    vector<float> h_a(N, 0.0f);
    vector<float> h_b(N, 0.0f);
    vector<float> h_c_cpu(N, 0.0f);
    vector<float> h_c_gpu(N, 0.0f);

    size_t size = N*sizeof(float);
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    for (int i = 0; i < N; i++) {
        h_b[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)rand() / RAND_MAX;
    }

    cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyKind::cudaMemcpyHostToDevice);


     printf("Performing warm-up runs.\n");
    for (int i = 0; i < 3; i++) {
        vectoraddCPU(h_a, h_b, h_c_cpu, N);
        vectoraddGPU<<<(N+256-1)/256, 256>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }

    printf("Benchmarking CPU implementation\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 200; i++) {
        double start_time = get_time();
        vectoraddCPU(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 200.0;

    printf("Benchmarking GPU implementation\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 200; i++) {
        double start_time = get_time();
        vectoraddGPU<<<(N+256-1)/256, 256>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 200.0;

    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    cudaMemcpy(h_c_gpu.data(), d_c, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-20) {
            correct = false;
            break;
        }
    }
    printf("Results are %s\n", correct ? "correct" : "incorrect");
    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}