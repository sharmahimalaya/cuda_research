// #include <cstdio>
// #include <cstdlib>
// #include <ctime>
// #include <cublas_v2.h>
// #include <cuda_runtime.h>

// int main() {
//   const int m = 1024;
//   const int k = 1024;
//   const int n = 1024;

//   float *a = (float *)malloc(m * k * sizeof(float));
//   float *b = (float *)malloc(k * n * sizeof(float));
//   float *c = (float *)malloc(m * n * sizeof(float));

//   srand((unsigned int)time(NULL));
//   for (int i = 0; i < m * k; i++) a[i] = static_cast<float>(rand()) / RAND_MAX;
//   for (int i = 0; i < k * n; i++) b[i] = static_cast<float>(rand()) / RAND_MAX;
//   for (int i = 0; i < m * n; i++) c[i] = 0.0f;

//   float *d_a, *d_b, *d_c;
//   cudaMalloc((void **)&d_a, m * k * sizeof(float));
//   cudaMalloc((void **)&d_b, k * n * sizeof(float));
//   cudaMalloc((void **)&d_c, m * n * sizeof(float));

//   cublasHandle_t handle;
//   cublasCreate(&handle);

//   cudaMemcpy(d_a, a, m * k * sizeof(float), cudaMemcpyHostToDevice);
//   cudaMemcpy(d_b, b, k * n * sizeof(float), cudaMemcpyHostToDevice);
//   cudaMemcpy(d_c, c, m * n * sizeof(float), cudaMemcpyHostToDevice);

//   float alpha = 1.0f;
//   float beta = 0.0f;

//   // Timing using CUDA events
//   cudaEvent_t start, stop;
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);
//   cudaEventRecord(start, 0);

//   // Perform SGEMM: C = alpha * A * B + beta * C
//   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
//               n, m, k,
//               &alpha,
//               d_b, n,
//               d_a, k,
//               &beta,
//               d_c, n);

//   cudaEventRecord(stop, 0);
//   cudaEventSynchronize(stop);

//   float elapsedTime;
//   cudaEventElapsedTime(&elapsedTime, start, stop); // ms

//   // Calculate GFLOPS: 2*m*n*k operations
//   double gflops = 2.0 * m * n * k / (elapsedTime / 1000.0) / 1e9;
//   printf("Time taken: %.3f ms\n", elapsedTime);
//   printf("Performance: %.2f GFLOPS\n", gflops);


//   cudaFree(d_a);
//   cudaFree(d_b);
//   cudaFree(d_c);
//   cublasDestroy(handle);
//   free(a);
//   free(b);
//   free(c);

//   return EXIT_SUCCESS;
// }



#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <cuda_runtime.h>

int main() {
  const int m = 6144, k = 6144, n = 6144;

  float *a = (float *)malloc(m * k * sizeof(float));
  float *b = (float *)malloc(k * n * sizeof(float));
  float *c = (float *)malloc(m * n * sizeof(float));

  srand((unsigned int)time(NULL));
  for (int i = 0; i < m * k; i++) a[i] = static_cast<float>(rand()) / RAND_MAX;
  for (int i = 0; i < k * n; i++) b[i] = static_cast<float>(rand()) / RAND_MAX;

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, m * k * sizeof(float));
  cudaMalloc(&d_b, k * n * sizeof(float));
  cudaMalloc(&d_c, m * n * sizeof(float));

  cudaMemcpy(d_a, a, m * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, k * n * sizeof(float), cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f;
  float beta = 0.0f;

    cublasSgemm(handle, 
              CUBLAS_OP_T, CUBLAS_OP_T,
              n, m, k,
              &alpha,
              d_b, k,   // Transposed view of B (n×k)
              d_a, m,   // Transposed view of A (k×m)
              &beta,
              d_c, n);

  // CUDA timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Perform SGEMM with transposed operands to match row-major layout
  cublasSgemm(handle, 
              CUBLAS_OP_T, CUBLAS_OP_T,
              n, m, k,
              &alpha,
              d_b, k,   // Transposed view of B (n×k)
              d_a, m,   // Transposed view of A (k×m)
              &beta,
              d_c, n);  // Result (n×m) interpreted as row-major m×n

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop); // ms

  double gflops = 2.0 * m * n * k / (elapsedTime / 1000.0) / 1e9;
  printf("Time taken: %.3f ms\n", elapsedTime);
  printf("Performance: %.2f GFLOPS\n", gflops);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cublasDestroy(handle);
  free(a);
  free(b);
  free(c);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return EXIT_SUCCESS;
}
