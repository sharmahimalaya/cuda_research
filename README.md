# CUDA Research Experiments 🚀

This repository contains **CUDA C/C++ research experiments** focused on understanding and benchmarking **GPU performance**, **memory behavior**, **thread/block configurations**, and **matrix multiplication optimizations**, including comparisons against **cuBLAS**.

The repository is organized as a collection of small, focused experiments rather than a single monolithic application.

---

## 📂 Repository Structure

```
.
├── 1dblocktiling.cu          # 1D block tiling implementation
├── 2dblocktiling.cu          # 2D block tiling implementation
├── cublas_implementation.cu  # cuBLAS-based matrix multiplication
├── cublas_sgemm.cu           # SGEMM using cuBLAS
├── events.cu                 # CUDA events and kernel timing
├── global_memory.cu          # Global memory access experiments
├── matmul_peak_tf32.cu       # TF32 peak performance experiments
├── matmulcourseversion.cu    # Reference / course-style matmul
├── mymatmulimplementation    # Custom matrix multiplication implementation
├── .vscode/                  # VSCode configuration
└── README.md
```

> Files without `.cu` extensions are the compiled binaries.

---

## 🎯 Objectives

* Explore **CUDA kernel design patterns**
* Study **1D vs 2D block tiling** strategies
* Understand **global vs shared memory** behavior
* Benchmark **custom GEMM kernels** against **cuBLAS**
* Measure kernel execution time using **CUDA events**
* Investigate **FP32 and TF32** performance characteristics

---

## 🧠 Topics Covered

* CUDA thread and block indexing
* Memory coalescing
* Shared memory tiling
* Kernel launch configuration
* CUDA event-based profiling
* cuBLAS (`cublasSgemm`)
* TF32 matrix multiplication
* Performance benchmarking

---

## 🛠 Requirements

* NVIDIA GPU with CUDA support
* CUDA Toolkit (CUDA 11+ recommended)
* `nvcc` compiler
* Linux or WSL environment (recommended)

Verify CUDA installation:

```bash
nvcc --version
nvidia-smi
```

---

## ⚙️ Compilation

Compile any CUDA source file using:

```bash
nvcc filename.cu -o filename
```

Example:

```bash
nvcc 2dblocktiling.cu -o 2dblocktiling
./2dblocktiling
```

### cuBLAS Programs

Link against cuBLAS explicitly:

```bash
nvcc cublas_sgemm.cu -lcublas -o cublas_sgemm
./cublas_sgemm
```

---

## 📊 Performance Measurement

Kernel execution time is measured using **CUDA Events**, providing accurate GPU-side timing. Some experiments aim to approach **theoretical peak throughput**, particularly for TF32 operations.

---

## 🔬 Research Notes

* This codebase is **experimental and educational**
* Focus is on exploration rather than production-level robustness
* Many kernels assume fixed matrix sizes or square matrices
* Error handling may be minimal in early experiments

---

## 🙌 Acknowledgements

* NVIDIA CUDA Programming Guide
* cuBLAS Documentation
* CUDA course and reference implementations
