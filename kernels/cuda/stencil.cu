#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

#include "cuda_helper.h"
#include "registrar.h"
#include "cuda_stencil.h"
#include "build_stencil_input.h"
#include "seed_values.h"

__global__ void stencil_kernel(double *A, double *B, int r, int c){
    int tid_x = blockDim.x * blockIdx.x + threadIdx.x;
    int tid_y = blockDim.y * blockIdx.y + threadIdx.y;

    if(tid_x < c - 1 && tid_x > 0 && tid_y < r - 1 && tid_y > 0){
        B[tid_y * c + tid_x] = (
            A[(tid_y - 1) * c + tid_x]     + 
            A[(tid_y + 1) * c + tid_x]     + 
            A[tid_y * c + (tid_x - 1)]     + 
            A[tid_y * c + (tid_x + 1)]       
            ) / 4.0;
    }
}

std::vector<double> cuda_stencil(
    const std::vector<double>& h_A,
    int r,
    int c,
    int iterations
){
    if (r <= 0 || c <= 0) {
        throw std::runtime_error("Empty matrix");
    }

    if (iterations < 0) {
        throw std::runtime_error("Iterations cannot be negative");
    }

    const std::size_t total_elements =
        static_cast<std::size_t>(r) * static_cast<std::size_t>(c);

    if (h_A.size() != total_elements) {
        throw std::runtime_error("Matrix dimensions do not match data");
    }

    const std::size_t size_bytes = sizeof(double) * total_elements;

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid(
        (c + threads_per_block.x - 1) / threads_per_block.x,
        (r + threads_per_block.y - 1) / threads_per_block.y
    );

    std::vector<double> h_B(total_elements, 0.0);

    if (r >= 3 && c >= 3) {
        for (int j = 0; j < c; ++j) {
            h_B[j] = h_A[j];
            h_B[(r - 1) * c + j] = h_A[(r - 1) * c + j];
        }

        for (int i = 0; i < r; ++i) {
            h_B[i * c] = h_A[i * c];
            h_B[i * c + (c - 1)] = h_A[i * c + (c - 1)];
        }
    } else {
        h_B = h_A;
    }

    double* d_A = nullptr;
    double* d_B = nullptr;

    try {
        cuda_check(
            cudaMalloc(reinterpret_cast<void**>(&d_A), size_bytes),
            "CUDA Stencil: Failed to allocate device memory for A"
        );

        cuda_check(
            cudaMalloc(reinterpret_cast<void**>(&d_B), size_bytes),
            "CUDA Stencil: Failed to allocate device memory for B"
        );

        cuda_check(
            cudaMemcpy(d_A, h_A.data(), size_bytes, cudaMemcpyHostToDevice),
            "CUDA Stencil: Failed to copy A to device"
        );

        cuda_check(
            cudaMemcpy(d_B, h_B.data(), size_bytes, cudaMemcpyHostToDevice),
            "CUDA Stencil: Failed to copy B to device"
        );

        for (int i = 0; i < iterations; ++i) {
            stencil_kernel<<<blocks_per_grid, threads_per_block>>>(
                d_A, d_B, r, c
            );

            cuda_check(
                cudaGetLastError(),
                "CUDA Stencil: Kernel launch failed"
            );

            double* temp = d_A;
            d_A = d_B;
            d_B = temp;
        }

        cuda_check(
            cudaMemcpy(h_B.data(), d_A, size_bytes, cudaMemcpyDeviceToHost),
            "CUDA Stencil: Failed to copy result from device"
        );

        cuda_check(cudaFree(d_A), "CUDA Stencil: Failed to free device buffer A");
        d_A = nullptr;

        cuda_check(cudaFree(d_B), "CUDA Stencil: Failed to free device buffer B");
        d_B = nullptr;

        return h_B;
    }
    catch (...) {
        if (d_A != nullptr) {
            cudaFree(d_A);
        }

        if (d_B != nullptr) {
            cudaFree(d_B);
        }

        throw;
    }
}

std::vector<double> cuda_stencil_wrapper(
    BenchmarkContext& ctx,
    int r,
    int c,
    int iterations
){
    if (r <= 0 || c <= 0) {
        throw std::runtime_error("Empty matrix");
    }

    if (iterations < 0) {
        throw std::runtime_error("Iterations cannot be negative");
    }

    ctx.iterations = iterations;
    ctx.problem_size = r * c;
    ctx.number_of_processors = get_number_of_processors(r * c);

    std::vector<double> input = std::move(*build_stencil_input(r, c, STENCIL));

    ctx.start_measurement();

    std::vector<double> result = cuda_stencil(input, r, c, iterations);

    ctx.stop_measurement();

    
    ctx.bytes_moved = 2.0 * r * c * sizeof(double) * iterations;
    ctx.flop_count = 4.0 * (r - 2) * (c - 2) * iterations;

    return result;
}

BENCHMARK_TEST(stencil_cuda_n32x32_iter10,
               stencil_n32x32_iter10,
               Paradigm::CUDA) {
    cuda_stencil_wrapper(ctx, 32, 32, 10);
}

BENCHMARK_TEST(stencil_cuda_n64x64_iter10,
               stencil_n64x64_iter10,
               Paradigm::CUDA) {
    cuda_stencil_wrapper(ctx, 64, 64, 10);
}

BENCHMARK_TEST(stencil_cuda_n128x128_iter10,
               stencil_n128x128_iter10,
               Paradigm::CUDA) {
    cuda_stencil_wrapper(ctx, 128, 128, 10);
}

BENCHMARK_TEST(stencil_cuda_n256x256_iter10,
               stencil_n256x256_iter10,
               Paradigm::CUDA) {
    cuda_stencil_wrapper(ctx, 256, 256, 10);
}

BENCHMARK_TEST(stencil_cuda_n512x512_iter10,
               stencil_n512x512_iter10,
               Paradigm::CUDA) {
    cuda_stencil_wrapper(ctx, 512, 512, 10);
}

BENCHMARK_TEST(stencil_cuda_n1024x1024_iter10,
               stencil_n1024x1024_iter10,
               Paradigm::CUDA) {
    cuda_stencil_wrapper(ctx, 1024, 1024, 10);
}

BENCHMARK_TEST(stencil_cuda_n256x256_iter1,
               stencil_n256x256_iter1,
               Paradigm::CUDA) {
    cuda_stencil_wrapper(ctx, 256, 256, 1);
}

BENCHMARK_TEST(stencil_cuda_n256x256_iter50,
               stencil_n256x256_iter50,
               Paradigm::CUDA) {
    cuda_stencil_wrapper(ctx, 256, 256, 50);
}

BENCHMARK_TEST(stencil_cuda_n256x256_iter100,
               stencil_n256x256_iter100,
               Paradigm::CUDA) {
    cuda_stencil_wrapper(ctx, 256, 256, 100);
}
