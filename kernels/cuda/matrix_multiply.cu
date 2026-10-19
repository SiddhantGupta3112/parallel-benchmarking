#include <cuda_runtime.h>
#include "registrar.h"
#include "cuda_matrix_multiply.h"
#include "cuda_helper.h"
#include "build_matrix.h"
#include "seed_values.h"

__global__ void matrix_muliply_kernel(double *A, double *B, double *C, int r1, int c1, int r2, int c2){
    int tid_x = blockDim.x * blockIdx.x + threadIdx.x;
    int tid_y = blockDim.y * blockIdx.y + threadIdx.y;

    if(tid_x < c2 && tid_y < r1){
        double sum = 0.0;
        for (int k = 0; k < c1; k++){
            sum += A[tid_y * c1 + k] * B[k * c2 + tid_x];
        }

        C[tid_y * c2 + tid_x] = sum;
    }
}

std::vector<double> cuda_matrix_multiply(
    const std::vector<double>& h_A,
    const std::vector<double>& h_B,
    int r1,
    int c1,
    int r2,
    int c2
){
    if (r1 <= 0 || c1 <= 0 || r2 <= 0 || c2 <= 0) {
        throw std::runtime_error("Empty matrix");
    }

    if (c1 != r2) {
        throw std::runtime_error("Invalid matrix sizes");
    }

    if (h_A.size() != static_cast<std::size_t>(r1) * c1 ||
        h_B.size() != static_cast<std::size_t>(r2) * c2) {
        throw std::runtime_error("Matrix dimensions do not match data");
    }

    dim3 threads_per_block(16, 16); 
    dim3 blocks_per_grid(
        (c2 + threads_per_block.x - 1) / threads_per_block.x, 
        (r1 + threads_per_block.y - 1) / threads_per_block.y  
    );

    std::vector<double> h_C(
    static_cast<std::size_t>(r1) * c2, 0.0);

    double *d_A = nullptr;
    double *d_B = nullptr;
    double *d_C = nullptr;

    try{
        cuda_check(cudaMalloc((void**)& d_A, sizeof(double) * r1 * c1), "Failed to allocate device memory for matrix A");
        cuda_check(cudaMalloc((void**)& d_B, sizeof(double) * r2 * c2), "Failed to allocate device memory for matrix B");
        cuda_check(cudaMalloc((void**)& d_C, sizeof(double) * r1 * c2), "Failed to allocate device memory for matrix C");

        cuda_check(cudaMemcpy(d_A, h_A.data(), sizeof(double) * r1 * c1, cudaMemcpyHostToDevice), "Failed to copy matrix A to device");
        cuda_check(cudaMemcpy(d_B, h_B.data(), sizeof(double) * r2 * c2, cudaMemcpyHostToDevice), "Failed to copy matrix B to device");

        matrix_muliply_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, r1, c1, r2, c2);
        cuda_check(cudaGetLastError(), "Matrix multiplication launch failed");

        cuda_check(cudaDeviceSynchronize(), "Failed to synchronize device");
        cuda_check(cudaMemcpy(h_C.data(), d_C, sizeof(double) * r1 * c2, cudaMemcpyDeviceToHost), "Failed to copy matrix C from device");

        
        cuda_check(cudaFree(d_A), "Failed to free device memory for matrix A");
        d_A = nullptr;
        cuda_check(cudaFree(d_B), "Failed to free device memory for matrix B");
        d_B = nullptr;
        cuda_check(cudaFree(d_C), "Failed to free device memory for matrix C");
        d_C = nullptr;

        return h_C;
        
    }catch (...){
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);

        throw;
    }
}


std::vector<double> cuda_matrix_multiplication_wrapper(
    BenchmarkContext &ctx,
    int r1,
    int c1,
    int r2,
    int c2
) {
    std::vector<double> h_A = std::move(*build_matrix(r1, c1, MATRIX_MULTIPLY_A_SEED));
    std::vector<double> h_B = std::move(*build_matrix(r2, c2, MATRIX_MULTIPLY_B_SEED));

    ctx.iterations = 1;
    ctx.problem_size = r1 * c2;
    ctx.number_of_processors = get_number_of_processors(r1 * c2);

    ctx.start_measurement();

    std::vector<double> h_C = cuda_matrix_multiply(h_A, h_B, r1, c1, r2, c2);

    ctx.stop_measurement();

    ctx.bytes_moved =
        (static_cast<double>(r1) * c1 + static_cast<double>(c1) * c2 + static_cast<double>(r1) * c2) * sizeof(double);
    ctx.flop_count =
        2.0 * r1 * c2 * c1;

    return h_C;
}

BENCHMARK_TEST(matrix_multiplication_CUDA_n32x32, matrix_multiplication_n32x32, Paradigm::CUDA) {
    cuda_matrix_multiplication_wrapper(ctx, 32, 32, 32, 32);
}

BENCHMARK_TEST(matrix_multiplication_CUDA_n64x64, matrix_multiplication_n64x64, Paradigm::CUDA) {
    cuda_matrix_multiplication_wrapper(ctx, 64, 64, 64, 64);
}

BENCHMARK_TEST(matrix_multiplication_CUDA_n128x128, matrix_multiplication_n128x128, Paradigm::CUDA) {
    cuda_matrix_multiplication_wrapper(ctx, 128, 128, 128, 128);
}

BENCHMARK_TEST(matrix_multiplication_CUDA_n256x256, matrix_multiplication_n256x256, Paradigm::CUDA) {
    cuda_matrix_multiplication_wrapper(ctx, 256, 256, 256, 256);
}

BENCHMARK_TEST(matrix_multiplication_CUDA_n384x384, matrix_multiplication_n384x384, Paradigm::CUDA) {
    cuda_matrix_multiplication_wrapper(ctx, 384, 384, 384, 384);
}

BENCHMARK_TEST(matrix_multiplication_CUDA_n512x512, matrix_multiplication_n512x512, Paradigm::CUDA) {
    cuda_matrix_multiplication_wrapper(ctx, 512, 512, 512, 512);
}

BENCHMARK_TEST(matrix_multiplication_CUDA_n768x768, matrix_multiplication_n768x768, Paradigm::CUDA) {
    cuda_matrix_multiplication_wrapper(ctx, 768, 768, 768, 768);
}

BENCHMARK_TEST(matrix_multiplication_CUDA_n1024x1024, matrix_multiplication_n1024x1024, Paradigm::CUDA) {
    cuda_matrix_multiplication_wrapper(ctx, 1024, 1024, 1024, 1024);
}

BENCHMARK_TEST(matrix_multiplication_CUDA_n1536x1536, matrix_multiplication_n1536x1536, Paradigm::CUDA) {
    cuda_matrix_multiplication_wrapper(ctx, 1536, 1536, 1536, 1536);
}

BENCHMARK_TEST(matrix_multiplication_CUDA_n2048x2048, matrix_multiplication_n2048x2048, Paradigm::CUDA) {
    cuda_matrix_multiplication_wrapper(ctx, 2048, 2048, 2048, 2048);
}

BENCHMARK_TEST(matrix_multiplication_CUDA_rect_128x256_256x64, matrix_multiplication_rect_128x256_256x64, Paradigm::CUDA) {
    cuda_matrix_multiplication_wrapper(ctx, 128, 256, 256, 64);
}

BENCHMARK_TEST(matrix_multiplication_CUDA_rect_64x128_128x256, matrix_multiplication_rect_64x128_128x256, Paradigm::CUDA) {
    cuda_matrix_multiplication_wrapper(ctx, 64, 128, 128, 256);
}

BENCHMARK_TEST(matrix_multiplication_CUDA_rect_256x64_64x128, matrix_multiplication_rect_256x64_64x128, Paradigm::CUDA) {
    cuda_matrix_multiplication_wrapper(ctx, 256, 64, 64, 128);
}

BENCHMARK_TEST(matrix_multiplication_CUDA_rect_32x256_256x32, matrix_multiplication_rect_32x256_256x32, Paradigm::CUDA) {
    cuda_matrix_multiplication_wrapper(ctx, 32, 256, 256, 32);
}