#include <cuda_runtime.h>
#include "registrar.h"
#include "cuda_matrix_multiply.h"
#include "helper.h"

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
    BenchmarkContext &ctx,
    const std::vector<double>& h_A,
    const std::vector<double>& h_B,
    int r1,
    int c1,
    int r2,
    int c2
) {
    if (r1 <= 0 || c1 <= 0 || r2 <= 0 || c2 <= 0) {
        throw std::runtime_error("Empty matrix");
    }

    if (c1 != r2) {
        throw std::runtime_error("Invalid matrix sizes");
    }

    if (h_A.size() != static_cast<std::size_t>(r1 * c1) ||
        h_B.size() != static_cast<std::size_t>(r2 * c2)) {
        throw std::runtime_error("Matrix dimensions do not match data");
    }

    ctx.iterations = 1;
    ctx.problem_size = r1 * c2;
    ctx.number_of_processors = get_number_of_processors(r1 * c2);

    dim3 threads_per_block(16, 16); 
    dim3 blocks_per_grid(
        (c2 + threads_per_block.x - 1) / threads_per_block.x, 
        (r1 + threads_per_block.y - 1) / threads_per_block.y  
    );

    std::vector<double> h_C(r1 * c2, 0.0);

    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)& d_A, sizeof(double) * r1 * c1);
    cudaMalloc((void**)& d_B, sizeof(double) * r2 * c2);
    cudaMalloc((void**)& d_C, sizeof(double) * r1 * c2);

    cudaMemcpy(d_A, h_A.data(), sizeof(double) * r1 * c1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeof(double) * r2 * c2, cudaMemcpyHostToDevice);

    matrix_muliply_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, d_C, r1, c1, r2, c2);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), d_C, sizeof(double) * r1 * c2, cudaMemcpyDeviceToHost);

    ctx.bytes_moved =
        (static_cast<double>(r1) * c1 + static_cast<double>(c1) * c2 + static_cast<double>(r1) * c2) * sizeof(double);
    ctx.flop_count =
        2.0 * r1 * c2 * c1;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return h_C;
}