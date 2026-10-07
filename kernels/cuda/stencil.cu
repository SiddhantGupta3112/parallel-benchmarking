#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

#include "helper.h"
#include "registrar.h"
#include "stencil.h"

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
    BenchmarkContext& ctx,
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

    if (h_A.size() != static_cast<std::size_t>(r * c)) {
        throw std::runtime_error("Matrix dimensions do not match data");
    }

    ctx.iterations = iterations;
    ctx.problem_size = r * c;
    ctx.number_of_processors = get_number_of_processors(r * c);

    dim3 threads_per_block(16, 16); 
    dim3 blocks_per_grid(
        (c + threads_per_block.x - 1) / threads_per_block.x, 
        (r + threads_per_block.y - 1) / threads_per_block.y  
    );

    std::vector<double> h_B(r * c, 0.0);
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

    double *d_A, *d_B;
    size_t size_bytes = sizeof(double) * r * c;
    
    cudaMalloc((void**) &d_A, size_bytes);
    cudaMalloc((void**) &d_B, size_bytes);

    cudaMemcpy(d_A, h_A.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_bytes, cudaMemcpyHostToDevice);

    for(int i = 0; i < iterations; i++){
        stencil_kernel<<<blocks_per_grid, threads_per_block>>>(d_A, d_B, r, c);
        cudaDeviceSynchronize();
        
        double *temp = d_A;
        d_A = d_B;
        d_B = temp;
    }

    cudaMemcpy(h_B.data(), d_A, size_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);

    ctx.bytes_moved = 2.0 * r * c * sizeof(double) * iterations;
    ctx.flop_count = 4.0 * (r - 2) * (c - 2) * iterations;

    return h_B;
}
