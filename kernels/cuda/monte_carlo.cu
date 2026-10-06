#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "helper.h"
#include "registrar.h"
#include "cuda_monte_carlo.h"
# define THREADS_PER_BLOCK 256

__global__ void monte_carlo_kernel(unsigned long long seed, int N, int *count_inside_circle){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < N){
        curandStatePhilox4_32_10_t local_state;
        curand_init(seed, tid, 0, &local_state);

        float x = curand_uniform(&local_state);
        float y = curand_uniform(&local_state);

        if(x*x + y*y <= 1){
            atomicAdd(count_inside_circle, 1);
        }
    }
}


double monte_carlo_cuda(BenchmarkContext &ctx, int N){
    ctx.problem_size = N;
    ctx.iterations = 1;
    ctx.number_of_processors = get_number_of_processors(N);

    int blocks_per_grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    int h_count_inside_circle = 0;
    int *d_count_inside_circle;

    cudaMalloc((void**)&d_count_inside_circle, sizeof(int));
    cudaMemset(d_count_inside_circle, 0, sizeof(int));

    unsigned long long seed = 12345ULL;
    monte_carlo_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(seed, N, d_count_inside_circle);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_count_inside_circle, d_count_inside_circle, sizeof(int), cudaMemcpyDeviceToHost);

    double pi = static_cast<double>(h_count_inside_circle) / N * 4.0;  

    ctx.flop_count = 3 * N;
    ctx.bytes_moved = 2.0 * N * sizeof(float);

    cudaFree(d_count_inside_circle);
    return pi;
}

BENCHMARK_TEST(monte_carlo_cuda_n10000, monte_carlo_n10000, Paradigm::CUDA) {
    monte_carlo_cuda(ctx, 10'000);
}

BENCHMARK_TEST(monte_carlo_cuda_n100000, monte_carlo_n100000, Paradigm::CUDA) {
    monte_carlo_cuda(ctx, 100'000);
}

BENCHMARK_TEST(monte_carlo_cuda_n1000000, monte_carlo_n1000000, Paradigm::CUDA) {
    monte_carlo_cuda(ctx, 1'000'000);
}

BENCHMARK_TEST(monte_carlo_cuda_n10000000, monte_carlo_n10000000, Paradigm::CUDA) {
    monte_carlo_cuda(ctx, 10'000'000);
}
