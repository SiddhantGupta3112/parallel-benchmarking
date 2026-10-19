
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdexcept>

#include "cuda_helper.h"
#include "registrar.h"
#include "cuda_monte_carlo.h"

#define THREADS_PER_BLOCK 256

__global__ void monte_carlo_kernel(
    unsigned long long seed,
    int N,
    int* count_inside_circle
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        curandStatePhilox4_32_10_t local_state;
        curand_init(seed, tid, 0, &local_state);

        float x = curand_uniform(&local_state);
        float y = curand_uniform(&local_state);

        if (x * x + y * y <= 1.0f) {
            atomicAdd(count_inside_circle, 1);
        }
    }
}

// Core computation: performs CUDA Monte Carlo without BenchmarkContext.
double monte_carlo_cuda(int N) {
    if (N <= 0) {
        throw std::invalid_argument("N must be a positive integer.");
    }

    int blocks_per_grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    int h_count_inside_circle = 0;
    int* d_count_inside_circle = nullptr;

    try {
        cuda_check(
            cudaMalloc(
                reinterpret_cast<void**>(&d_count_inside_circle),
                sizeof(int)
            ),
            "Failed to allocate device memory for count inside circle"
        );

        cuda_check(
            cudaMemset(d_count_inside_circle, 0, sizeof(int)),
            "Failed to initialize device memory for count inside circle"
        );

        constexpr unsigned long long seed = 12345ULL;

        monte_carlo_kernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(
            seed,
            N,
            d_count_inside_circle
        );

        cuda_check(
            cudaGetLastError(),
            "Monte Carlo kernel launch failed"
        );

        cuda_check(
            cudaDeviceSynchronize(),
            "Failed to synchronize device"
        );

        cuda_check(
            cudaMemcpy(
                &h_count_inside_circle,
                d_count_inside_circle,
                sizeof(int),
                cudaMemcpyDeviceToHost
            ),
            "Failed to copy count inside circle from device"
        );

        cuda_check(
            cudaFree(d_count_inside_circle),
            "Failed to free device memory for count inside circle"
        );

        d_count_inside_circle = nullptr;

        return static_cast<double>(h_count_inside_circle) / N * 4.0;
    }
    catch (...) {
        if (d_count_inside_circle != nullptr) {
            cudaFree(d_count_inside_circle);
        }

        throw;
    }
}

// Benchmark wrapper: manages measurement and metadata.
double cuda_monte_carlo_wrapper(BenchmarkContext& ctx, int N) {
    if (N <= 0) {
        throw std::invalid_argument("N must be a positive integer.");
    }

    ctx.problem_size = N;
    ctx.iterations = 1;
    ctx.number_of_processors = get_number_of_processors(N);

    ctx.start_measurement();

    double pi;

    try {
        pi = monte_carlo_cuda(N);
    }
    catch (...) {
        ctx.stop_measurement();
        throw;
    }

    ctx.stop_measurement();

    ctx.flop_count = 3.0 * N;
    ctx.bytes_moved = 2.0 * N * sizeof(float);

    return pi;
}

BENCHMARK_TEST(monte_carlo_CUDA_n10000, monte_carlo_n10000, Paradigm::CUDA) {
    cuda_monte_carlo_wrapper(ctx, 10'000);
}

BENCHMARK_TEST(monte_carlo_CUDA_n20000, monte_carlo_n20000, Paradigm::CUDA) {
    cuda_monte_carlo_wrapper(ctx, 20'000);
}

BENCHMARK_TEST(monte_carlo_CUDA_n50000, monte_carlo_n50000, Paradigm::CUDA) {
    cuda_monte_carlo_wrapper(ctx, 50'000);
}

BENCHMARK_TEST(monte_carlo_CUDA_n100000, monte_carlo_n100000, Paradigm::CUDA) {
    cuda_monte_carlo_wrapper(ctx, 100'000);
}

BENCHMARK_TEST(monte_carlo_CUDA_n200000, monte_carlo_n200000, Paradigm::CUDA) {
    cuda_monte_carlo_wrapper(ctx, 200'000);
}

BENCHMARK_TEST(monte_carlo_CUDA_n500000, monte_carlo_n500000, Paradigm::CUDA) {
    cuda_monte_carlo_wrapper(ctx, 500'000);
}

BENCHMARK_TEST(monte_carlo_CUDA_n1000000, monte_carlo_n1000000, Paradigm::CUDA) {
    cuda_monte_carlo_wrapper(ctx, 1'000'000);
}

BENCHMARK_TEST(monte_carlo_CUDA_n2000000, monte_carlo_n2000000, Paradigm::CUDA) {
    cuda_monte_carlo_wrapper(ctx, 2'000'000);
}

BENCHMARK_TEST(monte_carlo_CUDA_n5000000, monte_carlo_n5000000, Paradigm::CUDA) {
    cuda_monte_carlo_wrapper(ctx, 5'000'000);
}

BENCHMARK_TEST(monte_carlo_CUDA_n10000000, monte_carlo_n10000000, Paradigm::CUDA) {
    cuda_monte_carlo_wrapper(ctx, 10'000'000);
}

BENCHMARK_TEST(monte_carlo_CUDA_n20000000, monte_carlo_n20000000, Paradigm::CUDA) {
    cuda_monte_carlo_wrapper(ctx, 20'000'000);
}

BENCHMARK_TEST(monte_carlo_CUDA_n50000000, monte_carlo_n50000000, Paradigm::CUDA) {
    cuda_monte_carlo_wrapper(ctx, 50'000'000);
}

BENCHMARK_TEST(monte_carlo_CUDA_n100000000, monte_carlo_n100000000, Paradigm::CUDA) {
    cuda_monte_carlo_wrapper(ctx, 100'000'000);
}
