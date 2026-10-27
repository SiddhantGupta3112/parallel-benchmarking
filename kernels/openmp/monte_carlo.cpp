#include "openmp_monte_carlo.h"

#include <cstdint>
#include <random>
#include <stdexcept>

#include "openmp_helper.h"
#include "registrar.h"
#include "seed_values.h"

double openmp_monte_carlo(int N, int num_threads) {
    if (N <= 0) {
        throw std::invalid_argument("N must be a positive integer.");
    }

    OpenMPThreadScope thread_scope(num_threads);

    int count_inside = 0;

    // Each thread owns a generator (mt19937 is not thread-safe, so one must
    // never be shared) seeded with MONTE_CARLO_SEED + its thread number.
    // Thread 0 therefore uses exactly the serial seed. The N samples are split
    // into contiguous static chunks, and the per-thread hit counts are summed
    // with a reduction (each thread counts privately, then the partial sums are
    // added once at the end of the region).
    #pragma omp parallel reduction(+ : count_inside)
    {
        std::mt19937 gen(
            MONTE_CARLO_SEED + static_cast<std::uint32_t>(omp_get_thread_num())
        );
        std::uniform_real_distribution<double> generator(0.0, 1.0);

        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            double x = generator(gen);
            double y = generator(gen);

            if (x * x + y * y <= 1.0) {
                count_inside++;
            }
        }
    }

    return static_cast<double>(count_inside) / N * 4.0;
}

namespace {

double openmp_monte_carlo_wrapper(BenchmarkContext& ctx, int N, int num_threads) {
    if (N <= 0) {
        throw std::invalid_argument("N must be a positive integer.");
    }

    // Untimed: find out how many threads a parallel region really gets.
    const int threads = get_number_of_threads(num_threads);

    ctx.problem_size = N;
    ctx.iterations = 1;
    ctx.number_of_processors = threads;

    ctx.start_measurement();

    double pi = openmp_monte_carlo(N, num_threads);

    ctx.stop_measurement();

    ctx.flop_count = 3.0 * N;
    ctx.bytes_moved = 2.0 * N * sizeof(double);

    return pi;
}

} // namespace

// One registration per thread count (see OPENMP_THREAD_SWEEP in
// openmp_helper.h). kernel_name matches the serial benchmark exactly.
#define OPENMP_MONTE_CARLO_BENCHMARK(size_tag, N, T)                              \
    BENCHMARK_TEST(monte_carlo_OpenMP_##size_tag##_t##T,                          \
                   monte_carlo_##size_tag, Paradigm::OpenMP) {                    \
        openmp_monte_carlo_wrapper(ctx, N, T);                                    \
    }

OPENMP_THREAD_SWEEP(OPENMP_MONTE_CARLO_BENCHMARK, n10000, 10'000)
OPENMP_THREAD_SWEEP(OPENMP_MONTE_CARLO_BENCHMARK, n20000, 20'000)
OPENMP_THREAD_SWEEP(OPENMP_MONTE_CARLO_BENCHMARK, n50000, 50'000)
OPENMP_THREAD_SWEEP(OPENMP_MONTE_CARLO_BENCHMARK, n100000, 100'000)
OPENMP_THREAD_SWEEP(OPENMP_MONTE_CARLO_BENCHMARK, n200000, 200'000)
OPENMP_THREAD_SWEEP(OPENMP_MONTE_CARLO_BENCHMARK, n500000, 500'000)
OPENMP_THREAD_SWEEP(OPENMP_MONTE_CARLO_BENCHMARK, n1000000, 1'000'000)
OPENMP_THREAD_SWEEP(OPENMP_MONTE_CARLO_BENCHMARK, n2000000, 2'000'000)
OPENMP_THREAD_SWEEP(OPENMP_MONTE_CARLO_BENCHMARK, n5000000, 5'000'000)
OPENMP_THREAD_SWEEP(OPENMP_MONTE_CARLO_BENCHMARK, n10000000, 10'000'000)
OPENMP_THREAD_SWEEP(OPENMP_MONTE_CARLO_BENCHMARK, n20000000, 20'000'000)
OPENMP_THREAD_SWEEP(OPENMP_MONTE_CARLO_BENCHMARK, n50000000, 50'000'000)
OPENMP_THREAD_SWEEP(OPENMP_MONTE_CARLO_BENCHMARK, n100000000, 100'000'000)
