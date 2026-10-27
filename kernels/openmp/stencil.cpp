#include "openmp_stencil.h"

#include <stdexcept>

#include "build_stencil_input.h"
#include "openmp_helper.h"
#include "registrar.h"
#include "seed_values.h"

std::vector<double> openmp_stencil(
    const std::vector<double>& input,
    int rows,
    int cols,
    int iterations,
    int num_threads
) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Empty matrix");
    }

    if (iterations < 0) {
        throw std::runtime_error("Iterations cannot be negative");
    }

    if (input.size() != static_cast<std::size_t>(rows) * cols) {
        throw std::runtime_error(
            "Matrix dimensions do not match data"
        );
    }

    std::vector<double> current = input;
    std::vector<double> next(static_cast<std::size_t>(rows) * cols, 0.0);

    for (int j = 0; j < cols; ++j) {
        next[j] = current[j];

        next[(rows - 1) * cols + j] =
            current[(rows - 1) * cols + j];
    }

    for (int i = 0; i < rows; ++i) {
        next[i * cols] =
            current[i * cols];

        next[i * cols + (cols - 1)] =
            current[i * cols + (cols - 1)];
    }

    OpenMPThreadScope thread_scope(num_threads);

    for (int iter = 0; iter < iterations; ++iter) {

        // The row loop of one Jacobi sweep is parallel. `next` is written at
        // (i, j) by exactly one thread and `current` is only read, so there are
        // no races within a sweep. The implicit barrier at the end of the
        // parallel-for is what separates one sweep from the next: the swap
        // below runs on the main thread after every row has been written.
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < rows - 1; ++i) {
            for (int j = 1; j < cols - 1; ++j) {

                next[i * cols + j] =
                    (
                        current[(i - 1) * cols + j] +
                        current[(i + 1) * cols + j] +
                        current[i * cols + (j - 1)] +
                        current[i * cols + (j + 1)]
                    ) / 4.0;
            }
        }

        current.swap(next);
    }

    return current;
}

namespace {

std::vector<double> openmp_stencil_wrapper(
    BenchmarkContext& ctx,
    int rows,
    int cols,
    int iterations,
    int num_threads
) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Empty matrix");
    }

    if (iterations < 0) {
        throw std::runtime_error("Iterations cannot be negative");
    }

    std::vector<double> input = std::move(*build_stencil_input(rows, cols, STENCIL_SEED));

    // Untimed: find out how many threads a parallel region really gets.
    const int threads = get_number_of_threads(num_threads);

    ctx.start_measurement();

    std::vector<double> result = openmp_stencil(input, rows, cols, iterations, num_threads);

    ctx.stop_measurement();

    ctx.iterations = iterations;
    ctx.problem_size = rows * cols;
    ctx.number_of_processors = threads;

    ctx.bytes_moved =
        2.0 * rows * cols * sizeof(double) * iterations;

    ctx.flop_count =
        4.0 * (rows - 2) * (cols - 2) * iterations;

    return result;
}

} // namespace

// One registration per thread count (see OPENMP_THREAD_SWEEP in
// openmp_helper.h). kernel_name matches the serial benchmark exactly.
#define OPENMP_STENCIL_BENCHMARK(size_tag, rows, cols, iterations, T)             \
    BENCHMARK_TEST(stencil_OpenMP_##size_tag##_t##T,                              \
                   stencil_##size_tag, Paradigm::OpenMP) {                        \
        openmp_stencil_wrapper(ctx, rows, cols, iterations, T);                   \
    }

OPENMP_THREAD_SWEEP(OPENMP_STENCIL_BENCHMARK, n32x32_iter10, 32, 32, 10)
OPENMP_THREAD_SWEEP(OPENMP_STENCIL_BENCHMARK, n64x64_iter10, 64, 64, 10)
OPENMP_THREAD_SWEEP(OPENMP_STENCIL_BENCHMARK, n128x128_iter10, 128, 128, 10)
OPENMP_THREAD_SWEEP(OPENMP_STENCIL_BENCHMARK, n256x256_iter10, 256, 256, 10)
OPENMP_THREAD_SWEEP(OPENMP_STENCIL_BENCHMARK, n512x512_iter10, 512, 512, 10)
OPENMP_THREAD_SWEEP(OPENMP_STENCIL_BENCHMARK, n1024x1024_iter10, 1024, 1024, 10)
OPENMP_THREAD_SWEEP(OPENMP_STENCIL_BENCHMARK, n2048x2048_iter10, 2048, 2048, 10)
OPENMP_THREAD_SWEEP(OPENMP_STENCIL_BENCHMARK, n4096x4096_iter10, 4096, 4096, 10)

OPENMP_THREAD_SWEEP(OPENMP_STENCIL_BENCHMARK, n256x256_iter1, 256, 256, 1)
OPENMP_THREAD_SWEEP(OPENMP_STENCIL_BENCHMARK, n256x256_iter50, 256, 256, 50)
OPENMP_THREAD_SWEEP(OPENMP_STENCIL_BENCHMARK, n256x256_iter100, 256, 256, 100)
OPENMP_THREAD_SWEEP(OPENMP_STENCIL_BENCHMARK, n1024x1024_iter1, 1024, 1024, 1)
OPENMP_THREAD_SWEEP(OPENMP_STENCIL_BENCHMARK, n1024x1024_iter50, 1024, 1024, 50)
OPENMP_THREAD_SWEEP(OPENMP_STENCIL_BENCHMARK, n1024x1024_iter100, 1024, 1024, 100)
