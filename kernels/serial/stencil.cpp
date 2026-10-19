#include "stencil.h"
#include "build_stencil_input.h"
#include "seed_values.h"
#include "registrar.h"

#include <stdexcept>
#include <algorithm>
#include <cmath>

std::vector<double> stencil(
    const std::vector<double>& input,
    int rows,
    int cols,
    int iterations
){
    if (input.size() != static_cast<std::size_t>(rows * cols)) {
        throw std::runtime_error(
            "Matrix dimensions do not match data"
        );
    }

    std::vector<double> current = input;
    std::vector<double> next(rows * cols, 0.0);

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

    for (int iter = 0; iter < iterations; ++iter) {

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

std::vector<double> stencil_wrapper(
    BenchmarkContext& ctx,
    int rows,
    int cols,
    int iterations
) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Empty matrix");
    }

    if (iterations < 0) {
        throw std::runtime_error("Iterations cannot be negative");
    }

    std::vector<double> input = std::move(*build_stencil_input(rows, cols, STENCIL_SEED));

    ctx.start_measurement();

    std::vector<double> result = stencil(input, rows, cols, iterations);

    ctx.stop_measurement();

    ctx.iterations = iterations;
    ctx.problem_size = rows * cols;
    ctx.number_of_processors = 1;

    ctx.bytes_moved =
        2.0 * rows * cols * sizeof(double) * iterations;

    ctx.flop_count =
        4.0 * (rows - 2) * (cols - 2) * iterations;

    return result;
}

BENCHMARK_TEST(stencil_serial_n32x32_iter10, stencil_n32x32_iter10, Paradigm::Serial) {
    stencil_wrapper(ctx, 32, 32, 10);
}

BENCHMARK_TEST(stencil_serial_n64x64_iter10, stencil_n64x64_iter10, Paradigm::Serial) {
    stencil_wrapper(ctx, 64, 64, 10);
}

BENCHMARK_TEST(stencil_serial_n128x128_iter10, stencil_n128x128_iter10, Paradigm::Serial) {
    stencil_wrapper(ctx, 128, 128, 10);
}

BENCHMARK_TEST(stencil_serial_n256x256_iter10, stencil_n256x256_iter10, Paradigm::Serial) {
    stencil_wrapper(ctx, 256, 256, 10);
}

BENCHMARK_TEST(stencil_serial_n512x512_iter10, stencil_n512x512_iter10, Paradigm::Serial) {
    stencil_wrapper(ctx, 512, 512, 10);
}

BENCHMARK_TEST(stencil_serial_n1024x1024_iter10, stencil_n1024x1024_iter10, Paradigm::Serial) {
    stencil_wrapper(ctx, 1024, 1024, 10);
}

BENCHMARK_TEST(stencil_serial_n2048x2048_iter10, stencil_n2048x2048_iter10, Paradigm::Serial) {
    stencil_wrapper(ctx, 2048, 2048, 10);
}

BENCHMARK_TEST(stencil_serial_n4096x4096_iter10, stencil_n4096x4096_iter10, Paradigm::Serial) {
    stencil_wrapper(ctx, 4096, 4096, 10);
}

BENCHMARK_TEST(stencil_serial_n256x256_iter1, stencil_n256x256_iter1, Paradigm::Serial) {
    stencil_wrapper(ctx, 256, 256, 1);
}

BENCHMARK_TEST(stencil_serial_n256x256_iter50, stencil_n256x256_iter50, Paradigm::Serial) {
    stencil_wrapper(ctx, 256, 256, 50);
}

BENCHMARK_TEST(stencil_serial_n256x256_iter100, stencil_n256x256_iter100, Paradigm::Serial) {
    stencil_wrapper(ctx, 256, 256, 100);
}

BENCHMARK_TEST(stencil_serial_n1024x1024_iter1, stencil_n1024x1024_iter1, Paradigm::Serial) {
    stencil_wrapper(ctx, 1024, 1024, 1);
}

BENCHMARK_TEST(stencil_serial_n1024x1024_iter50, stencil_n1024x1024_iter50, Paradigm::Serial) {
    stencil_wrapper(ctx, 1024, 1024, 50);
}

BENCHMARK_TEST(stencil_serial_n1024x1024_iter100, stencil_n1024x1024_iter100, Paradigm::Serial) {
    stencil_wrapper(ctx, 1024, 1024, 100);
}