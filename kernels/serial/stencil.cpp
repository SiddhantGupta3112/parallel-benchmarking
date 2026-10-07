#include "stencil.h"

#include <stdexcept>
#include <algorithm>
#include <cmath>

std::vector<double> stencil(
    BenchmarkContext& ctx,
    const std::vector<double>& input,
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

    ctx.iterations = iterations;
    ctx.problem_size = rows * cols;
    ctx.number_of_processors = 1;

    ctx.bytes_moved =
        2.0 * rows * cols * sizeof(double) * iterations;

    ctx.flop_count =
        4.0 * (rows - 2) * (cols - 2) * iterations;

    return current;
}

