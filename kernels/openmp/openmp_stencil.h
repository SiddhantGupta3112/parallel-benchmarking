#ifndef OPENMP_STENCIL_H
#define OPENMP_STENCIL_H

#include "benchmark_context.h"
#include <vector>

// OpenMP version of the serial stencil: same Jacobi update with fixed
// boundaries and ping-pong buffers, with the row loop of every iteration split
// across `num_threads` threads.
// Throws std::runtime_error for invalid dimensions, iterations or data size and
// std::invalid_argument if num_threads <= 0.
std::vector<double> openmp_stencil(
    const std::vector<double>& input,
    int rows,
    int cols,
    int iterations,
    int num_threads
);

#endif // OPENMP_STENCIL_H
