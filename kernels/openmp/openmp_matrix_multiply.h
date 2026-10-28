#ifndef OPENMP_MATRIX_MULTIPLY_H
#define OPENMP_MATRIX_MULTIPLY_H

#include "benchmark_context.h"
#include <vector>

// OpenMP version of the serial matrix_multiply: same algorithm and i-k-j loop
// order, with the outer i loop split across `num_threads` threads.
// Throws std::runtime_error for invalid dimensions or data sizes and
// std::invalid_argument if num_threads <= 0.
std::vector<double> openmp_matrix_multiply(
    const std::vector<double>& A,
    const std::vector<double>& B,
    int r1,
    int c1,
    int r2,
    int c2,
    int num_threads
);

#endif // OPENMP_MATRIX_MULTIPLY_H
