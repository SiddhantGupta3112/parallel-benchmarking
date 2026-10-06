#ifndef CUDA_MATRIX_MULTIPLY_H
#define CUDA_MATRIX_MULTIPLY_H

#include "benchmark_context.h"
#include <vector>

std::vector<double> cuda_matrix_multiply(
    BenchmarkContext &ctx,
    const std::vector<double>& A,
    const std::vector<double>& B,
    int r1,
    int c1,
    int r2,
    int c2
);

#endif // CUDA_MATRIX_MULTIPLY_H