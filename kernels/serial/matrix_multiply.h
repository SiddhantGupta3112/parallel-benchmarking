#ifndef MATRIX_MULTIPLY_H
#define MATRIX_MULTIPLY_H

#include <vector>
#include "benchmark_context.h"

std::vector<double> matrix_multiply(
    BenchmarkContext &ctx,
    const std::vector<double>& A,
    const std::vector<double>& B,
    int r1,
    int c1,
    int r2,
    int c2
);
#endif