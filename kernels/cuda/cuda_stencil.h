#ifndef CUDA_STENCIL_H
#define CUDA_STENCIL_H

#include "benchmark_context.h"
#include <vector>

std::vector<double> cuda_stencil(
    const std::vector<double>& input,
    int rows,
    int cols,
    int iterations
);
#endif