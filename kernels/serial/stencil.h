#ifndef STENCIL_H
#define STENCIL_H

#include "benchmark_context.h"
#include <vector>

std::vector<double> stencil(
    const std::vector<double>& input,
    int rows,
    int cols,
    int iterations
);
#endif