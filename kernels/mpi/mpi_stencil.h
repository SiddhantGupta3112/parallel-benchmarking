#ifndef MPI_STENCIL_H
#define MPI_STENCIL_H

#include <mpi.h>
#include <vector>
#include "benchmark_context.h"

std::vector<double> stencil_mpi(
    const std::vector<double>& input,
    int rows,
    int cols,
    int iterations,
    MPI_Comm comm = MPI_COMM_WORLD
);

std::vector<double> stencil_mpi_wrapper(
    BenchmarkContext& ctx,
    int rows,
    int cols,
    int iterations
);

#endif
