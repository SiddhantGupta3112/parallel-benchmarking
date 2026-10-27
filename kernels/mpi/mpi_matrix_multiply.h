#ifndef MPI_MATRIX_MULTIPLY_H
#define MPI_MATRIX_MULTIPLY_H

#include <mpi.h>
#include <vector>
#include "benchmark_context.h"

std::vector<double> matrix_multiply_mpi(
    const std::vector<double>& A,
    const std::vector<double>& B,
    int r1,
    int c1,
    int r2,
    int c2,
    MPI_Comm comm = MPI_COMM_WORLD,
    double* communication_time_ms = nullptr
);

std::vector<double> matrix_multiplication_mpi_wrapper(
    BenchmarkContext& ctx,
    int r1,
    int c1,
    int r2,
    int c2
);

#endif
