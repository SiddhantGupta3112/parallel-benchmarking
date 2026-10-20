#ifndef MPI_MONTE_CARLO_H
#define MPI_MONTE_CARLO_H

#include <mpi.h>
#include "benchmark_context.h"

double monte_carlo_mpi(
    int N,
    MPI_Comm comm = MPI_COMM_WORLD
);

double monte_carlo_mpi_wrapper(
    BenchmarkContext& ctx,
    int N
);

#endif
