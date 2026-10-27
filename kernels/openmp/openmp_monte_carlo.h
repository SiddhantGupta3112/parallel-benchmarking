#ifndef OPENMP_MONTE_CARLO_H
#define OPENMP_MONTE_CARLO_H

#include "benchmark_context.h"

// OpenMP version of the serial monte_carlo: same estimator, with the N samples
// split across `num_threads` threads. Thread k uses its own std::mt19937
// seeded with MONTE_CARLO_SEED + k, so results are reproducible for a given
// (N, thread count); with one thread the result equals the serial one exactly.
// Throws std::invalid_argument if N <= 0 or num_threads <= 0.
double openmp_monte_carlo(int N, int num_threads);

#endif // OPENMP_MONTE_CARLO_H
