#include "mpi_monte_carlo.h"

#include "seed_values.h"

#include <mpi.h>
#include "registrar.h"

#include <cstdint>
#include <random>
#include <stdexcept>

double monte_carlo_mpi(int N, MPI_Comm comm) {
    if (N <= 0) {
        throw std::invalid_argument(
            "N must be a positive integer."
        );
    }

    int rank;
    int size;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Divide N samples as evenly as possible among all ranks.
    const int local_N =
        N / size + (rank < N % size ? 1 : 0);

    // Use a different, deterministic random stream for each rank.
    std::mt19937 gen(
        MONTE_CARLO_SEED + static_cast<std::uint32_t>(rank)
    );

    std::uniform_real_distribution<double> generator(
        0.0, 1.0
    );

    long long local_count_inside = 0;

    // Each rank processes its own samples.
    for (int i = 0; i < local_N; ++i) {
        const double x = generator(gen);
        const double y = generator(gen);

        if (x * x + y * y <= 1.0) {
            ++local_count_inside;
        }
    }

    // Combine all local counts on rank 0.
    long long total_count_inside = 0;

    MPI_Reduce(
        &local_count_inside,
        &total_count_inside,
        1,
        MPI_LONG_LONG_INT,
        MPI_SUM,
        0,
        comm
    );

    // Calculate the estimate on rank 0.
    double pi_estimate = 0.0;

    if (rank == 0) {
        pi_estimate =
            4.0 * static_cast<double>(total_count_inside) / N;
    }

    // Make the estimate available on every rank.
    MPI_Bcast(
        &pi_estimate,
        1,
        MPI_DOUBLE,
        0,
        comm
    );

    return pi_estimate;
}

double monte_carlo_mpi_wrapper(
    BenchmarkContext& ctx,
    int N
) {
    if (N <= 0) {
        throw std::invalid_argument(
            "N must be a positive integer."
        );
    }

    int size;

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set the workload metadata.
    ctx.problem_size = N;
    ctx.iterations = 1;
    ctx.number_of_processors = size;

    ctx.start_measurement();

    const double pi = monte_carlo_mpi(
        N,
        MPI_COMM_WORLD
    );

    ctx.stop_measurement();

    // Preserve the metric formulas used by Serial.
    ctx.flop_count = 3.0 * N;
    ctx.bytes_moved =
        2.0 * static_cast<double>(N) * sizeof(double);

    return pi;
}
namespace {

struct MonteCarloConfig {
    const char* suffix;
    int samples;
};

constexpr MonteCarloConfig monte_carlo_configs[] = {
    {"n10000", 10'000},
    {"n20000", 20'000},
    {"n50000", 50'000},
    {"n100000", 100'000},
    {"n200000", 200'000},
    {"n500000", 500'000},
    {"n1000000", 1'000'000},
    {"n2000000", 2'000'000},
    {"n5000000", 5'000'000},
    {"n10000000", 10'000'000},
    {"n20000000", 20'000'000},
    {"n50000000", 50'000'000},
    {"n100000000", 100'000'000}
};

constexpr int mpi_rank_counts[] = {
    1, 2, 4, 8, 12, 16, 20
};

struct RegisterMonteCarloMpiBenchmarks {
    RegisterMonteCarloMpiBenchmarks() {
        for (const auto& config : monte_carlo_configs) {
            const std::string kernel_name =
                std::string("monte_carlo_") + config.suffix;

            for (int ranks : mpi_rank_counts) {
                const std::string benchmark_name =
                    std::string("monte_carlo_mpi_") +
                    config.suffix +
                    "_np" +
                    std::to_string(ranks);

                get_registry().register_benchmark(
                    benchmark_name,
                    kernel_name,
                    Paradigm::MPI,
                    [config](BenchmarkContext& ctx) {
                        monte_carlo_mpi_wrapper(
                            ctx,
                            config.samples
                        );
                    }
                );
            }
        }
    }
};

RegisterMonteCarloMpiBenchmarks
    register_monte_carlo_mpi_benchmarks;

}