#include "mpi_stencil.h"
#include "build_stencil_input.h"
#include "seed_values.h"
#include "registrar.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include "mpi_rank_counts.h"

std::vector<double> stencil_mpi(
    const std::vector<double>& input,
    int rows,
    int cols,
    int iterations,
    MPI_Comm comm
) {
    if (rows < 3 || cols < 3) {
        throw std::runtime_error("Stencil dimensions must be at least 3x3");
    }

    if (iterations < 0) {
        throw std::runtime_error("Iterations cannot be negative");
    }

        int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int valid_input = 1;

    if (rank == 0) {
        valid_input =
            input.size() == static_cast<std::size_t>(rows) * cols;
    }

    MPI_Bcast(&valid_input, 1, MPI_INT, 0, comm);

    if (!valid_input) {
        throw std::runtime_error(
            "Root matrix dimensions do not match input data"
        );
    }

    std::vector<int> row_counts(size);
    std::vector<int> row_displacements(size);

    int base_rows = rows / size;
    int remainder = rows % size;
    int displacement = 0;

    for (int r = 0; r < size; ++r) {
        row_counts[r] = base_rows + (r < remainder ? 1 : 0);
        row_displacements[r] = displacement;
        displacement += row_counts[r];
    }

    int local_rows = row_counts[rank];
    int local_start = row_displacements[rank];

    std::vector<int> active_ranks;
    for (int r = 0; r < size; ++r) {
        if (row_counts[r] > 0) {
            active_ranks.push_back(r);
        }
    }

    int active_index = -1;
    for (int i = 0; i < static_cast<int>(active_ranks.size()); ++i) {
        if (active_ranks[i] == rank) {
            active_index = i;
            break;
        }
    }

    int previous_rank = MPI_PROC_NULL;
    int next_rank = MPI_PROC_NULL;

    if (active_index > 0) {
        previous_rank = active_ranks[active_index - 1];
    }

    if (active_index >= 0 &&
        active_index + 1 < static_cast<int>(active_ranks.size())) {
        next_rank = active_ranks[active_index + 1];
    }

    std::vector<double> local_current(
        static_cast<std::size_t>(local_rows + 2) * cols, 0.0
    );
    std::vector<double> local_next(
        static_cast<std::size_t>(local_rows + 2) * cols, 0.0
    );

    std::vector<int> scatter_counts(size);
    std::vector<int> scatter_displacements(size);

    for (int r = 0; r < size; ++r) {
        scatter_counts[r] = row_counts[r] * cols;
        scatter_displacements[r] = row_displacements[r] * cols;
    }

    double dummy = 0.0;

    MPI_Scatterv(
        rank == 0 ? input.data() : &dummy,
        scatter_counts.data(),
        scatter_displacements.data(),
        MPI_DOUBLE,
        local_rows > 0 ? local_current.data() + cols : &dummy,
        local_rows * cols,
        MPI_DOUBLE,
        0,
        comm
    );

    for (int i = 0; i < local_rows; ++i) {
        int global_row = local_start + i;

        if (global_row == 0 || global_row == rows - 1) {
            std::copy_n(
                local_current.data() + (i + 1) * cols,
                cols,
                local_next.data() + (i + 1) * cols
            );
        }
    }

    for (int i = 0; i < local_rows; ++i) {
        local_next[(i + 1) * cols] = local_current[(i + 1) * cols];
        local_next[(i + 1) * cols + cols - 1] =
            local_current[(i + 1) * cols + cols - 1];
    }

    for (int iter = 0; iter < iterations; ++iter) {
        if (local_rows > 0) {
            MPI_Sendrecv(
                local_current.data() + cols,
                cols,
                MPI_DOUBLE,
                previous_rank,
                100,
                local_current.data() + (local_rows + 1) * cols,
                cols,
                MPI_DOUBLE,
                next_rank,
                100,
                comm,
                MPI_STATUS_IGNORE
            );

            MPI_Sendrecv(
                local_current.data() + local_rows * cols,
                cols,
                MPI_DOUBLE,
                next_rank,
                200,
                local_current.data(),
                cols,
                MPI_DOUBLE,
                previous_rank,
                200,
                comm,
                MPI_STATUS_IGNORE
            );

            for (int i = 1; i <= local_rows; ++i) {
                int global_row = local_start + i - 1;

                if (global_row == 0 || global_row == rows - 1) {
                    std::copy_n(
                        local_current.data() + i * cols,
                        cols,
                        local_next.data() + i * cols
                    );
                    continue;
                }

                local_next[i * cols] = local_current[i * cols];
                local_next[i * cols + cols - 1] =
                    local_current[i * cols + cols - 1];

                for (int j = 1; j < cols - 1; ++j) {
                    local_next[i * cols + j] = (
                        local_current[(i - 1) * cols + j] +
                        local_current[(i + 1) * cols + j] +
                        local_current[i * cols + j - 1] +
                        local_current[i * cols + j + 1]
                    ) / 4.0;
                }
            }

            local_current.swap(local_next);
        }
    }

    std::vector<double> result;
    if (rank == 0) {
        result.resize(static_cast<std::size_t>(rows) * cols);
    }

    MPI_Gatherv(
        local_rows > 0 ? local_current.data() + cols : &dummy,
        local_rows * cols,
        MPI_DOUBLE,
        rank == 0 ? result.data() : &dummy,
        scatter_counts.data(),
        scatter_displacements.data(),
        MPI_DOUBLE,
        0,
        comm
    );

    return result;
}

std::vector<double> stencil_mpi_wrapper(
    BenchmarkContext& ctx,
    int rows,
    int cols,
    int iterations
) {
    if (rows < 3 || cols < 3) {
        throw std::runtime_error("Stencil dimensions must be at least 3x3");
    }

    if (iterations < 0) {
        throw std::runtime_error("Iterations cannot be negative");
    }

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<double> input;
    int inputs_ready = 1;

    if (rank == 0) {
        try {
            auto generated = build_stencil_input(
                rows, cols, STENCIL_SEED
            );

            if (!generated) {
                throw std::runtime_error("Stencil input generation failed");
            }

            input = std::move(*generated);
        } catch (...) {
            inputs_ready = 0;
        }
    }

    MPI_Bcast(
        &inputs_ready, 1, MPI_INT, 0, MPI_COMM_WORLD
    );

    if (!inputs_ready) {
        throw std::runtime_error(
            "Rank 0 failed to generate stencil input"
        );
    }

    ctx.start_measurement();

    std::vector<double> result = stencil_mpi(
        input, rows, cols, iterations, MPI_COMM_WORLD
    );
    ctx.stop_measurement();

    ctx.iterations = iterations;
    ctx.problem_size = rows * cols;
    ctx.number_of_processors = size;
    ctx.bytes_moved =
        2.0 * rows * cols * sizeof(double) * iterations;
    ctx.flop_count =
        4.0 * (rows - 2) * (cols - 2) * iterations;

    return result;
}

namespace{
struct StencilConfig {
    int rows;
    int cols;
    int iterations;
};


void register_stencil_benchmarks() {
    const StencilConfig configs[] = {
        {32, 32, 10},
        {64, 64, 10},
        {128, 128, 10},
        {256, 256, 10},
        {512, 512, 10},
        {1024, 1024, 10},
        {2048, 2048, 10},
        {4096, 4096, 10},
        {256, 256, 1},
        {256, 256, 50},
        {256, 256, 100},
        {1024, 1024, 1},
        {1024, 1024, 50},
        {1024, 1024, 100}
    };


    for (const auto& config : configs) {
        const std::string kernel_name =
            "stencil_n" + std::to_string(config.rows) + "x" +
            std::to_string(config.cols) + "_iter" +
            std::to_string(config.iterations);

        for (int ranks : MPI_RANK_COUNTS) {
            const std::string benchmark_name =
                "stencil_mpi_n" + std::to_string(config.rows) + "x" +
                std::to_string(config.cols) + "_iter" +
                std::to_string(config.iterations) + "_np" +
                std::to_string(ranks);

            get_registry().register_benchmark(
                benchmark_name,
                kernel_name,
                Paradigm::MPI,
                [config](BenchmarkContext& ctx) {
                    stencil_mpi_wrapper(
                        ctx, config.rows, config.cols, config.iterations
                    );
                }
            );
        }
    }
}

const bool stencil_benchmarks_registered = []() {
    register_stencil_benchmarks();
    return true;
}();
}
