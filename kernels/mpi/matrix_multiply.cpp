#include "registrar.h"
#include "mpi_matrix_multiply.h"
#include "build_matrix.h"
#include "seed_values.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

std::vector<double> matrix_multiply_mpi(
    const std::vector<double>& A,
    const std::vector<double>& B,
    int r1,
    int c1,
    int r2,
    int c2,
    MPI_Comm comm
) {
    if (r1 <= 0 || c1 <= 0 || r2 <= 0 || c2 <= 0) {
        throw std::runtime_error("Empty matrix");
    }

    if (c1 != r2) {
        throw std::runtime_error("Invalid matrix sizes");
    }

    const std::size_t a_elements =
        static_cast<std::size_t>(r1) * c1;

    const std::size_t b_elements =
        static_cast<std::size_t>(r2) * c2;

    const std::size_t c_elements =
        static_cast<std::size_t>(r1) * c2;

    if (a_elements > INT_MAX ||
        b_elements > INT_MAX ||
        c_elements > INT_MAX) {
        throw std::runtime_error(
            "Matrix is too large for MPI int counts"
        );
    }

    int rank;
    int size;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int valid_inputs = 1;

    if (rank == 0) {
        valid_inputs =
            A.size() == a_elements &&
            B.size() == b_elements;
    }

    MPI_Bcast(
        &valid_inputs,
        1,
        MPI_INT,
        0,
        comm
    );

    if (!valid_inputs) {
        throw std::runtime_error(
            "Root matrix dimensions do not match input data"
        );
    }

    std::vector<double> global_B(b_elements);

    if (rank == 0) {
        global_B = B;
    }

    std::vector<int> row_counts(size);
    std::vector<int> row_displacements(size);

    std::vector<int> a_counts(size);
    std::vector<int> a_displacements(size);

    std::vector<int> c_counts(size);
    std::vector<int> c_displacements(size);

    int next_row = 0;

    for (int p = 0; p < size; ++p) {
        row_counts[p] =
            r1 / size + (p < r1 % size ? 1 : 0);

        row_displacements[p] = next_row;

        a_counts[p] = row_counts[p] * c1;
        a_displacements[p] = next_row * c1;

        c_counts[p] = row_counts[p] * c2;
        c_displacements[p] = next_row * c2;

        next_row += row_counts[p];
    }

    const int local_rows = row_counts[rank];
    const int local_a_count = a_counts[rank];
    const int local_c_count = c_counts[rank];

    std::vector<double> local_A(
        std::max<std::size_t>(
            1,
            static_cast<std::size_t>(local_a_count)
        )
    );

    std::vector<double> local_C(
        std::max<std::size_t>(
            1,
            static_cast<std::size_t>(local_c_count)
        ),
        0.0
    );

    MPI_Bcast(
        global_B.data(),
        static_cast<int>(b_elements),
        MPI_DOUBLE,
        0,
        comm
    );

    MPI_Scatterv(
        rank == 0 ? A.data() : nullptr,
        a_counts.data(),
        a_displacements.data(),
        MPI_DOUBLE,
        local_A.data(),
        local_a_count,
        MPI_DOUBLE,
        0,
        comm
    );

    for (int i = 0; i < local_rows; ++i) {
        for (int k = 0; k < c1; ++k) {
            for (int j = 0; j < c2; ++j) {
                local_C[i * c2 + j] +=
                    local_A[i * c1 + k] *
                    global_B[k * c2 + j];
            }
        }
    }

    std::vector<double> C;

    if (rank == 0) {
        C.resize(c_elements);
    }

    MPI_Gatherv(
        local_C.data(),
        local_c_count,
        MPI_DOUBLE,
        rank == 0 ? C.data() : nullptr,
        c_counts.data(),
        c_displacements.data(),
        MPI_DOUBLE,
        0,
        comm
    );

    return C;
}

std::vector<double> matrix_multiplication_mpi_wrapper(
    BenchmarkContext& ctx,
    int r1,
    int c1,
    int r2,
    int c2
) {
    int rank;
    int size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<double> A;
    std::vector<double> B;

    int inputs_ready = 1;

    if (rank == 0) {
        try {
            auto matrix_A = build_matrix(
                r1,
                c1,
                MATRIX_MULTIPLY_A_SEED
            );

            auto matrix_B = build_matrix(
                r2,
                c2,
                MATRIX_MULTIPLY_B_SEED
            );

            if (!matrix_A || !matrix_B) {
                throw std::runtime_error(
                    "Matrix generation failed"
                );
            }

            A = std::move(*matrix_A);
            B = std::move(*matrix_B);
        } catch (...) {
            inputs_ready = 0;
        }
    }

    MPI_Bcast(
        &inputs_ready,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    if (!inputs_ready) {
        throw std::runtime_error(
            "Rank 0 failed to generate matrix inputs"
        );
    }

    ctx.start_measurement();

    std::vector<double> C = matrix_multiply_mpi(
        A,
        B,
        r1,
        c1,
        r2,
        c2,
        MPI_COMM_WORLD
    );

    ctx.stop_measurement();

    ctx.iterations = 1;
    ctx.problem_size = r1 * c2;
    ctx.number_of_processors = size;

    ctx.bytes_moved =
        (
            static_cast<double>(r1) * c1 +
            static_cast<double>(c1) * c2 +
            static_cast<double>(r1) * c2
        ) * sizeof(double);

    ctx.flop_count =
        2.0 * r1 * c2 * c1;

    return C;
}
namespace {

struct MatrixMultiplyConfig {
    const char* suffix;
    int r1;
    int c1;
    int r2;
    int c2;
};

constexpr MatrixMultiplyConfig matrix_configs[] = {
    {"n32x32", 32, 32, 32, 32},
    {"n64x64", 64, 64, 64, 64},
    {"n128x128", 128, 128, 128, 128},
    {"n256x256", 256, 256, 256, 256},
    {"n384x384", 384, 384, 384, 384},
    {"n512x512", 512, 512, 512, 512},
    {"n768x768", 768, 768, 768, 768},
    {"n1024x1024", 1024, 1024, 1024, 1024},
    {"n1536x1536", 1536, 1536, 1536, 1536},
    {"n2048x2048", 2048, 2048, 2048, 2048},
    {"rect_128x256_256x64", 128, 256, 256, 64},
    {"rect_64x128_128x256", 64, 128, 128, 256},
    {"rect_256x64_64x128", 256, 64, 64, 128},
    {"rect_32x256_256x32", 32, 256, 256, 32}
};

constexpr int mpi_rank_counts[] = {
    1, 2, 4, 8, 12, 16, 20
};

struct RegisterMatrixMultiplyMpiBenchmarks {
    RegisterMatrixMultiplyMpiBenchmarks() {
        for (const auto& config : matrix_configs) {
            const std::string kernel_name =
                std::string("matrix_multiplication_") +
                config.suffix;

            for (int ranks : mpi_rank_counts) {
                const std::string benchmark_name =
                    std::string("matrix_multiplication_mpi_") +
                    config.suffix +
                    "_np" +
                    std::to_string(ranks);

                get_registry().register_benchmark(
                    benchmark_name,
                    kernel_name,
                    Paradigm::MPI,
                    [config](BenchmarkContext& ctx) {
                        matrix_multiplication_mpi_wrapper(
                            ctx,
                            config.r1,
                            config.c1,
                            config.r2,
                            config.c2
                        );
                    }
                );
            }
        }
    }
};

RegisterMatrixMultiplyMpiBenchmarks
    register_matrix_multiply_mpi_benchmarks;

}
