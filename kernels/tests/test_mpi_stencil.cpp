#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include "mpi_stencil.h"
#include "stencil.h"

#include <algorithm>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <stdexcept>

std::vector<double> create_stencil_test_input(int rows, int cols) {
    std::vector<double> input(
        static_cast<std::size_t>(rows) * cols
    );

    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<double>((i * 7 + 3) % 23) / 5.0;
    }

    return input;
}

void check_mpi_stencil(int rows, int cols, int iterations) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<double> input = create_stencil_test_input(rows, cols);

    std::vector<double> actual = stencil_mpi(
        input, rows, cols, iterations
    );

    int correct = 1;

    if (rank == 0) {
        std::vector<double> expected = stencil(
            input, rows, cols, iterations
        );

        if (actual.size() != expected.size()) {
            correct = 0;
        } else {
            for (std::size_t i = 0; i < expected.size(); ++i) {
                double tolerance =
                    1e-10 * std::max(1.0, std::abs(expected[i]));

                if (std::abs(actual[i] - expected[i]) > tolerance) {
                    correct = 0;
                    break;
                }
            }
        }
    }

    MPI_Bcast(&correct, 1, MPI_INT, 0, MPI_COMM_WORLD);

    REQUIRE(correct == 1);
}

TEST_CASE("MPI stencil matches serial for square and rectangular grids") {
    check_mpi_stencil(8, 8, 5);
    check_mpi_stencil(9, 7, 4);
    check_mpi_stencil(17, 11, 10);
}

TEST_CASE("MPI stencil preserves input when iterations are zero") {
    check_mpi_stencil(7, 9, 0);
}

TEST_CASE("MPI stencil matches serial for one iteration") {
    check_mpi_stencil(5, 6, 1);
}

TEST_CASE("MPI stencil handles more ranks than rows") {
    check_mpi_stencil(3, 5, 4);
}
TEST_CASE("MPI stencil handles the minimum valid grid") {
    check_mpi_stencil(3, 3, 0);
    check_mpi_stencil(3, 3, 1);
    check_mpi_stencil(3, 3, 3);
}
TEST_CASE("MPI stencil repeated calls leave no stale state") {
    check_mpi_stencil(5, 7, 1);
    check_mpi_stencil(9, 4, 3);
    check_mpi_stencil(3, 3, 0);
    check_mpi_stencil(8, 6, 5);
}

TEST_CASE("MPI stencil preserves all boundary cells") {
    constexpr int rows = 7;
    constexpr int cols = 9;
    constexpr int iterations = 5;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const std::vector<double> input =
        create_stencil_test_input(rows, cols);

    const std::vector<double> actual =
        stencil_mpi(input, rows, cols, iterations);

    int boundaries_preserved = 1;

    if (rank == 0) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (r == 0 || r == rows - 1 ||
                    c == 0 || c == cols - 1) {
                    const std::size_t index =
                        static_cast<std::size_t>(r) * cols + c;

                    if (actual.size() != input.size() ||
                        actual[index] != input[index]) {
                        boundaries_preserved = 0;
                    }
                }
            }
        }
    }

    MPI_Bcast(
        &boundaries_preserved, 1, MPI_INT, 0, MPI_COMM_WORLD
    );

    REQUIRE(boundaries_preserved == 1);
}

TEST_CASE("MPI stencil accepts input only on rank zero") {
    constexpr int rows = 7;
    constexpr int cols = 9;
    constexpr int iterations = 4;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<double> input;

    if (rank == 0) {
        input = create_stencil_test_input(rows, cols);
    }

    const std::vector<double> actual =
        stencil_mpi(input, rows, cols, iterations);

    int correct = 1;

    if (rank == 0) {
        const std::vector<double> expected =
            stencil(input, rows, cols, iterations);

        if (actual.size() != expected.size()) {
            correct = 0;
        } else {
            for (std::size_t i = 0; i < expected.size(); ++i) {
                const double tolerance =
                    1e-10 * std::max(1.0, std::abs(expected[i]));

                if (std::abs(actual[i] - expected[i]) > tolerance) {
                    correct = 0;
                    break;
                }
            }
        }
    }

    MPI_Bcast(&correct, 1, MPI_INT, 0, MPI_COMM_WORLD);

    REQUIRE(correct == 1);
}

TEST_CASE("MPI stencil rejects dimensions smaller than 3") {
    const std::vector<double> input;

    REQUIRE_THROWS_AS(
        stencil_mpi(input, 2, 3, 1),
        std::runtime_error
    );

    REQUIRE_THROWS_AS(
        stencil_mpi(input, 3, 2, 1),
        std::runtime_error
    );

    REQUIRE_THROWS_AS(
        stencil_mpi(input, 0, 3, 1),
        std::runtime_error
    );
}

TEST_CASE("MPI stencil rejects negative iteration counts") {
    const std::vector<double> input;

    REQUIRE_THROWS_AS(
        stencil_mpi(input, 3, 3, -1),
        std::runtime_error
    );
}

TEST_CASE("MPI stencil rejects invalid root input size") {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<double> input;

    if (rank == 0) {
        input = create_stencil_test_input(3, 3);
        input.pop_back();
    }

    REQUIRE_THROWS_AS(
        stencil_mpi(input, 3, 3, 1),
        std::runtime_error
    );
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int result = 0;

    try {
        result = Catch::Session().run(argc, argv);
    } catch (...) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Finalize();

    return result;
}
