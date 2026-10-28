#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include "mpi_matrix_multiply.h"
#include "matrix_multiply.h"

#include <algorithm>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <stdexcept>

std::vector<double> create_test_matrix(
    int rows,
    int cols,
    int offset
) {
    std::vector<double> matrix(
        static_cast<std::size_t>(rows) * cols
    );

    for (std::size_t i = 0; i < matrix.size(); ++i) {
        matrix[i] =
            (static_cast<int>((i * 7 + offset) % 19) - 9) * 0.125;
    }

    return matrix;
}

void check_mpi_matrix_multiplication(
    int r1,
    int c1,
    int r2,
    int c2
) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<double> A = create_test_matrix(r1, c1, 3);
    std::vector<double> B = create_test_matrix(r2, c2, 11);

    std::vector<double> actual = matrix_multiply_mpi(
        A,
        B,
        r1,
        c1,
        r2,
        c2
    );

    int correct = 1;

    if (rank == 0) {
        std::vector<double> expected = matrix_multiply(
            A,
            B,
            r1,
            c1,
            r2,
            c2
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

    MPI_Bcast(
        &correct,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    REQUIRE(correct == 1);
}

TEST_CASE("MPI matrix multiplication matches Serial for square matrices") {
    check_mpi_matrix_multiplication(1, 1, 1, 1);
    check_mpi_matrix_multiplication(2, 2, 2, 2);
    check_mpi_matrix_multiplication(3, 3, 3, 3);
    check_mpi_matrix_multiplication(31, 31, 31, 31);
    check_mpi_matrix_multiplication(32, 32, 32, 32);
    check_mpi_matrix_multiplication(33, 33, 33, 33);
    check_mpi_matrix_multiplication(100, 100, 100, 100);
}

TEST_CASE("MPI matrix multiplication matches Serial for rectangular matrices") {
    check_mpi_matrix_multiplication(1, 5, 5, 1);
    check_mpi_matrix_multiplication(5, 1, 1, 7);
    check_mpi_matrix_multiplication(31, 17, 17, 33);
    check_mpi_matrix_multiplication(64, 23, 23, 11);
    check_mpi_matrix_multiplication(17, 31, 31, 2);
}

TEST_CASE("MPI matrix multiplication handles more ranks than rows") {
    check_mpi_matrix_multiplication(2, 3, 3, 4);
}
TEST_CASE("MPI matrix multiplication handles repeated calls") {
    check_mpi_matrix_multiplication(7, 5, 5, 3);
    check_mpi_matrix_multiplication(2, 3, 3, 4);
    check_mpi_matrix_multiplication(1, 5, 5, 1);
}

TEST_CASE("MPI matrix multiplication accepts inputs only on rank zero") {
    constexpr int r1 = 7;
    constexpr int c1 = 5;
    constexpr int r2 = 5;
    constexpr int c2 = 3;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<double> A;
    std::vector<double> B;

    if (rank == 0) {
        A = create_test_matrix(r1, c1, 3);
        B = create_test_matrix(r2, c2, 11);
    }

    const std::vector<double> actual =
        matrix_multiply_mpi(A, B, r1, c1, r2, c2);

    int correct = 1;

    if (rank == 0) {
        const std::vector<double> expected =
            matrix_multiply(A, B, r1, c1, r2, c2);

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

TEST_CASE("MPI matrix multiplication rejects invalid dimensions") {
    const std::vector<double> empty;

    REQUIRE_THROWS_AS(
        matrix_multiply_mpi(empty, empty, 0, 2, 2, 2),
        std::runtime_error
    );

    REQUIRE_THROWS_AS(
        matrix_multiply_mpi(empty, empty, 2, 0, 0, 2),
        std::runtime_error
    );

    REQUIRE_THROWS_AS(
        matrix_multiply_mpi(empty, empty, 2, 3, 4, 2),
        std::runtime_error
    );
}

TEST_CASE("MPI matrix multiplication rejects malformed root input") {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<double> A;
    std::vector<double> B;

    if (rank == 0) {
        A = create_test_matrix(3, 2, 3);
        B = create_test_matrix(2, 4, 11);
        A.pop_back();
    }

    REQUIRE_THROWS_AS(
        matrix_multiply_mpi(A, B, 3, 2, 2, 4),
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
