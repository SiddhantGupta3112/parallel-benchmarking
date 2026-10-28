#include <catch2/catch_test_macros.hpp>

#include "openmp_matrix_multiply.h"
#include "matrix_multiply.h"
#include "build_matrix.h"
#include "seed_values.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <omp.h>

namespace {

// Sweep counts from the benchmark plus awkward ones (3, 7). Many are larger
// than the number of rows in the small cases below, and many do not divide
// the row count, so threads end up with unequal (or empty) row ranges.
const std::vector<int> THREAD_COUNTS = {1, 2, 3, 4, 7, 8, 12, 16, 20};

void require_matches_serial(
    const std::vector<double>& A,
    const std::vector<double>& B,
    int r1,
    int c1,
    int r2,
    int c2
) {
    const auto expected = matrix_multiply(A, B, r1, c1, r2, c2);

    for (int t : THREAD_COUNTS) {
        CAPTURE(r1, c1, r2, c2, t);

        const auto actual = openmp_matrix_multiply(A, B, r1, c1, r2, c2, t);

        REQUIRE(actual.size() == expected.size());

        for (std::size_t i = 0; i < expected.size(); ++i) {
            INFO("Output index: " << i);
            REQUIRE(std::abs(actual[i] - expected[i]) < 1e-9);
        }
    }
}

} // namespace

TEST_CASE("OpenMP matrix multiplication with identity matrix") {
    std::vector<double> A = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    std::vector<double> I = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };

    for (int t : THREAD_COUNTS) {
        CAPTURE(t);
        REQUIRE(openmp_matrix_multiply(A, I, 3, 3, 3, 3, t) == A);
    }
}

TEST_CASE("OpenMP matrix multiplication produces correct result") {
    std::vector<double> A = {
        1, 2, 3,
        4, 5, 6
    };

    std::vector<double> B = {
        7,  8,
        9,  10,
        11, 12
    };

    std::vector<double> expected = {
        58,  64,
        139, 154
    };

    for (int t : THREAD_COUNTS) {
        CAPTURE(t);
        REQUIRE(openmp_matrix_multiply(A, B, 2, 3, 3, 2, t) == expected);
    }
}

TEST_CASE("OpenMP matrix multiplication handles rectangular matrices") {
    std::vector<double> A = {
        1, 2, 3,
        4, 5, 6
    };

    std::vector<double> B = {
        2,
        3,
        4
    };

    std::vector<double> expected = {
        20,
        47
    };

    for (int t : THREAD_COUNTS) {
        CAPTURE(t);
        REQUIRE(openmp_matrix_multiply(A, B, 2, 3, 3, 1, t) == expected);
    }
}

TEST_CASE("OpenMP matrix multiplication rejects invalid dimensions") {
    std::vector<double> A = {1, 2, 3, 4};
    std::vector<double> B = {5, 6, 7, 8};

    SECTION("Incompatible dimensions") {
        REQUIRE_THROWS_AS(
            openmp_matrix_multiply(A, B, 2, 2, 1, 4, 4),
            std::runtime_error
        );
    }

    SECTION("Zero dimensions") {
        REQUIRE_THROWS_AS(
            openmp_matrix_multiply(A, B, 0, 2, 2, 2, 4),
            std::runtime_error
        );
    }

    SECTION("Negative dimensions") {
        REQUIRE_THROWS_AS(
            openmp_matrix_multiply(A, B, -2, 2, 2, 2, 4),
            std::runtime_error
        );
    }
}

TEST_CASE("OpenMP matrix multiplication rejects mismatched input sizes") {
    std::vector<double> A = {
        1, 2, 3
    };

    std::vector<double> B = {
        4, 5,
        6, 7
    };

    SECTION("A contains too few elements") {
        REQUIRE_THROWS_AS(
            openmp_matrix_multiply(A, B, 2, 2, 2, 2, 4),
            std::runtime_error
        );
    }

    SECTION("B contains too few elements") {
        std::vector<double> valid_A = {
            1, 2,
            3, 4
        };

        std::vector<double> short_B = {
            5, 6, 7
        };

        REQUIRE_THROWS_AS(
            openmp_matrix_multiply(valid_A, short_B, 2, 2, 2, 2, 4),
            std::runtime_error
        );
    }
}

TEST_CASE("OpenMP matrix multiplication rejects non-positive thread counts") {
    std::vector<double> A = {1, 2, 3, 4};
    std::vector<double> B = {5, 6, 7, 8};

    REQUIRE_THROWS_AS(openmp_matrix_multiply(A, B, 2, 2, 2, 2, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(openmp_matrix_multiply(A, B, 2, 2, 2, 2, -3), std::invalid_argument);
}

TEST_CASE("OpenMP matrix multiplication matches serial for awkward dimensions") {
    const int r1 = 37;
    const int c1 = 53;
    const int r2 = 53;
    const int c2 = 29;

    std::vector<double> A(r1 * c1);
    std::vector<double> B(r2 * c2);

    for (int i = 0; i < r1 * c1; ++i) {
        A[i] = (i % 17 - 8) * 0.25;
    }

    for (int i = 0; i < r2 * c2; ++i) {
        B[i] = (i % 13 - 6) * 0.5;
    }

    require_matches_serial(A, B, r1, c1, r2, c2);
}

TEST_CASE("OpenMP matrix multiplication matches serial for sizes not divisible by the thread count") {
    struct Shape { int r1, c1, c2; };

    // Prime and tiny dimensions, a single row, a single column, a single inner
    // element, and fewer rows than most thread counts.
    const Shape shapes[] = {
        {1, 1, 1},
        {1, 50, 1},
        {50, 1, 1},
        {1, 1, 50},
        {2, 3, 4},
        {5, 3, 2},
        {7, 13, 5},
        {31, 64, 33},
        {61, 17, 23},
        {100, 37, 53},
    };

    for (const Shape& s : shapes) {
        // Same generator and seeds as the benchmark wrapper uses.
        std::vector<double> A = std::move(*build_matrix(s.r1, s.c1, MATRIX_MULTIPLY_A_SEED));
        std::vector<double> B = std::move(*build_matrix(s.c1, s.c2, MATRIX_MULTIPLY_B_SEED));

        require_matches_serial(A, B, s.r1, s.c1, s.c1, s.c2);
    }
}

TEST_CASE("OpenMP matrix multiplication matches serial for a benchmark-sized square matrix") {
    const int n = 128;

    std::vector<double> A = std::move(*build_matrix(n, n, MATRIX_MULTIPLY_A_SEED));
    std::vector<double> B = std::move(*build_matrix(n, n, MATRIX_MULTIPLY_B_SEED));

    require_matches_serial(A, B, n, n, n, n);
}

TEST_CASE("OpenMP matrix multiplication returns fresh output on every call") {
    std::vector<double> A = std::move(*build_matrix(9, 7, MATRIX_MULTIPLY_A_SEED));
    std::vector<double> B = std::move(*build_matrix(7, 5, MATRIX_MULTIPLY_B_SEED));

    const auto expected = matrix_multiply(A, B, 9, 7, 7, 5);

    auto first = openmp_matrix_multiply(A, B, 9, 7, 7, 5, 4);

    // Scribble over the first result; a second call must not be affected.
    for (double& v : first) {
        v = -1.0;
    }

    const auto second = openmp_matrix_multiply(A, B, 9, 7, 7, 5, 4);

    REQUIRE(second.size() == expected.size());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(std::abs(second[i] - expected[i]) < 1e-9);
    }
}

TEST_CASE("OpenMP matrix multiplication leaves the global thread count unchanged") {
    std::vector<double> A = {1, 2, 3, 4};
    std::vector<double> B = {5, 6, 7, 8};

    const int original = omp_get_max_threads();
    const int requested = original == 3 ? 5 : 3;

    openmp_matrix_multiply(A, B, 2, 2, 2, 2, requested);

    REQUIRE(omp_get_max_threads() == original);
}
