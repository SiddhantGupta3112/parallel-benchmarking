#include <catch2/catch_test_macros.hpp>
#include "cuda_matrix_multiply.h"
#include "matrix_multiply.h"
#include <stdexcept>
#include <vector>
#include <cmath>

TEST_CASE("Cuda matrix multiplication with identity matrix") {
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

    REQUIRE(cuda_matrix_multiply(A, I, 3, 3, 3, 3) == A);
    
}

TEST_CASE("Cuda matrix multiplication produces correct result") {
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

    REQUIRE(cuda_matrix_multiply(A, B, 2, 3, 3, 2) == expected);
}

TEST_CASE("CUDA matrix multiplication handles rectangular matrices") {
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

    REQUIRE(cuda_matrix_multiply(A, B, 2, 3, 3, 1) == expected);
}

TEST_CASE("Cuda matrix multiplication rejects invalid dimensions") {
    std::vector<double> A = {1, 2, 3, 4};
    std::vector<double> B = {5, 6, 7, 8};

    SECTION("Incompatible dimensions") {
        REQUIRE_THROWS_AS(
            cuda_matrix_multiply(A, B, 2, 2, 1, 4),
            std::runtime_error
        );
    }

    SECTION("Zero dimensions") {
        REQUIRE_THROWS_AS(
            cuda_matrix_multiply(A, B, 0, 2, 2, 2),
            std::runtime_error
        );
    }

    SECTION("Negative dimensions") {
        REQUIRE_THROWS_AS(
            cuda_matrix_multiply(A, B, -2, 2, 2, 2),
            std::runtime_error
        );
    }
}

TEST_CASE("Cuda matrix multiplication rejects mismatched input sizes") {
    std::vector<double> A = {
        1, 2, 3
    };

    std::vector<double> B = {
        4, 5,
        6, 7
    };

    SECTION("A contains too few elements") {
        REQUIRE_THROWS_AS(
            cuda_matrix_multiply(A, B, 2, 2, 2, 2),
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
            cuda_matrix_multiply(valid_A, short_B, 2, 2, 2, 2),
            std::runtime_error
        );
    }
}

TEST_CASE("CUDA matrix multiplication matches serial for awkward dimensions") {
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

    const auto expected = matrix_multiply(A, B, r1, c1, r2, c2);
    const auto actual = cuda_matrix_multiply(A, B, r1, c1, r2, c2);

    REQUIRE(actual.size() == expected.size());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        INFO("Output index: " << i);
        REQUIRE(std::abs(actual[i] - expected[i]) < 1e-9);
    }
}
