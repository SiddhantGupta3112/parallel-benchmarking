#include <catch2/catch_test_macros.hpp>
#include "matrix_multiply.h"

#include <stdexcept>
#include <vector>

TEST_CASE("Matrix multiplication with identity matrix") {
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

    REQUIRE(matrix_multiply(A, I, 3, 3, 3, 3) == A);
}

TEST_CASE("Matrix multiplication produces correct result") {
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

    REQUIRE(matrix_multiply(A, B, 2, 3, 3, 2) == expected);
}

TEST_CASE("Matrix multiplication handles rectangular matrices") {
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

    REQUIRE(matrix_multiply(A, B, 2, 3, 3, 1) == expected);
}

TEST_CASE("Matrix multiplication rejects invalid dimensions") {
    std::vector<double> A = {1, 2, 3, 4};
    std::vector<double> B = {5, 6, 7, 8};

    SECTION("Incompatible dimensions") {
        // A is 2x2; B is 1x4. Inner dimensions differ.
        REQUIRE_THROWS_AS(
            matrix_multiply(A, B, 2, 2, 1, 4),
            std::runtime_error
        );
    }

    SECTION("Zero dimensions") {
        REQUIRE_THROWS_AS(
            matrix_multiply(A, B, 0, 2, 2, 2),
            std::runtime_error
        );
    }

    SECTION("Negative dimensions") {
        REQUIRE_THROWS_AS(
            matrix_multiply(A, B, -2, 2, 2, 2),
            std::runtime_error
        );
    }
}

TEST_CASE("Matrix multiplication rejects mismatched input sizes") {
    std::vector<double> A = {
        1, 2, 3
    };

    std::vector<double> B = {
        4, 5,
        6, 7
    };

    SECTION("A contains too few elements") {
        REQUIRE_THROWS_AS(
            matrix_multiply(A, B, 2, 2, 2, 2),
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
            matrix_multiply(valid_A, short_B, 2, 2, 2, 2),
            std::runtime_error
        );
    }
}
