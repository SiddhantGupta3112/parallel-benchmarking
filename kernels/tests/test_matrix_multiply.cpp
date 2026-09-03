#include <catch2/catch_test_macros.hpp>
#include "matrix_multiply.h"
#include <stdexcept>
#include <vector>

TEST_CASE("Correct matrix multiplication with identity matrix") {
    BenchmarkContext ctx;

    int r1 = 3;
    int c1 = 3;
    int r2 = 3;
    int c2 = 3;

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

    std::vector<double> result =
        matrix_multiply(ctx, A, I, r1, c1, r2, c2);

    REQUIRE(result == A);
}


TEST_CASE("Matrix multiplication rejects invalid sizes") {

    BenchmarkContext ctx;

    SECTION("Incompatible dimensions") {
        int r1 = 2;
        int c1 = 3;
        int r2 = 2;
        int c2 = 2;

        std::vector<double> A = {
            1, 2, 3,
            4, 5, 6
        };

        std::vector<double> B = {
            1, 2,
            3, 4
        };

        REQUIRE_THROWS_AS(
            matrix_multiply(ctx, A, B, r1, c1, r2, c2),
            std::runtime_error
        );
    }

    SECTION("Empty A") {
        int r1 = 0;
        int c1 = 0;
        int r2 = 2;
        int c2 = 2;

        std::vector<double> A;

        std::vector<double> B = {
            1, 0,
            0, 1
        };

        REQUIRE_THROWS_AS(
            matrix_multiply(ctx, A, B, r1, c1, r2, c2),
            std::runtime_error
        );
    }

    SECTION("Empty B") {
        int r1 = 2;
        int c1 = 2;
        int r2 = 0;
        int c2 = 0;

        std::vector<double> A = {
            1, 2,
            3, 4
        };

        std::vector<double> B;

        REQUIRE_THROWS_AS(
            matrix_multiply(ctx, A, B, r1, c1, r2, c2),
            std::runtime_error
        );
    }
}

