#include "build_matrix.h"

#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <vector>

TEST_CASE("Matrix is built with the correct total size") {
    int r = 10;
    int c = 10;
    int total_elements = r * c;

    std::unique_ptr<std::vector<double>> built_matrix = build_matrix(r, c, 1234U);

    REQUIRE(built_matrix->size() == total_elements);
}

TEST_CASE("Matrices built with identical seeds are exactly identical") {
    int r = 10;
    int c = 10;

    std::unique_ptr<std::vector<double>> built_matrix_1 = build_matrix(r, c, 1234U);
    std::unique_ptr<std::vector<double>> built_matrix_2 = build_matrix(r, c, 1234U);

    REQUIRE(*built_matrix_1 == *built_matrix_2);
}

TEST_CASE("Matrices built with different seeds are not identical") {
    int r = 10;
    int c = 10;

    std::unique_ptr<std::vector<double>> built_matrix_1 = build_matrix(r, c, 1234U);
    std::unique_ptr<std::vector<double>> built_matrix_2 = build_matrix(r, c, 5678U);

    REQUIRE(*built_matrix_1 != *built_matrix_2);
}

TEST_CASE("Matrix elements fall within the valid real distribution bounds") {
    int r = 10;
    int c = 10;
    std::unique_ptr<std::vector<double>> built_matrix = build_matrix(r, c, 1234U);

    for (double val : *built_matrix) {
        REQUIRE(val >= 0.0);
        REQUIRE(val <= 1.0);
    }
}


