#include <catch2/catch_test_macros.hpp>

#include "stencil.h"

#include <cmath>
#include <stdexcept>
#include <vector>

TEST_CASE("Stencil preserves a uniform grid") {
    int rows = 5, cols = 5;
    std::vector<double> grid(rows * cols, 10.0);

    std::vector<double> result = stencil(grid, rows, cols, 10);

    REQUIRE(result == grid);
}

TEST_CASE("Stencil converges toward uniform boundary value") {
    int rows = 5, cols = 5;
    double V = 10.0;

    std::vector<double> grid(rows * cols, 0.0);

    for (int j = 0; j < cols; ++j) {
        grid[j] = V;
        grid[(rows - 1) * cols + j] = V;
    }

    for (int i = 0; i < rows; ++i) {
        grid[i * cols] = V;
        grid[i * cols + cols - 1] = V;
    }

    std::vector<double> result = stencil(grid, rows, cols, 500);

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            REQUIRE(std::abs(result[i * cols + j] - V) < 0.01);
        }
    }
}

TEST_CASE("Stencil preserves all boundary values") {
    int rows = 4, cols = 5;
    std::vector<double> grid(rows * cols, 0.0);

    for (int j = 0; j < cols; ++j) {
        grid[j] = 1.0 + j;
        grid[(rows - 1) * cols + j] = 10.0 + j;
    }

    for (int i = 0; i < rows; ++i) {
        grid[i * cols] = 20.0 + i;
        grid[i * cols + cols - 1] = 30.0 + i;
    }

    std::vector<double> result = stencil(grid, rows, cols, 5);

    for (int j = 0; j < cols; ++j) {
        REQUIRE(result[j] == grid[j]);
        REQUIRE(result[(rows - 1) * cols + j] ==
                grid[(rows - 1) * cols + j]);
    }

    for (int i = 0; i < rows; ++i) {
        REQUIRE(result[i * cols] == grid[i * cols]);
        REQUIRE(result[i * cols + cols - 1] ==
                grid[i * cols + cols - 1]);
    }
}

TEST_CASE("Stencil with zero iterations returns the input unchanged") {
    int rows = 3, cols = 4;
    std::vector<double> grid = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };

    REQUIRE(stencil(grid, rows, cols, 0) == grid);
}

TEST_CASE("Stencil performs one iteration correctly") {
    int rows = 3, cols = 3;
    std::vector<double> grid = {
        10, 10, 10,
        10,  0, 10,
        10, 10, 10
    };

    std::vector<double> result = stencil(grid, rows, cols, 1);

    REQUIRE(result[4] == 10.0);

    for (int i = 0; i < rows; ++i) {
        REQUIRE(result[i * cols] == grid[i * cols]);
        REQUIRE(result[i * cols + cols - 1] ==
                grid[i * cols + cols - 1]);
    }
}

TEST_CASE("Stencil rejects mismatched input dimensions") {
    std::vector<double> grid(8, 0.0);

    REQUIRE_THROWS_AS(stencil(grid, 3, 3, 1), std::runtime_error);
}

TEST_CASE("Stencil with zero iterations preserves a small grid") {
    std::vector<double> grid = {1, 2, 3, 4};

    REQUIRE(stencil(grid, 2, 2, 0) == grid);
}
