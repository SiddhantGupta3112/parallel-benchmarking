#include <catch2/catch_test_macros.hpp>

#include "cuda_stencil.h"
#include "stencil.h"  

#include <cmath>
#include <stdexcept>
#include <vector>

TEST_CASE("CUDA stencil preserves a uniform grid") {
    int rows = 5, cols = 5;
    std::vector<double> grid(rows * cols, 10.0);

    std::vector<double> result = cuda_stencil(grid, rows, cols, 10);

    REQUIRE(result.size() == grid.size());

    for (std::size_t i = 0; i < grid.size(); ++i) {
        REQUIRE(std::abs(result[i] - grid[i]) < 1e-9);
    }
}

TEST_CASE("CUDA stencil converges toward uniform boundary value") {
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

    std::vector<double> result = cuda_stencil(grid, rows, cols, 500);

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            REQUIRE(std::abs(result[i * cols + j] - V) < 0.01);
        }
    }
}

TEST_CASE("CUDA stencil preserves boundary values") {
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

    std::vector<double> result = cuda_stencil(grid, rows, cols, 5);

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

TEST_CASE("CUDA stencil with zero iterations returns input unchanged") {
    int rows = 3, cols = 4;
    std::vector<double> grid = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };

    REQUIRE(cuda_stencil(grid, rows, cols, 0) == grid);
}

TEST_CASE("CUDA stencil performs one iteration correctly") {
    int rows = 3, cols = 3;
    std::vector<double> grid = {
        10, 10, 10,
        10,  0, 10,
        10, 10, 10
    };

    std::vector<double> result = cuda_stencil(grid, rows, cols, 1);

    REQUIRE(result.size() == grid.size());
    REQUIRE(std::abs(result[4] - 10.0) < 1e-9);
}

TEST_CASE("CUDA stencil rejects mismatched input dimensions") {
    std::vector<double> grid(8, 0.0);

    REQUIRE_THROWS_AS(cuda_stencil(grid, 3, 3, 1),
                      std::runtime_error);
}

TEST_CASE("CUDA stencil rejects negative iterations") {
    std::vector<double> grid(9, 0.0);

    REQUIRE_THROWS_AS(cuda_stencil(grid, 3, 3, -1),
                      std::runtime_error);
}

TEST_CASE("CUDA stencil rejects non-positive dimensions") {
    std::vector<double> grid;

    REQUIRE_THROWS_AS(cuda_stencil(grid, 0, 3, 1),
                      std::runtime_error);

    REQUIRE_THROWS_AS(cuda_stencil(grid, 3, -1, 1),
                      std::runtime_error);
}

TEST_CASE("CUDA stencil matches serial on awkward dimensions") {
    const int rows = 50;
    const int cols = 70;
    const int iterations = 10;

    std::vector<double> grid(rows * cols, 0.0);

    for (int j = 0; j < cols; ++j) {
        grid[j] = 10.0;
        grid[(rows - 1) * cols + j] = 10.0;
    }

    for (int i = 0; i < rows; ++i) {
        grid[i * cols] = 10.0;
        grid[i * cols + cols - 1] = 10.0;
    }

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            grid[i * cols + j] = (i * 7 + j * 3) % 17;
        }
    }

    const auto expected = stencil(grid, rows, cols, iterations);
    const auto actual = cuda_stencil(grid, rows, cols, iterations);

    REQUIRE(actual.size() == expected.size());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        INFO("Output index: " << i);
        REQUIRE(std::abs(actual[i] - expected[i]) < 1e-9);
    }
}
