#include <catch2/catch_test_macros.hpp>
#include "stencil.h"

TEST_CASE("Stencil converges toward uniform boundary value") {
    BenchmarkContext ctx;
    int rows = 5, cols = 5;
    double V = 10.0;

    std::vector<double> grid(rows * cols, 0.0);
    for (int j = 0; j < cols; j++) {
        grid[0 * cols + j] = V;
        grid[(rows - 1) * cols + j] = V;
    }
    for (int i = 0; i < rows; i++) {
        grid[i * cols + 0] = V;
        grid[i * cols + (cols - 1)] = V;
    }

    std::vector<double> result = stencil(ctx, grid, rows, cols, 50000);

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            REQUIRE(std::abs(result[i * cols + j] - V) < 0.01);
        }
    }
}