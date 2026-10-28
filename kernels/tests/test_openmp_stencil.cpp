#include <catch2/catch_test_macros.hpp>

#include "openmp_stencil.h"
#include "stencil.h"
#include "build_matrix.h"
#include "build_stencil_input.h"
#include "seed_values.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <omp.h>

namespace {

// Sweep counts from the benchmark plus awkward ones (3, 7). Many are larger
// than the number of interior rows in the small cases below, and many do not
// divide the row count, so threads end up with unequal (or empty) row ranges.
const std::vector<int> THREAD_COUNTS = {1, 2, 3, 4, 7, 8, 12, 16, 20};

// The OpenMP stencil performs exactly the same floating-point operations in
// the same order as the serial one (no reductions), so results must be
// bit-identical, not merely close.
void require_matches_serial(
    const std::vector<double>& grid,
    int rows,
    int cols,
    int iterations
) {
    const auto expected = stencil(grid, rows, cols, iterations);

    for (int t : THREAD_COUNTS) {
        CAPTURE(rows, cols, iterations, t);

        const auto actual = openmp_stencil(grid, rows, cols, iterations, t);

        REQUIRE(actual == expected);
    }
}

} // namespace

TEST_CASE("OpenMP stencil preserves a uniform grid") {
    int rows = 5, cols = 5;
    std::vector<double> grid(rows * cols, 10.0);

    for (int t : THREAD_COUNTS) {
        CAPTURE(t);

        std::vector<double> result = openmp_stencil(grid, rows, cols, 10, t);

        REQUIRE(result.size() == grid.size());

        for (std::size_t i = 0; i < grid.size(); ++i) {
            REQUIRE(std::abs(result[i] - grid[i]) < 1e-9);
        }
    }
}

TEST_CASE("OpenMP stencil converges toward uniform boundary value") {
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

    for (int t : THREAD_COUNTS) {
        CAPTURE(t);

        std::vector<double> result = openmp_stencil(grid, rows, cols, 500, t);

        for (int i = 1; i < rows - 1; ++i) {
            for (int j = 1; j < cols - 1; ++j) {
                REQUIRE(std::abs(result[i * cols + j] - V) < 0.01);
            }
        }
    }
}

TEST_CASE("OpenMP stencil preserves boundary values") {
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

    for (int t : THREAD_COUNTS) {
        CAPTURE(t);

        std::vector<double> result = openmp_stencil(grid, rows, cols, 5, t);

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
}

TEST_CASE("OpenMP stencil with zero iterations returns input unchanged") {
    int rows = 3, cols = 4;
    std::vector<double> grid = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };

    for (int t : THREAD_COUNTS) {
        CAPTURE(t);
        REQUIRE(openmp_stencil(grid, rows, cols, 0, t) == grid);
    }
}

TEST_CASE("OpenMP stencil with zero iterations preserves a small grid") {
    std::vector<double> grid = {1, 2, 3, 4};

    for (int t : THREAD_COUNTS) {
        CAPTURE(t);
        REQUIRE(openmp_stencil(grid, 2, 2, 0, t) == grid);
    }
}

TEST_CASE("OpenMP stencil performs one iteration correctly") {
    int rows = 3, cols = 3;
    std::vector<double> grid = {
        10, 10, 10,
        10,  0, 10,
        10, 10, 10
    };

    for (int t : THREAD_COUNTS) {
        CAPTURE(t);

        std::vector<double> result = openmp_stencil(grid, rows, cols, 1, t);

        REQUIRE(result.size() == grid.size());
        REQUIRE(std::abs(result[4] - 10.0) < 1e-9);
    }
}

TEST_CASE("OpenMP stencil rejects mismatched input dimensions") {
    std::vector<double> grid(8, 0.0);

    REQUIRE_THROWS_AS(openmp_stencil(grid, 3, 3, 1, 4),
                      std::runtime_error);
}

TEST_CASE("OpenMP stencil rejects negative iterations") {
    std::vector<double> grid(9, 0.0);

    REQUIRE_THROWS_AS(openmp_stencil(grid, 3, 3, -1, 4),
                      std::runtime_error);
}

TEST_CASE("OpenMP stencil rejects non-positive dimensions") {
    std::vector<double> grid;

    REQUIRE_THROWS_AS(openmp_stencil(grid, 0, 3, 1, 4),
                      std::runtime_error);

    REQUIRE_THROWS_AS(openmp_stencil(grid, 3, -1, 1, 4),
                      std::runtime_error);
}

TEST_CASE("OpenMP stencil rejects non-positive thread counts") {
    std::vector<double> grid(9, 0.0);

    REQUIRE_THROWS_AS(openmp_stencil(grid, 3, 3, 1, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(openmp_stencil(grid, 3, 3, 1, -2), std::invalid_argument);
}

TEST_CASE("OpenMP stencil matches serial on awkward dimensions") {
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

    require_matches_serial(grid, rows, cols, iterations);
}

TEST_CASE("OpenMP stencil matches serial for sizes not divisible by the thread count") {
    struct Shape { int rows, cols, iterations; };

    // Prime and tiny shapes (including ones with no interior at all), odd and
    // even iteration counts (the result ends up in different ping-pong buffers),
    // and fewer interior rows than most thread counts.
    const Shape shapes[] = {
        {1, 1, 3},
        {1, 9, 3},
        {9, 1, 3},
        {2, 9, 4},
        {3, 3, 1},
        {3, 3, 2},
        {4, 7, 5},
        {7, 5, 10},
        {7, 5, 11},
        {17, 19, 8},
        {31, 64, 7},
        {101, 37, 8},
    };

    for (const Shape& s : shapes) {
        // Arbitrary values everywhere (boundary and interior), so a wrong
        // neighbour or a missed cell cannot go unnoticed.
        std::vector<double> grid = std::move(*build_matrix(s.rows, s.cols, STENCIL_SEED));

        require_matches_serial(grid, s.rows, s.cols, s.iterations);
    }
}

TEST_CASE("OpenMP stencil matches serial for the benchmark input") {
    const int rows = 67;
    const int cols = 129;

    // Same generator and seed as the benchmark wrapper uses.
    std::vector<double> grid = std::move(*build_stencil_input(rows, cols, STENCIL_SEED));

    require_matches_serial(grid, rows, cols, 10);
}

TEST_CASE("OpenMP stencil returns fresh output on every call") {
    const int rows = 11;
    const int cols = 13;

    std::vector<double> grid = std::move(*build_matrix(rows, cols, STENCIL_SEED));
    const auto expected = stencil(grid, rows, cols, 6);

    auto first = openmp_stencil(grid, rows, cols, 6, 4);

    // Scribble over the first result; a second call must not be affected.
    for (double& v : first) {
        v = -1.0;
    }

    REQUIRE(openmp_stencil(grid, rows, cols, 6, 4) == expected);
}

TEST_CASE("OpenMP stencil leaves the global thread count unchanged") {
    std::vector<double> grid(9, 1.0);

    const int original = omp_get_max_threads();
    const int requested = original == 3 ? 5 : 3;

    openmp_stencil(grid, 3, 3, 2, requested);

    REQUIRE(omp_get_max_threads() == original);
}
