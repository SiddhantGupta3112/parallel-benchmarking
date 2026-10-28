
#include "build_stencil_input.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

TEST_CASE("Stencil input has the correct total size") {
    int r = 10;
    int c = 10;

    auto matrix = build_stencil_input(r, c, 1234U);

    REQUIRE(matrix->size() == static_cast<std::size_t>(r) * c);
}

TEST_CASE("Stencil inputs with identical seeds are identical") {
    int r = 10;
    int c = 10;

    auto matrix1 = build_stencil_input(r, c, 1234U);
    auto matrix2 = build_stencil_input(r, c, 1234U);

    REQUIRE(*matrix1 == *matrix2);
}

TEST_CASE("Stencil inputs with different seeds have different boundary values") {
    int r = 10;
    int c = 10;

    auto matrix1 = build_stencil_input(r, c, 1234U);
    auto matrix2 = build_stencil_input(r, c, 5678U);

    // Compare a boundary element, not the entire matrix:
    // all interior elements are zero in both matrices.
    REQUIRE((*matrix1)[0] != (*matrix2)[0]);
}

TEST_CASE("Stencil input has constant boundaries and zero interior") {
    int r = 10;
    int c = 10;

    auto matrix = build_stencil_input(r, c, 1234U);
    double boundary_value = (*matrix)[0];

    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            double value = (*matrix)[i * c + j];

            if (i == 0 || i == r - 1 || j == 0 || j == c - 1) {
                REQUIRE(value == boundary_value);
            } else {
                REQUIRE(value == 0.0);
            }
        }
    }

    REQUIRE(boundary_value > 0.0);
    REQUIRE(boundary_value <= 1.0);
}

TEST_CASE("Stencil input rejects dimensions smaller than 3x3") {
    SECTION("Too few rows") {
        REQUIRE_THROWS_AS(
            build_stencil_input(2, 10, 1234U),
            std::runtime_error
        );
    }

    SECTION("Too few columns") {
        REQUIRE_THROWS_AS(
            build_stencil_input(10, 2, 1234U),
            std::runtime_error
        );
    }

    SECTION("Zero dimensions") {
        REQUIRE_THROWS_AS(
            build_stencil_input(0, 0, 1234U),
            std::runtime_error
        );
    }
}
