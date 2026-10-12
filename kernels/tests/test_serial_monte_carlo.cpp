#include <catch2/catch_test_macros.hpp>
#include "monte_carlo.h"

TEST_CASE("Serial Monte Carlo pi calculations within reasonable error bounds") {
    BenchmarkContext ctx;

    double pi = monte_carlo(ctx, 10000);
    REQUIRE(pi > 3.1);
    REQUIRE(pi < 3.2);
}

TEST_CASE("Serial Monte Carlo rejects non-positive N") {
    BenchmarkContext ctx;

    REQUIRE_THROWS_AS(monte_carlo(ctx, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(monte_carlo(ctx, -100), std::invalid_argument);
}