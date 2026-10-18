#include <catch2/catch_test_macros.hpp>
#include "cuda_monte_carlo.h"

TEST_CASE("CUDA Monte Carlo pi calculations within reasonable error bounds") {
    double pi = monte_carlo_cuda(10000);
    REQUIRE(pi > 3.1);
    REQUIRE(pi < 3.2);
}

TEST_CASE("CUDA MONTE CARLO rejects non-positive N") {
    REQUIRE_THROWS_AS(monte_carlo_cuda(0), std::invalid_argument);
    REQUIRE_THROWS_AS(monte_carlo_cuda(-100), std::invalid_argument);
}
