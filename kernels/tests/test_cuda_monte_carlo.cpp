#include <catch2/catch_test_macros.hpp>
#include "cuda_monte_carlo.h"

TEST_CASE("CUDA Monte Carlo pi calculations within reasonable error bounds") {
    BenchmarkContext ctx;

    double pi = monte_carlo_cuda(ctx, 10000);
    REQUIRE(pi > 3.1);
    REQUIRE(pi < 3.2);
}
