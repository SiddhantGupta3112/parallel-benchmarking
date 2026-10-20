#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include "mpi_monte_carlo.h"

#include <cmath>
#include <mpi.h>
#include <stdexcept>

TEST_CASE("MPI Monte Carlo estimates pi accurately") {
    constexpr int N = 100000;

    const double estimate = monte_carlo_mpi(N);
    const double pi = std::acos(-1.0);

    REQUIRE(std::isfinite(estimate));
    REQUIRE(std::abs(estimate - pi) < 0.03);
}

TEST_CASE("MPI Monte Carlo rejects non-positive sample counts") {
    REQUIRE_THROWS_AS(
        monte_carlo_mpi(0),
        std::invalid_argument
    );

    REQUIRE_THROWS_AS(
        monte_carlo_mpi(-100),
        std::invalid_argument
    );
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int result = 0;

    try {
        result = Catch::Session().run(argc, argv);
    } catch (...) {
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Finalize();

    return result;
}
