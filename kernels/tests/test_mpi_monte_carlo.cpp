#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include "mpi_monte_carlo.h"
#include "monte_carlo.h"

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

TEST_CASE("MPI Monte Carlo handles uneven and tiny sample counts") {
    for (int N : {1, 7, 1001}) {
        const double estimate = monte_carlo_mpi(N);

        REQUIRE(std::isfinite(estimate));
        REQUIRE(estimate >= 0.0);
        REQUIRE(estimate <= 4.0);
    }
}

TEST_CASE("MPI Monte Carlo repeated calls are deterministic") {
    constexpr int N = 1001;

    const double first = monte_carlo_mpi(N);
    const double second = monte_carlo_mpi(N);
    const double third = monte_carlo_mpi(N);

    REQUIRE(first == second);
    REQUIRE(second == third);
}

TEST_CASE("MPI Monte Carlo matches Serial with one rank") {
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 1) {
        SUCCEED("Serial equivalence applies only to one MPI rank");
        return;
    }

    constexpr int N = 1001;

    const double mpi_estimate = monte_carlo_mpi(N);
    const double serial_estimate = monte_carlo(N);

    REQUIRE(std::isfinite(mpi_estimate));
    REQUIRE(mpi_estimate == serial_estimate);
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
