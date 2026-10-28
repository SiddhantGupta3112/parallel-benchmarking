#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <stdexcept>

#include <omp.h>

#include "openmp_helper.h"

TEST_CASE("OpenMPThreadScope sets the thread count and restores the previous one") {
    const int original = omp_get_max_threads();
    const int other = original == 3 ? 5 : 3;

    {
        OpenMPThreadScope scope(other);
        REQUIRE(omp_get_max_threads() == other);
    }

    REQUIRE(omp_get_max_threads() == original);
}

TEST_CASE("OpenMPThreadScope rejects non-positive thread counts") {
    const int original = omp_get_max_threads();

    REQUIRE_THROWS_AS(OpenMPThreadScope(0), std::invalid_argument);
    REQUIRE_THROWS_AS(OpenMPThreadScope(-4), std::invalid_argument);

    REQUIRE(omp_get_max_threads() == original);
}

TEST_CASE("get_number_of_threads reports the threads a parallel region really gets") {
    const int original = omp_get_max_threads();

    for (int t : {1, 2, 3, 4, 7, 8, 12, 16, 20}) {
        CAPTURE(t);

        const int actual = get_number_of_threads(t);

        if (omp_get_dynamic()) {
            REQUIRE(actual >= 1);
            REQUIRE(actual <= t);
        } else {
            REQUIRE(actual == std::min(t, omp_get_thread_limit()));
        }
    }

    REQUIRE(omp_get_max_threads() == original);
}

TEST_CASE("get_number_of_threads rejects non-positive thread counts") {
    REQUIRE_THROWS_AS(get_number_of_threads(0), std::invalid_argument);
    REQUIRE_THROWS_AS(get_number_of_threads(-1), std::invalid_argument);
}
