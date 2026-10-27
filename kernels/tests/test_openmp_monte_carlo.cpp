#include <catch2/catch_test_macros.hpp>

#include "openmp_monte_carlo.h"
#include "openmp_helper.h"
#include "monte_carlo.h"
#include "seed_values.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include <omp.h>

namespace {

// Sweep counts from the benchmark plus awkward ones (3, 7).
const std::vector<int> THREAD_COUNTS = {1, 2, 3, 4, 7, 8, 12, 16, 20};

constexpr double PI = 3.14159265358979323846;

// Number of samples inside the unit quarter circle drawn by one thread: a
// plain re-implementation of the serial loop on its own generator.
int hits_for_thread(unsigned seed, int samples) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> generator(0.0, 1.0);

    int hits = 0;

    for (int i = 0; i < samples; ++i) {
        double x = generator(gen);
        double y = generator(gen);

        if (x * x + y * y <= 1.0) {
            ++hits;
        }
    }

    return hits;
}

} // namespace

// The seeds are fixed, so each assertion below is deterministic for a given
// (N, thread count); the bounds were checked for 1..64 threads.

TEST_CASE("OpenMP Monte Carlo pi calculations within reasonable error bounds") {
    for (int t : THREAD_COUNTS) {
        CAPTURE(t);

        double pi = openmp_monte_carlo(10000, t);
        REQUIRE(pi > 3.1);
        REQUIRE(pi < 3.2);
    }
}

TEST_CASE("OpenMP Monte Carlo is accurate and agrees with serial for a large N") {
    const int N = 1'000'000;
    const double serial_pi = monte_carlo(N);

    for (int t : THREAD_COUNTS) {
        CAPTURE(t);

        double pi = openmp_monte_carlo(N, t);

        REQUIRE(std::abs(pi - PI) < 0.01);
        REQUIRE(std::abs(pi - serial_pi) < 0.01);
    }
}

TEST_CASE("OpenMP Monte Carlo with one thread reproduces the serial result exactly") {
    // Thread 0 uses the serial seed and generates the same stream in the same
    // order, so with a single thread the two estimators are identical.
    for (int N : {1, 2, 3, 100, 10'007, 10'000, 123'457}) {
        CAPTURE(N);
        REQUIRE(openmp_monte_carlo(N, 1) == monte_carlo(N));
    }
}

TEST_CASE("OpenMP Monte Carlo matches an independent per-thread reference") {
    // Documented contract: thread k draws from mt19937(MONTE_CARLO_SEED + k).
    // When N is divisible by the number of threads, every thread gets exactly
    // N / threads samples, so the expected hit count can be rebuilt here from
    // scratch. This pins the seeding scheme: identical seeds on all threads
    // (correlated streams) or a missing/duplicated chunk would change the count.
    // 10080 is divisible by every thread count in THREAD_COUNTS.
    const int N = 10'080;

    for (int t : THREAD_COUNTS) {
        CAPTURE(t);

        const int threads = get_number_of_threads(t);

        if (N % threads != 0) {
            continue;
        }

        int expected_hits = 0;
        for (int k = 0; k < threads; ++k) {
            expected_hits += hits_for_thread(
                MONTE_CARLO_SEED + static_cast<unsigned>(k), N / threads);
        }

        const double expected_pi = static_cast<double>(expected_hits) / N * 4.0;

        REQUIRE(openmp_monte_carlo(N, t) == expected_pi);
    }
}

TEST_CASE("OpenMP Monte Carlo is reproducible for a given N and thread count") {
    for (int t : THREAD_COUNTS) {
        CAPTURE(t);
        REQUIRE(openmp_monte_carlo(50'000, t) == openmp_monte_carlo(50'000, t));
    }
}

TEST_CASE("OpenMP Monte Carlo handles N not divisible by the thread count") {
    // Prime N: the static chunks have unequal sizes for every thread count > 1.
    for (int t : THREAD_COUNTS) {
        CAPTURE(t);

        double pi = openmp_monte_carlo(10'007, t);

        REQUIRE(pi > 3.1);
        REQUIRE(pi < 3.2);
    }
}

TEST_CASE("OpenMP Monte Carlo handles fewer samples than threads") {
    for (int N : {1, 2, 3, 5}) {
        for (int t : THREAD_COUNTS) {
            CAPTURE(N, t);

            double pi = openmp_monte_carlo(N, t);

            // pi = 4 * hits / N with 0 <= hits <= N.
            REQUIRE(pi >= 0.0);
            REQUIRE(pi <= 4.0);
            REQUIRE(std::abs(pi / 4.0 * N - std::round(pi / 4.0 * N)) < 1e-9);
        }
    }
}

TEST_CASE("OpenMP Monte Carlo rejects non-positive N") {
    REQUIRE_THROWS_AS(openmp_monte_carlo(0, 4), std::invalid_argument);
    REQUIRE_THROWS_AS(openmp_monte_carlo(-100, 4), std::invalid_argument);
}

TEST_CASE("OpenMP Monte Carlo rejects non-positive thread counts") {
    REQUIRE_THROWS_AS(openmp_monte_carlo(10000, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(openmp_monte_carlo(10000, -2), std::invalid_argument);
}

TEST_CASE("OpenMP Monte Carlo leaves the global thread count unchanged") {
    const int original = omp_get_max_threads();
    const int requested = original == 3 ? 5 : 3;

    openmp_monte_carlo(1000, requested);

    REQUIRE(omp_get_max_threads() == original);
}
