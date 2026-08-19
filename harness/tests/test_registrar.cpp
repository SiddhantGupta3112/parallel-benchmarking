#include "registrar.h"
#include <catch2/catch_test_macros.hpp>

static bool ran_a = false;
BENCHMARK_TEST(dummy_openmp, dummy_kernel, Paradigm::OpenMP) {
    ran_a = true;
}

TEST_CASE("Registered benchmark appears in registry with correct metadata") {
    auto& benchmarks = get_registry().access_registered_benchmarks();

    REQUIRE(!benchmarks.empty());

    bool found = false;
    RegisteredBenchmark benchmark;

    for (const auto& b : benchmarks) {
        if (b.name == "dummy_openmp") {
            found = true;
            benchmark = b;
            break;
        }
    }

    REQUIRE(found);
    REQUIRE(benchmark.name == "dummy_openmp");
    REQUIRE(benchmark.kernel_name == "dummy_kernel");
    REQUIRE(benchmark.paradigm == Paradigm::OpenMP);
}

TEST_CASE("Registered benchmark callable executes correctly") {
    auto& benchmarks = get_registry().access_registered_benchmarks();

    BenchmarkContext ctx;
    ctx.bytes_moved = 1000;
    ctx.flop_count = 1000;
    for (const auto& benchmark : benchmarks) {
        if (benchmark.name == "dummy_openmp") {
            ran_a = false;

            benchmark.callable(ctx);

            REQUIRE(ran_a);
            return;
        }
    }

    FAIL("dummy_openmp was not found in the registry");
}