#include "benchmark_context.h"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Unset measurement hooks throw exceptions") {
    BenchmarkContext ctx;

    REQUIRE_THROWS_AS(ctx.start_measurement(), std::runtime_error);
    REQUIRE_THROWS_AS(ctx.stop_measurement(), std::runtime_error);
}

TEST_CASE("Ending measurement before starting throws exception") {
    BenchmarkContext ctx;

    ctx.assign_measurement_functions(
        []() { /* start measurement */ },
        []() { /* stop measurement */ }
    );

    REQUIRE_THROWS_AS(ctx.stop_measurement(), std::runtime_error);
}

TEST_CASE("Measurement hooks execute correctly") {
    BenchmarkContext ctx;

    int start_calls = 0;
    int stop_calls = 0;

    ctx.assign_measurement_functions(
        [&]() { ++start_calls; },
        [&]() { ++stop_calls; }
    );

    ctx.start_measurement();
    REQUIRE(start_calls == 1);
    REQUIRE(stop_calls == 0);

    ctx.stop_measurement();
    REQUIRE(stop_calls == 1);
    REQUIRE(ctx.is_measurement_stopped());
}


TEST_CASE("Measurement can be restarted before stopping") {
    BenchmarkContext ctx;
    int start_calls = 0;

    ctx.assign_measurement_functions(
        [&]() { ++start_calls; },
        []() {}
    );

    ctx.start_measurement();
    ctx.start_measurement();

    REQUIRE(start_calls == 2);

    ctx.stop_measurement();
}

TEST_CASE("Stopping measurement more than once invokes hook only once") {
    BenchmarkContext ctx;
    int stop_calls = 0;

    ctx.assign_measurement_functions(
        []() {},
        [&]() { ++stop_calls; }
    );

    ctx.start_measurement();
    ctx.stop_measurement();
    ctx.stop_measurement();

    REQUIRE(stop_calls == 1);
}

TEST_CASE("Measurement cannot restart after stopping") {
    BenchmarkContext ctx;
    int start_calls = 0;

    ctx.assign_measurement_functions(
        [&]() { ++start_calls; },
        []() {}
    );

    ctx.start_measurement();
    ctx.stop_measurement();
    ctx.start_measurement();

    REQUIRE(start_calls == 1);
}
