#include "derived_metrics.h"
#include "benchmark_result.h"
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>

TEST_CASE("Derived metrics compute correctly") {
    BenchmarkResult serial;

    serial.name = "Test_Metrics";
    serial.paradigm = Paradigm::Serial;
    serial.problem_size = 1000;
    serial.iterations = 1000;
    serial.wall_time_ms = 1000;
    serial.number_of_processors = 1;

    BenchmarkResult parallel;

    parallel.name = "Test_Metrics";
    parallel.paradigm = Paradigm::OpenMP;
    parallel.problem_size = 1000;
    parallel.iterations = 1000;
    parallel.wall_time_ms = 250;
    parallel.number_of_processors = 4;

    compute_derived_metrics(
        parallel,
        serial,
        1000000,
        500000
    );

    REQUIRE(parallel.speedup == 4.0);
    REQUIRE(parallel.efficiency == 1.0);
}

TEST_CASE("Achieved FLOPS and arithmetic intensity compute correctly") {
    BenchmarkResult result;

    result.name = "Test_Metrics";
    result.paradigm = Paradigm::OpenMP;
    result.problem_size = 1000;
    result.iterations = 1000;
    result.wall_time_ms = 250;
    result.number_of_processors = 4;

    double flop_count = 1000000;
    double bytes_moved = 500000;

    compute_achieved_metrics(
        result,
        flop_count,
        bytes_moved
    );

    REQUIRE(result.achieved_flops == 4000000.0);
    REQUIRE(result.arithmetic_intensity == 2.0);
}

TEST_CASE("Error on invalid wall-clock time") {
    BenchmarkResult serial;

    serial.name = "Test_Metrics";
    serial.paradigm = Paradigm::Serial;
    serial.problem_size = 1000;
    serial.iterations = 1000;
    serial.wall_time_ms = 1000;
    serial.number_of_processors = 1;

    BenchmarkResult parallel;

    parallel.name = "Test_Metrics";
    parallel.paradigm = Paradigm::OpenMP;
    parallel.problem_size = 1000;
    parallel.iterations = 1000;
    parallel.wall_time_ms = 0;
    parallel.number_of_processors = 4;

    REQUIRE_THROWS_AS(
        compute_derived_metrics(
            parallel,
            serial,
            1000000,
            500000
        ),
        std::runtime_error
    );
}

TEST_CASE("Error when baseline and result configurations mismatch") {
    BenchmarkResult serial;

    serial.problem_size = 1000;
    serial.iterations = 1000;
    serial.wall_time_ms = 1000;
    serial.number_of_processors = 1;

    BenchmarkResult parallel;

    parallel.problem_size = 2000;  // mismatch
    parallel.iterations = 1000;
    parallel.wall_time_ms = 250;
    parallel.number_of_processors = 4;

    REQUIRE_THROWS_AS(
        compute_derived_metrics(
            parallel,
            serial,
            1000000,
            500000
        ),
        std::runtime_error
    );
}