#include <catch2/catch_test_macros.hpp>
#include <unordered_map>
#include <thread>
#include <chrono>
#include <cmath> 

#include "runner.h"
#include "benchmark_settings.h"

TEST_CASE("Error when serial benchmark missing") {
    std::unordered_map<std::string, BenchmarkResult> map;
    BenchmarkSettings settings;
    Registry local_registry;

    local_registry.register_benchmark(
        "matrix_multiply_serial",
        "matrix_multiply",
        Paradigm::OpenMP,
        [](BenchmarkContext& ctx) {
            ctx.problem_size = 100;
            ctx.iterations = 1;
            ctx.number_of_processors = 1;
            ctx.flop_count = 1000.0;
            ctx.bytes_moved = 500.0;
        }
    );

    REQUIRE_THROWS_AS(run_non_serial_benchmarks(settings, map, {}, local_registry), std::runtime_error);
}

TEST_CASE("Error when no benchmark registered at all") {
    Registry local_registry;
    

    REQUIRE_THROWS_AS(get_serial_benchmarks(local_registry), std::runtime_error);
}

TEST_CASE("Error when benchmarks exists but no Serial") {
    Registry local_registry;

    local_registry.register_benchmark(
        "matrix_multiply_serial",
        "matrix_multiply",
        Paradigm::OpenMP,
        [](BenchmarkContext& ctx) {
            ctx.problem_size = 100;
            ctx.iterations = 1;
            ctx.number_of_processors = 1;
            ctx.flop_count = 1000.0;
            ctx.bytes_moved = 500.0;
        }
    );
    
    REQUIRE_THROWS_AS(get_serial_benchmarks(local_registry), std::runtime_error);
}

TEST_CASE("End-to-end runner smoke test") {
    constexpr const char* kernel_name = "runner_smoke_test_kernel";
    Registry local_registry;
    BenchmarkSettings settings;

    local_registry.register_benchmark(
        "runner_smoke_serial",
        kernel_name,
        Paradigm::Serial,
        [](BenchmarkContext& ctx) {
            ctx.problem_size = 100;
            ctx.iterations = 10;
            ctx.number_of_processors = 1;
            ctx.flop_count = 1000.0;
            ctx.bytes_moved = 500.0;
        }
    );

    local_registry.register_benchmark(
        "runner_smoke_openmp",
        kernel_name,
        Paradigm::OpenMP,
        [](BenchmarkContext& ctx) {
            ctx.problem_size = 100;
            ctx.iterations = 10;
            ctx.number_of_processors = 4;
            ctx.flop_count = 1000.0;
            ctx.bytes_moved = 500.0;
        }
    );

    auto serial_results = run_serial_benchmarks(settings, {}, local_registry);
    
    REQUIRE(serial_results.find(kernel_name) != serial_results.end());

    const BenchmarkResult& serial_result =
        serial_results.at(kernel_name);

    REQUIRE(serial_result.name == "runner_smoke_serial");
    REQUIRE(serial_result.paradigm == Paradigm::Serial);

    REQUIRE(serial_result.problem_size == 100);
    REQUIRE(serial_result.iterations == 10);
    REQUIRE(serial_result.number_of_processors == 1);

    REQUIRE(serial_result.wall_time_ms > 0.0);
    REQUIRE(serial_result.cpu_time_ms >= 0.0);
    REQUIRE(serial_result.peak_rss_kb > 0);

    auto parallel_results =
        run_non_serial_benchmarks(settings, serial_results, {}, local_registry);

    REQUIRE(parallel_results.size() == 1);

    const BenchmarkResult& parallel_result =
        parallel_results.front();

    REQUIRE(parallel_result.name == "runner_smoke_openmp");
    REQUIRE(parallel_result.paradigm == Paradigm::OpenMP);

    REQUIRE(parallel_result.problem_size == 100);
    REQUIRE(parallel_result.iterations == 10);
    REQUIRE(parallel_result.number_of_processors == 4);

    REQUIRE(parallel_result.wall_time_ms > 0.0);
    REQUIRE(parallel_result.cpu_time_ms >= 0.0);
    REQUIRE(parallel_result.peak_rss_kb > 0);


    REQUIRE(parallel_result.speedup > 0.0);
    REQUIRE(parallel_result.efficiency > 0.0);
    REQUIRE(parallel_result.achieved_flops > 0.0);
    REQUIRE(parallel_result.arithmetic_intensity > 0.0);
}

TEST_CASE("Restarting measurement excludes setup time") {
    Registry local_registry;
    BenchmarkSettings settings;

    local_registry.register_benchmark(
        "setup_then_measure",
        "setup_then_measure",
        Paradigm::Serial,
        [](BenchmarkContext& ctx) {
            ctx.problem_size = 1;
            ctx.iterations = 1;
            ctx.number_of_processors = 1;
            ctx.bytes_moved = 100.0;

            std::this_thread::sleep_for(
                std::chrono::milliseconds(200)
            );

            ctx.start_measurement(); 

            std::this_thread::sleep_for(
                std::chrono::milliseconds(100)
            );

            ctx.stop_measurement();
        }
    );

    local_registry.register_benchmark(
        "measure_without_setup",
        "measure_without_setup",
        Paradigm::Serial,
        [](BenchmarkContext& ctx) {
            ctx.problem_size = 1;
            ctx.iterations = 1;
            ctx.number_of_processors = 1;
            ctx.bytes_moved = 100.0;

            std::this_thread::sleep_for(
                std::chrono::milliseconds(100)
            );

            ctx.stop_measurement();
        }
    );

    auto results = run_serial_benchmarks(settings, {}, local_registry);

    const double setup_time =
        results.at("setup_then_measure").wall_time_ms;

    const double normal_time =
        results.at("measure_without_setup").wall_time_ms;

    REQUIRE(setup_time < 180.0);
    REQUIRE(normal_time < 180.0);

    REQUIRE(std::abs(setup_time - normal_time) < 60.0);
}
