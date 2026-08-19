#include <runner.h>
#include <catch2/catch_test_macros.hpp>
#include <unordered_map>

TEST_CASE("Error when serial benchmark missing") {
    std::unordered_map<std::string, BenchmarkResult> map;

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

    REQUIRE_THROWS_AS(run_non_serial_benchmarks(map, local_registry), std::runtime_error);
}

TEST_CASE("Error when no benchamrk registered at all") {
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

    auto serial_results = run_serial_benchmarks(local_registry);
    
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
        run_non_serial_benchmarks(serial_results, local_registry);

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
