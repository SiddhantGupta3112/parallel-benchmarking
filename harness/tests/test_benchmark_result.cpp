#include "benchmark_result.h"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("BenchmarkResult compulsory fields can be populated") {
    BenchmarkResult result;

    result.name = "OpenMP test";
    result.paradigm = Paradigm::OpenMP;
    result.problem_size = 10000;
    result.iterations = 100;

    result.wall_time_ms = 25.5;
    result.speedup = 3.2;
    result.efficiency = 0.8;
    result.achieved_flops = 1.5e9;
    result.arithmetic_intensity = 4.2;
    result.cpu_time_ms = 30.0;
    result.peak_rss_kb = 10240;

    REQUIRE(result.name == "OpenMP test");
    REQUIRE(result.paradigm == Paradigm::OpenMP);
    REQUIRE(result.problem_size == 10000);
    REQUIRE(result.iterations == 100);

    REQUIRE(result.wall_time_ms == 25.5);
    REQUIRE(result.speedup == 3.2);
    REQUIRE(result.efficiency == 0.8);
    REQUIRE(result.achieved_flops == 1.5e9);
    REQUIRE(result.arithmetic_intensity == 4.2);
    REQUIRE(result.cpu_time_ms == 30.0);
    REQUIRE(result.peak_rss_kb == 10240);
}

TEST_CASE("Optional GPU metrics are empty by default") {
    BenchmarkResult result;

    REQUIRE_FALSE(result.gpu_utilization_pct.has_value());
    REQUIRE_FALSE(result.gpu_memory_used_mb.has_value());
    REQUIRE_FALSE(result.gpu_memory_throughput_gbps.has_value());
    REQUIRE_FALSE(result.mpi_comm_overhead_fraction.has_value());
}

TEST_CASE("Optional metrics can contain values") {
    BenchmarkResult result;

    result.gpu_utilization_pct = 85.5;
    result.gpu_memory_used_mb = 4096.0;
    result.gpu_memory_throughput_gbps = 750.0;
    result.mpi_comm_overhead_fraction = 0.15;

    REQUIRE(result.gpu_utilization_pct.has_value());
    REQUIRE(result.gpu_memory_used_mb.has_value());
    REQUIRE(result.gpu_memory_throughput_gbps.has_value());
    REQUIRE(result.mpi_comm_overhead_fraction.has_value());

    REQUIRE(result.gpu_utilization_pct.value() == 85.5);
    REQUIRE(result.gpu_memory_used_mb.value() == 4096.0);
    REQUIRE(result.gpu_memory_throughput_gbps.value() == 750.0);
    REQUIRE(result.mpi_comm_overhead_fraction.value() == 0.15);
}