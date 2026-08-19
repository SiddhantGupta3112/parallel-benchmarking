#ifndef RUNNER_H
#define RUNNER_H

#include <unordered_map>
#include <vector>
#include "benchmark_result.h"
#include "registrar.h"

std::unordered_map<std::string, RegisteredBenchmark> get_serial_benchmarks(Registry& registry = get_registry());
std::unordered_map<std::string, BenchmarkResult> run_serial_benchmarks(Registry& registry = get_registry());
std::vector<BenchmarkResult> run_non_serial_benchmarks(
    std::unordered_map<std::string, BenchmarkResult> serial_results,
    Registry& registry = get_registry()
);

#endif