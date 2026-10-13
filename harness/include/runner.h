#ifndef RUNNER_H
#define RUNNER_H

#include <unordered_map>
#include <vector>
#include <fstream>
#include <functional>
#include "benchmark_result.h"

#include "benchmark_result.h"
#include "registrar.h"


using ResultSink = std::function<void(const BenchmarkResult&)>;

std::unordered_map<std::string, RegisteredBenchmark> get_serial_benchmarks(Registry& registry = get_registry());

std::unordered_map<std::string, BenchmarkResult>
run_serial_benchmarks(
    ResultSink result_sink = {},
    Registry& registry = get_registry()
);

std::vector<BenchmarkResult>
run_non_serial_benchmarks(
    const std::unordered_map<std::string, BenchmarkResult>& serial_results,
    ResultSink result_sink = {},
    Registry& registry = get_registry()
);

#endif