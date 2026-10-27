#ifndef BASELINE_LOADER_H
#define BASELINE_LOADER_H

#include "benchmark_result.h"

#include <string>
#include <unordered_map>

using BaselineMap =
    std::unordered_map<std::string, BenchmarkResult>;

BaselineMap load_serial_baselines(const std::string& filename);

#endif
