#ifndef CUDA_METRICS_H
#define CUDA_METRICS_H

#include "benchmark_result.h"

void collect_gpu_utilization_metrics(BenchmarkResult& result);
double read_current_gpu_memory_mb();
void ensure_nvml_initialized();

#endif