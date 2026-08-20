#ifndef DERIVED_METRICS_H
#define DERIVED_METRICS_H

#include "benchmark_result.h"
void compute_achieved_metrics(BenchmarkResult& result, double flop_count, double bytes_moved);
void compute_derived_metrics(BenchmarkResult& result, const BenchmarkResult& serial_baseline, double flop_count, double bytes_moved);

#endif