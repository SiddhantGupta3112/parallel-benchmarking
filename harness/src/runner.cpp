#include "registrar.h"
#include "stdexcept"
#include <unordered_map>
#include "cpu_metrics.h"
#include "cuda_metrics.h"
#include "scoped_timer.h"
#include "cuda_metrics.h"
#include "derived_metrics.h"
#include "live_monitor.h"
#include "runner.h"

std::unordered_map<std::string, RegisteredBenchmark> get_serial_benchmarks(Registry& registry){
    std::vector<RegisteredBenchmark>& benchmarks = registry.access_registered_benchmarks();

    if(benchmarks.empty()){
        throw std::runtime_error("No benchmarks registed for benchmarking");
    }

    std::unordered_map<std::string, RegisteredBenchmark> serial_benchmarks;

    for(const auto& b: benchmarks){
        if(b.paradigm == Paradigm::Serial){
            serial_benchmarks[b.kernel_name] = b;
        }
    }

    if(serial_benchmarks.empty()){
        throw std::runtime_error("No serial benchmarks found in registered benchmarks");
    }

    return serial_benchmarks;
}

std::unordered_map<std::string, BenchmarkResult> run_serial_benchmarks(Registry& registry) {
    std::unordered_map<std::string, RegisteredBenchmark> serial_benchmarks =
        get_serial_benchmarks(registry);

    std::unordered_map<std::string, BenchmarkResult> serial_results;

    for (const auto& [benchmark_name, benchmark] : serial_benchmarks) {
        BenchmarkContext context;

        Timer timer;
        CpuMetrics cpu_metrics;

        cpu_metrics.start();
        timer.start();

        benchmark.callable(context);

        timer.stop();
        cpu_metrics.stop();

        BenchmarkResult result;
        result.name = benchmark.name;
        result.paradigm = benchmark.paradigm;
        result.number_of_processors = context.number_of_processors;
        result.problem_size = context.problem_size;
        result.iterations = context.iterations;
        
        result.wall_time_ms = timer.get_elapsed_time();
        result.peak_rss_kb = cpu_metrics.get_peak_rss_kb();
        result.cpu_time_ms = cpu_metrics.get_cpu_time_ms();

        serial_results[benchmark_name] = result;
    }

    return serial_results;
}


std::vector<BenchmarkResult> run_non_serial_benchmarks(std::unordered_map<std::string, BenchmarkResult> serial_benchmarks, Registry& registry){
    std::vector<RegisteredBenchmark>& all_benchmarks = registry.access_registered_benchmarks();

    std::vector<BenchmarkResult> non_serial_benchmark_results;

    for (const auto& benchmark : all_benchmarks) {
        if(benchmark.paradigm == Paradigm::Serial){
            continue;
        }

        if(serial_benchmarks.find(benchmark.kernel_name) == serial_benchmarks.end()){
            throw std::runtime_error("No serial benchamrk found for " + benchmark.kernel_name);
        }

        const BenchmarkResult& serial_benchmark = serial_benchmarks[benchmark.kernel_name];

        BenchmarkContext context;

        Timer timer;
        CpuMetrics cpu_metrics;
        LiveMonitor monitor;

        if(benchmark.paradigm == Paradigm::CUDA){
            monitor.start();
        }
        cpu_metrics.start();
        timer.start();

        benchmark.callable(context);

        timer.stop();
        cpu_metrics.stop();
        if(benchmark.paradigm == Paradigm::CUDA){
            monitor.stop();
        }

        BenchmarkResult result;
        result.name = benchmark.name;
        result.paradigm = benchmark.paradigm;
        result.number_of_processors = context.number_of_processors;
        result.problem_size = context.problem_size;
        result.iterations = context.iterations;
        
        result.wall_time_ms = timer.get_elapsed_time();
        result.peak_rss_kb = cpu_metrics.get_peak_rss_kb();
        result.cpu_time_ms = cpu_metrics.get_cpu_time_ms();

        compute_derived_metrics(result, serial_benchmark, context.flop_count, context.bytes_moved);

        if(benchmark.paradigm == Paradigm::CUDA){
            monitor.populate(result);
            collect_gpu_utilization_metrics(result);
        }

        non_serial_benchmark_results.push_back(result);
    }

    return non_serial_benchmark_results;
}
