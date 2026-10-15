#include "registrar.h"
#include <stdexcept>
#include <unordered_map>
#include "cpu_metrics.h"
#include "scoped_timer.h"
#include "derived_metrics.h"
#include "live_monitor.h"
#include "runner.h"
#include "result_writer.h"

#ifdef HAS_CUDA
#include "cuda_metrics.h"
#endif

void start_measurement_helper(Timer& timer, CpuMetrics& cpu_metrics){
    cpu_metrics.start();
    timer.start();
}

void stop_measurement_helper(Timer& timer, CpuMetrics& cpu_metrics){
    timer.stop();
    cpu_metrics.stop();
}

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

std::unordered_map<std::string, BenchmarkResult> run_serial_benchmarks(
    ResultSink result_sink,
    Registry& registry
){ 
    std::unordered_map<std::string, RegisteredBenchmark> serial_benchmarks =
        get_serial_benchmarks(registry);

    std::unordered_map<std::string, BenchmarkResult> serial_results;

    for (const auto& [benchmark_name, benchmark] : serial_benchmarks) {
        BenchmarkContext context;

        Timer timer;
        CpuMetrics cpu_metrics;

        context.assign_measurement_functions(
            [&]() { start_measurement_helper(timer, cpu_metrics); },
            [&]() { stop_measurement_helper(timer, cpu_metrics); }
        );

        context.start_measurement();
        

        benchmark.callable(context);

        
        context.stop_measurement();
        

        BenchmarkResult result;
        result.name = benchmark.name;
        result.paradigm = benchmark.paradigm;
        result.number_of_processors = context.number_of_processors;
        result.problem_size = context.problem_size;
        result.iterations = context.iterations;
        
        result.wall_time_ms = timer.get_elapsed_time();
        result.peak_rss_kb = cpu_metrics.get_peak_rss_kb();
        result.cpu_time_ms = cpu_metrics.get_cpu_time_ms();

        compute_achieved_metrics(result, context.flop_count, context.bytes_moved);
        
        result.speedup = 1.0;
        result.efficiency = 1.0;

        serial_results[benchmark_name] = result;

        if (result_sink) {
            result_sink(result);
        }
            
    }

    return serial_results;
}


std::vector<BenchmarkResult> run_non_serial_benchmarks(
    const std::unordered_map<std::string, BenchmarkResult>& serial_benchmarks,
    ResultSink result_sink,
    Registry& registry
){
    std::vector<RegisteredBenchmark>& all_benchmarks = registry.access_registered_benchmarks();

    std::vector<BenchmarkResult> non_serial_benchmark_results;

    for (const auto& benchmark : all_benchmarks) {
        if(benchmark.paradigm == Paradigm::Serial){
            continue;
        }

        if(serial_benchmarks.find(benchmark.kernel_name) == serial_benchmarks.end()){
            throw std::runtime_error("No serial benchamrk found for " + benchmark.kernel_name);
        }

        const BenchmarkResult& serial_benchmark = serial_benchmarks.at(benchmark.kernel_name);

        BenchmarkContext context;

        Timer timer;
        CpuMetrics cpu_metrics;
        

        #ifdef HAS_CUDA
            LiveMonitor monitor;
            if (benchmark.paradigm == Paradigm::CUDA) {
                monitor.start();
            }
            #else
            if (benchmark.paradigm == Paradigm::CUDA) {
                throw std::runtime_error(
                    "CUDA benchmark requested, but CUDA support is disabled"
                );
            }
        #endif
    

        context.assign_measurement_functions(
            [&]() {
                start_measurement_helper(timer, cpu_metrics);
            },
            [&]() {
                stop_measurement_helper(timer, cpu_metrics);
                #ifdef HAS_CUDA
                    if (!context.is_measurement_stopped() &&
                        benchmark.paradigm == Paradigm::CUDA) {
                        monitor.stop();
                    }
                #endif
                
            }
        );

        context.start_measurement();

        benchmark.callable(context);

        context.stop_measurement();

        BenchmarkResult result;
        #ifdef HAS_CUDA
        if (benchmark.paradigm == Paradigm::CUDA) {
            monitor.populate(result);
            collect_gpu_utilization_metrics(result);
        }
        #endif
        
        
        result.name = benchmark.name;
        result.paradigm = benchmark.paradigm;
        result.number_of_processors = context.number_of_processors;
        result.problem_size = context.problem_size;
        result.iterations = context.iterations;
        
        result.wall_time_ms = timer.get_elapsed_time();
        result.peak_rss_kb = cpu_metrics.get_peak_rss_kb();
        result.cpu_time_ms = cpu_metrics.get_cpu_time_ms();

        compute_achieved_metrics(result, context.flop_count, context.bytes_moved);
        compute_derived_metrics(result, serial_benchmark, context.flop_count, context.bytes_moved);
        
        non_serial_benchmark_results.push_back(result);
        
        if(result_sink){
            result_sink(result);
        }
    }

    return non_serial_benchmark_results;
}
