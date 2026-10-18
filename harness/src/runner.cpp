#include "registrar.h"
#include "cpu_metrics.h"
#include "scoped_timer.h"
#include "derived_metrics.h"
#include "live_monitor.h"
#include "runner.h"
#include "result_writer.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef HAS_CUDA
#include "cuda_metrics.h"
#endif


void start_measurement_helper(Timer& timer, CpuMetrics& cpu_metrics) {
    cpu_metrics.start();
    timer.start();
}


void stop_measurement_helper(Timer& timer, CpuMetrics& cpu_metrics) {
    timer.stop();
    cpu_metrics.stop();
}


namespace {

std::string make_run_name(
    const std::string& benchmark_name,
    bool is_warmup,
    int run_index,
    const BenchmarkSettings& settings
) {
    if (settings.warmup == 0 &&
        settings.repetitions == 1 &&
        !is_warmup) {
        return benchmark_name;
    }

    if (is_warmup) {
        return benchmark_name + "_warmup_" +
               std::to_string(run_index + 1);
    }

    return benchmark_name + "_run_" +
           std::to_string(run_index + 1);
}


BenchmarkResult collect_result(
    const RegisteredBenchmark& benchmark,
    BenchmarkContext& context,
    Timer& timer,
    CpuMetrics& cpu_metrics,
    const std::string& result_name
) {
    BenchmarkResult result;

    result.name = result_name;
    result.paradigm = benchmark.paradigm;

    result.number_of_processors = context.number_of_processors;
    result.problem_size = context.problem_size;
    result.iterations = context.iterations;

    result.wall_time_ms = timer.get_elapsed_time();
    result.peak_rss_kb = cpu_metrics.get_peak_rss_kb();
    result.cpu_time_ms = cpu_metrics.get_cpu_time_ms();

    compute_achieved_metrics(
        result,
        context.flop_count,
        context.bytes_moved
    );

    // Speedup and efficiency are placeholders for serial benchmarks.
    result.speedup = 1.0;
    result.efficiency = 1.0;

    return result;
}

} // namespace


std::unordered_map<std::string, RegisteredBenchmark>
get_serial_benchmarks(Registry& registry) {
    std::vector<RegisteredBenchmark>& benchmarks =
        registry.access_registered_benchmarks();

    if (benchmarks.empty()) {
        throw std::runtime_error(
            "No benchmarks registered for benchmarking"
        );
    }

    std::unordered_map<std::string, RegisteredBenchmark>
        serial_benchmarks;

    for (const auto& benchmark : benchmarks) {
        if (benchmark.paradigm == Paradigm::Serial) {
            serial_benchmarks[benchmark.kernel_name] = benchmark;
        }
    }

    if (serial_benchmarks.empty()) {
        throw std::runtime_error(
            "No serial benchmarks found in registered benchmarks"
        );
    }

    return serial_benchmarks;
}


std::unordered_map<std::string, BenchmarkResult>
run_serial_benchmarks(
    const BenchmarkSettings& settings,
    ResultSink result_sink,
    Registry& registry
) {
    const auto serial_benchmarks = get_serial_benchmarks(registry);

    std::unordered_map<std::string, BenchmarkResult> serial_results;

    for (const auto& [benchmark_name, benchmark] : serial_benchmarks) {
        for (int i = 0;
             i < settings.warmup + settings.repetitions;
             ++i) {

            const bool is_warmup_run = i < settings.warmup;

            BenchmarkContext context;
            Timer timer;
            CpuMetrics cpu_metrics;

            context.assign_measurement_functions(
                [&]() {
                    start_measurement_helper(timer, cpu_metrics);
                },
                [&]() {
                    stop_measurement_helper(timer, cpu_metrics);
                }
            );

            context.start_measurement();
            benchmark.callable(context);
            context.stop_measurement();

            const int run_index = is_warmup_run
                ? i
                : i - settings.warmup;

            const std::string result_name = make_run_name(
                benchmark.name,
                is_warmup_run,
                run_index,
                settings
            );

            BenchmarkResult result = collect_result(
                benchmark,
                context,
                timer,
                cpu_metrics,
                result_name
            );


            if (result_sink) {
                result_sink(result);
            }

            if (!is_warmup_run) {
                serial_results[benchmark_name] = result;
            }
        }
    }

    return serial_results;
}


std::vector<BenchmarkResult> run_non_serial_benchmarks(
    const BenchmarkSettings& settings,
    const std::unordered_map<std::string, BenchmarkResult>& serial_benchmarks,
    ResultSink result_sink,
    Registry& registry
) {
    std::vector<RegisteredBenchmark>& all_benchmarks =
        registry.access_registered_benchmarks();

    std::vector<BenchmarkResult> non_serial_benchmark_results;

    for (const auto& benchmark : all_benchmarks) {
        if (benchmark.paradigm == Paradigm::Serial) {
            continue;
        }

        if (serial_benchmarks.find(benchmark.kernel_name) ==
            serial_benchmarks.end()) {
            throw std::runtime_error(
                "No serial benchmark found for " +
                benchmark.kernel_name
            );
        }

        const BenchmarkResult& serial_benchmark =
            serial_benchmarks.at(benchmark.kernel_name);

#ifndef HAS_CUDA
        if (benchmark.paradigm == Paradigm::CUDA) {
            throw std::runtime_error(
                "CUDA benchmark requested, but CUDA support is disabled"
            );
        }
#endif

        for (int i = 0;
             i < settings.warmup + settings.repetitions;
             ++i) {

            const bool is_warmup_run = i < settings.warmup;

            BenchmarkContext context;
            Timer timer;
            CpuMetrics cpu_metrics;

#ifdef HAS_CUDA
            LiveMonitor monitor;

            if (benchmark.paradigm == Paradigm::CUDA) {
                monitor.start();
            }
#endif

            context.assign_measurement_functions(
                [&]() {
                    start_measurement_helper(timer, cpu_metrics);
                },
                [&]() {
                    stop_measurement_helper(timer, cpu_metrics);

#ifdef HAS_CUDA
                    if (benchmark.paradigm == Paradigm::CUDA) {
                        monitor.stop();
                    }
#endif
                }
            );

            context.start_measurement();
            benchmark.callable(context);
            context.stop_measurement();

            const int run_index = is_warmup_run
                ? i
                : i - settings.warmup;

            const std::string result_name = make_run_name(
                benchmark.name,
                is_warmup_run,
                run_index,
                settings
            );

            BenchmarkResult result = collect_result(
                benchmark,
                context,
                timer,
                cpu_metrics,
                result_name
            );

#ifdef HAS_CUDA
            if (benchmark.paradigm == Paradigm::CUDA) {
                monitor.populate(result);
                collect_gpu_utilization_metrics(result);
            }
#endif

            
            compute_derived_metrics(
                result,
                serial_benchmark,
                context.flop_count,
                context.bytes_moved
            );

            non_serial_benchmark_results.push_back(result);

            if (result_sink) {
                result_sink(result);
            }
        }
    }

    return non_serial_benchmark_results;
}