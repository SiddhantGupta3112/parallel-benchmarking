#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>

#include "benchmark_context.h"
#include "benchmark_settings.h"
#include "cpu_metrics.h"
#include "derived_metrics.h"
#include "metadata_collector.h"
#include "registrar.h"
#include "result_writer.h"
#include "runner.h"
#include "scoped_timer.h"

#ifndef PB_SOURCE_DIR
#define PB_SOURCE_DIR "."
#endif

#ifndef PB_BUILD_DIR
#define PB_BUILD_DIR "build"
#endif

namespace {

constexpr int MAX_REPETITIONS = 1000;
constexpr int MAX_WARMUP = 1000;

struct Options {
    int repetitions = BenchmarkSettings::DEFAULT_REPETITIONS;
    int warmup = BenchmarkSettings::DEFAULT_WARMUP;
    std::string baseline;
};

int parse_integer(const std::string& input) {
    std::size_t position = 0;
    int value = std::stoi(input, &position);

    if (position != input.size()) {
        throw std::invalid_argument(
            "Expected an integer, got: " + input
        );
    }

    return value;
}

int parse_arguments(
    int argc,
    char* argv[],
    Options& options,
    int rank
) {
    try {
        for (int i = 1; i < argc; ++i) {
            std::string argument = argv[i];

            if (argument == "--help") {
                if (rank == 0) {
                    std::cout
                        << "Usage: " << argv[0]
                        << " --baseline <serial.csv> [options]\n\n"
                        << "Options:\n"
                        << "  --baseline <file>   Serial baseline CSV\n"
                        << "  --repetitions <n>   Measured repetitions (1-1000)\n"
                        << "  --warmup <n>        Warm-up repetitions (0-1000)\n"
                        << "  --help              Display this help\n";
                }

                return 2;
            }

            if (argument != "--baseline" &&
                argument != "--repetitions" &&
                argument != "--warmup") {
                throw std::invalid_argument(
                    "Unknown argument: " + argument
                );
            }

            if (i + 1 >= argc) {
                throw std::invalid_argument(
                    "Missing value for " + argument
                );
            }

            std::string value = argv[++i];

            if (argument == "--baseline") {
                options.baseline = value;
            } else if (argument == "--repetitions") {
                options.repetitions = parse_integer(value);
            } else {
                options.warmup = parse_integer(value);
            }
        }

        if (options.baseline.empty()) {
            throw std::invalid_argument(
                "--baseline is required"
            );
        }

        if (options.repetitions < 1 ||
            options.repetitions > MAX_REPETITIONS) {
            throw std::invalid_argument(
                "--repetitions must be between 1 and 1000"
            );
        }

        if (options.warmup < 0 ||
            options.warmup > MAX_WARMUP) {
            throw std::invalid_argument(
                "--warmup must be between 0 and 1000"
            );
        }

        return 0;

    } catch (const std::exception& error) {
        if (rank == 0) {
            std::cerr << "Argument error: "
                      << error.what() << '\n';
        }

        return 1;
    }
}

std::vector<std::string> parse_csv_line(
    const std::string& line
) {
    std::vector<std::string> fields;
    std::string field;
    bool inside_quotes = false;

    for (std::size_t i = 0; i < line.size(); ++i) {
        char character = line[i];

        if (character == '"') {
            if (inside_quotes &&
                i + 1 < line.size() &&
                line[i + 1] == '"') {
                field += '"';
                ++i;
            } else {
                inside_quotes = !inside_quotes;
            }
        } else if (character == ',' && !inside_quotes) {
            fields.push_back(field);
            field.clear();
        } else {
            field += character;
        }
    }

    if (inside_quotes) {
        throw std::runtime_error(
            "Malformed CSV: unterminated quoted field"
        );
    }

    fields.push_back(field);

    return fields;
}

const std::string& csv_field(
    const std::vector<std::string>& fields,
    const std::unordered_map<std::string, std::size_t>& columns,
    const std::string& name
) {
    auto column = columns.find(name);

    if (column == columns.end()) {
        throw std::runtime_error(
            "Missing CSV column: " + name
        );
    }

    if (column->second >= fields.size()) {
        throw std::runtime_error(
            "CSV row is missing column: " + name
        );
    }

    return fields[column->second];
}

double parse_double(const std::string& value) {
    std::size_t position = 0;
    double result = std::stod(value, &position);

    if (position != value.size() || !std::isfinite(result)) {
        throw std::runtime_error(
            "Invalid numeric CSV value: " + value
        );
    }

    return result;
}

int parse_csv_integer(const std::string& value) {
    int result = parse_integer(value);
    return result;
}

long parse_csv_long(const std::string& value) {
    std::size_t position = 0;
    long result = std::stol(value, &position);

    if (position != value.size()) {
        throw std::runtime_error(
            "Invalid integer CSV value: " + value
        );
    }

    return result;
}

using BaselineMap =
    std::unordered_map<std::string, BenchmarkResult>;

BaselineMap load_serial_baselines(
    const std::string& filename
) {
    std::ifstream file(filename);

    if (!file) {
        throw std::runtime_error(
            "Cannot open baseline CSV: " + filename
        );
    }

    std::string line;

    if (!std::getline(file, line)) {
        throw std::runtime_error(
            "Baseline CSV is empty"
        );
    }

    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }

    const auto header = parse_csv_line(line);

    std::unordered_map<std::string, std::size_t> columns;

    for (std::size_t i = 0; i < header.size(); ++i) {
        columns[header[i]] = i;
    }

    using BaselineSamples =
        std::unordered_map<std::string, std::vector<BenchmarkResult>>;

    BaselineSamples samples;

    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (line.empty()) {
            continue;
        }

        const auto fields = parse_csv_line(line);

        const std::string& paradigm =
            csv_field(fields, columns, "paradigm");

        const std::string& run_type =
            csv_field(fields, columns, "run_type");

        if (paradigm != "Serial" || run_type != "core") {
            continue;
        }

        BenchmarkResult result;

        result.name = csv_field(fields, columns, "name");
        result.kernel_name =
            csv_field(fields, columns, "kernel_name");

        result.paradigm = Paradigm::Serial;
        result.run_type = RunType::Core;

        result.rep_index = parse_csv_integer(
            csv_field(fields, columns, "rep_index")
        );

        result.problem_size = parse_csv_integer(
            csv_field(fields, columns, "problem_size")
        );

        result.iterations = parse_csv_integer(
            csv_field(fields, columns, "iterations")
        );

        result.number_of_processors = parse_csv_integer(
            csv_field(fields, columns, "number_of_processors")
        );

        result.wall_time_ms = parse_double(
            csv_field(fields, columns, "wall_time_ms")
        );

        result.cpu_time_ms = parse_double(
            csv_field(fields, columns, "cpu_time_ms")
        );

        result.peak_rss_kb = parse_csv_long(
            csv_field(fields, columns, "peak_rss_kb")
        );

        result.flop_count = parse_double(
            csv_field(fields, columns, "flop_count")
        );

        result.bytes_moved = parse_double(
            csv_field(fields, columns, "bytes_moved")
        );

        if (result.wall_time_ms <= 0.0 ||
            result.problem_size <= 0 ||
            result.number_of_processors <= 0) {
            throw std::runtime_error(
                "Invalid Serial baseline for " +
                result.kernel_name
            );
        }

        samples[result.kernel_name].push_back(result);
    }

    if (samples.empty()) {
        throw std::runtime_error(
            "No Serial core measurements found in baseline CSV"
        );
    }

    BaselineMap baselines;

    for (auto& entry : samples) {
        const std::string& kernel_name = entry.first;
        auto& repetitions = entry.second;

        std::vector<double> times;
        times.reserve(repetitions.size());

        const BenchmarkResult& first = repetitions.front();

        for (const auto& repetition : repetitions) {
            if (repetition.problem_size != first.problem_size ||
                repetition.iterations != first.iterations) {
                throw std::runtime_error(
                    "Inconsistent Serial configurations for " +
                    kernel_name
                );
            }

            times.push_back(repetition.wall_time_ms);
        }

        std::sort(times.begin(), times.end());

        const std::size_t middle = times.size() / 2;
        double median;

        if (times.size() % 2 == 0) {
            median = (
                times[middle - 1] + times[middle]
            ) / 2.0;
        } else {
            median = times[middle];
        }

        BenchmarkResult baseline = first;
        baseline.wall_time_ms = median;
        baseline.speedup = 1.0;
        baseline.efficiency = 1.0;

        compute_achieved_metrics(
            baseline,
            baseline.flop_count,
            baseline.bytes_moved
        );

        baselines.emplace(kernel_name, baseline);
    }

    return baselines;
}

std::string generate_file_name() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t raw_time =
        std::chrono::system_clock::to_time_t(now);

    std::tm local_time{};

    if (localtime_r(&raw_time, &local_time) == nullptr) {
        throw std::runtime_error(
            "Could not determine local time"
        );
    }

    std::ostringstream timestamp;

    timestamp
        << std::put_time(&local_time, "%Y-%m-%d_%H-%M-%S")
        << "_pid" << getpid();

    return "results/" + timestamp.str() + ".csv";
}

std::string join_arguments(
    int argc,
    char* argv[]
) {
    std::string command;

    for (int i = 0; i < argc; ++i) {
        if (i > 0) {
            command += ' ';
        }

        command += argv[i];
    }

    return command;
}

int registered_rank_count(
    const std::string& name
) {
    const std::size_t marker = name.rfind("_np");

    if (marker == std::string::npos) {
        return -1;
    }

    const std::string suffix =
        name.substr(marker + 3);

    if (suffix.empty()) {
        throw std::runtime_error(
            "Invalid MPI rank suffix in benchmark: " + name
        );
    }

    const int ranks = parse_integer(suffix);

    if (ranks <= 0) {
        throw std::runtime_error(
            "MPI rank count must be positive in benchmark: " +
            name
        );
    }

    return ranks;
}

void start_mpi_measurement(
    Timer& timer,
    CpuMetrics& cpu_metrics
) {
    MPI_Barrier(MPI_COMM_WORLD);

    cpu_metrics.start();
    timer.start();
}

void stop_mpi_measurement(
    Timer& timer,
    CpuMetrics& cpu_metrics
) {
    timer.stop();
    cpu_metrics.stop();
}

void run_mpi_benchmarks(
    const BenchmarkSettings& settings,
    const BaselineMap& baselines,
    ResultSink result_sink,
    int rank,
    int world_size
) {
    const auto& registered =
        get_registry().access_registered_benchmarks();

    std::vector<RegisteredBenchmark> active;

    for (const auto& benchmark : registered) {
        if (benchmark.paradigm != Paradigm::MPI) {
            continue;
        }

        const int required_ranks =
            registered_rank_count(benchmark.name);

        if (required_ranks != -1 &&
            required_ranks != world_size) {
            continue;
        }

        if (baselines.find(benchmark.kernel_name) ==
            baselines.end()) {
            throw std::runtime_error(
                "No Serial baseline found for " +
                benchmark.kernel_name
            );
        }

        active.push_back(benchmark);
    }

    if (active.empty()) {
        throw std::runtime_error(
            "No MPI benchmarks registered for " +
            std::to_string(world_size) + " ranks"
        );
    }

    for (const auto& benchmark : active) {
        const BenchmarkResult& baseline =
            baselines.at(benchmark.kernel_name);

        for (int i = 0;
             i < settings.warmup + settings.repetitions;
             ++i) {

            const bool is_warmup = i < settings.warmup;

            BenchmarkContext context;
            Timer timer;
            CpuMetrics cpu_metrics;

            context.assign_measurement_functions(
                [&]() {
                    start_mpi_measurement(
                        timer,
                        cpu_metrics
                    );
                },
                [&]() {
                    stop_mpi_measurement(
                        timer,
                        cpu_metrics
                    );
                }
            );

            benchmark.callable(context);

            if (!context.is_measurement_stopped()) {
                throw std::runtime_error(
                    "Benchmark did not stop measurement: " +
                    benchmark.name
                );
            }

            if (context.number_of_processors != world_size) {
                throw std::runtime_error(
                    "Incorrect processor count reported by " +
                    benchmark.name
                );
            }

            const double local_wall_time =
                timer.get_elapsed_time();

            const double local_cpu_time =
                cpu_metrics.get_cpu_time_ms();

            const long local_peak_rss =
                cpu_metrics.get_peak_rss_kb();

            double maximum_wall_time = 0.0;
            double total_cpu_time = 0.0;
            long maximum_peak_rss = 0;

            MPI_Reduce(
                &local_wall_time,
                &maximum_wall_time,
                1,
                MPI_DOUBLE,
                MPI_MAX,
                0,
                MPI_COMM_WORLD
            );

            MPI_Reduce(
                &local_cpu_time,
                &total_cpu_time,
                1,
                MPI_DOUBLE,
                MPI_SUM,
                0,
                MPI_COMM_WORLD
            );

            MPI_Reduce(
                &local_peak_rss,
                &maximum_peak_rss,
                1,
                MPI_LONG,
                MPI_MAX,
                0,
                MPI_COMM_WORLD
            );

            if (rank == 0) {
                BenchmarkResult result;

                result.name = benchmark.name;
                result.kernel_name = benchmark.kernel_name;
                result.paradigm = Paradigm::MPI;

                result.run_type = is_warmup
                    ? RunType::Warmup
                    : RunType::Core;

                result.rep_index = is_warmup
                    ? i
                    : i - settings.warmup;

                result.problem_size = context.problem_size;
                result.iterations = context.iterations;
                result.number_of_processors = world_size;

                result.wall_time_ms = maximum_wall_time;
                result.cpu_time_ms = total_cpu_time;
                result.peak_rss_kb = maximum_peak_rss;

                result.flop_count = context.flop_count;
                result.bytes_moved = context.bytes_moved;

                compute_achieved_metrics(
                    result,
                    context.flop_count,
                    context.bytes_moved
                );

                compute_derived_metrics(
                    result,
                    baseline,
                    context.flop_count,
                    context.bytes_moved
                );

                if (result_sink) {
                    result_sink(result);
                }
            }
        }
    }
}

} 

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::string filename;
    RunMetadata metadata;
    bool metadata_started = false;

    try {
        Options options;

        const int parse_status =
            parse_arguments(argc, argv, options, rank);

        if (parse_status == 1) {
            MPI_Finalize();
            return 1;
        }

        if (parse_status == 2) {
            MPI_Finalize();
            return 0;
        }

        BenchmarkSettings settings;
        settings.repetitions = options.repetitions;
        settings.warmup = options.warmup;

        const BaselineMap baselines =
            load_serial_baselines(options.baseline);

        std::ofstream output_file;

        if (rank == 0) {
            std::cout
                << "MPI ranks: " << world_size << '\n'
                << "Warm-up repetitions: " << settings.warmup << '\n'
                << "Measured repetitions: "
                << settings.repetitions << '\n';

            filename = generate_file_name();

            initialize_raw_data_csv(filename);

            metadata.schema_version = "1";
            metadata.csv_filename = filename;
            metadata.warmup_count = settings.warmup;
            metadata.repetitions = settings.repetitions;
            metadata.command_line =
                join_arguments(argc, argv);
            metadata.start_timestamp = current_timestamp();
            metadata.source_directory = PB_SOURCE_DIR;
            metadata.build_directory = PB_BUILD_DIR;
            metadata.query_cuda_runtime = false;

            if (const char* note = std::getenv("PB_NOTE")) {
                metadata.note = note;
            }

            write_run_metadata(metadata);
            metadata_started = true;

            output_file = open_file(
                filename,
                std::ios::app
            );

            std::cout << "Writing results to: "
                      << filename << '\n';
        }

        MPI_Barrier(MPI_COMM_WORLD);

        ResultSink csv_sink;

        if (rank == 0) {
            csv_sink = [&output_file](
                const BenchmarkResult& result
            ) {
                write_result_row(output_file, result);
                output_file.flush();

                if (!output_file) {
                    throw std::runtime_error(
                        "Failed to write benchmark: " +
                        result.name
                    );
                }
            };
        }

        run_mpi_benchmarks(
            settings,
            baselines,
            csv_sink,
            rank,
            world_size
        );

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            metadata.end_status = "completed";

            if (!append_run_end_metadata(metadata)) {
                std::cerr
                    << "Warning: could not append end metadata\n";
            }

            output_file.close();

            std::cout
                << "MPI benchmark run completed.\n"
                << "Results written to: " << filename << '\n';
        }

        MPI_Finalize();
        return 0;

    } catch (const std::exception& error) {
        if (rank == 0) {
            std::cerr << "Fatal MPI benchmark error: "
                      << error.what() << '\n';

            if (metadata_started) {
                metadata.end_status =
                    std::string("failed: ") + error.what();

                append_run_end_metadata(metadata);
            }
        }

        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
