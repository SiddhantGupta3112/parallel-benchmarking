#include <cstdlib>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>

#include "runner.h"
#include "result_writer.h"
#include "benchmark_settings.h"
#include "metadata_collector.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

// Locations baked in by CMake (see target_compile_definitions on
// benchmark_runner). The fallbacks keep the file compiling without them.
#ifndef PB_SOURCE_DIR
#define PB_SOURCE_DIR "."
#endif
#ifndef PB_BUILD_DIR
#define PB_BUILD_DIR "build"
#endif


namespace {

    constexpr int PARSE_SUCCESS = 0;
    constexpr int PARSE_ERROR = 1;
    constexpr int PARSE_HELP = 2;

    constexpr int MAX_REPETITIONS = 1000;
    constexpr int MAX_WARMUP = 1000;

    // Version of the result CSV layout, recorded in the metadata file.
    constexpr const char* CSV_SCHEMA_VERSION = "1";


    std::string generate_file_name() {
        auto now = std::chrono::system_clock::now();
        std::time_t time = std::chrono::system_clock::to_time_t(now);

        std::tm local_time = *std::localtime(&time);

        std::ostringstream timestamp;

        timestamp << std::put_time(&local_time, "%Y-%m-%d_%H-%M-%S");

        return "results/" + timestamp.str() + ".csv";
    }

    // Rebuilds the command line as typed, for the metadata file.
    std::string join_arguments(int argc, char* argv[]) {
        std::string joined;

        for (int i = 0; i < argc; ++i) {
            if (i > 0) {
                joined += ' ';
            }
            joined += argv[i];
        }

        return joined;
    }

    int parse_integer(const char* value) {
        std::string input(value);
        std::size_t pos = 0;

        int result = std::stoi(input, &pos);

        if (pos != input.size()) {
            throw std::invalid_argument(
                "Expected an integer, got: " + input
            );
        }

        return result;
    }

    int parse_arguments(int argc, char* argv[], BenchmarkSettings& settings) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];

            if (arg == "--help") {
                std::cout
                    << "Usage: " << argv[0] << " [options]\n\n"
                    << "Options:\n"
                    << "  --repetitions <num>  Number of measured runs "
                    << "(1-" << MAX_REPETITIONS << ")\n"
                    << "  --warmup <num>       Number of warmup runs "
                    << "(0-" << MAX_WARMUP << ")\n"
                    << "  --help               Show this help message\n\n"
                    << "Environment:\n"
                    << "  PB_NOTE              Free-text note stored in the metadata file\n"
                    << "                       (power mode, plugged in, apps closed, ...)\n";

                return PARSE_HELP;
            }

            if (arg == "--repetitions" || arg == "--warmup") {
                const bool is_repetitions = (arg == "--repetitions");

                if (i + 1 >= argc) {
                    std::cerr
                        << "Error: Missing value for " << arg << "\n";
                    return PARSE_ERROR;
                }

                int value;

                try {
                    value = parse_integer(argv[++i]);
                } catch (const std::exception& e) {
                    std::cerr
                        << "Error: Invalid value for " << arg
                        << ": " << e.what() << "\n";
                    return PARSE_ERROR;
                }

                if (is_repetitions) {
                    if (value < 1 || value > MAX_REPETITIONS) {
                        std::cerr
                            << "Error: --repetitions must be between 1 and "
                            << MAX_REPETITIONS << ".\n";
                        return PARSE_ERROR;
                    }

                    settings.repetitions = value;
                } else {
                    if (value < 0 || value > MAX_WARMUP) {
                        std::cerr
                            << "Error: --warmup must be between 0 and "
                            << MAX_WARMUP << ".\n";
                        return PARSE_ERROR;
                    }

                    settings.warmup = value;
                }
            } else {
                std::cerr << "Error: Unknown argument: " << arg << "\n";
                std::cerr << "Use --help to see available options.\n";
                return PARSE_ERROR;
            }
        }

        return PARSE_SUCCESS;
    }

}

int main(int argc, char* argv[]) {
    std::string filename;
    RunMetadata metadata;
    bool metadata_started = false;

    try {
        BenchmarkSettings settings;
        int parse_result = parse_arguments(argc, argv, settings);

        if(parse_result == PARSE_ERROR){
            return 1;
        }

        if(parse_result == PARSE_HELP){
            return 0;
        }


        std::cout << "Benchmark Settings:\n";
        std::cout << "Repetitions: " << settings.repetitions << "\n";
        std::cout << "Warmup: " << settings.warmup << "\n";

        filename = generate_file_name();
        initialize_raw_data_csv(filename);

        std::cout << "Writing Results to: " << filename << "\n";

        // Record how this run was produced BEFORE any benchmark starts, so
        // even a crashed run leaves its metadata behind.
        metadata.schema_version   = CSV_SCHEMA_VERSION;
        metadata.csv_filename     = filename;
        metadata.warmup_count     = settings.warmup;
        metadata.repetitions      = settings.repetitions;
        metadata.command_line     = join_arguments(argc, argv);
        metadata.start_timestamp  = current_timestamp();
        metadata.source_directory = PB_SOURCE_DIR;
        metadata.build_directory  = PB_BUILD_DIR;
        if (const char* note = std::getenv("PB_NOTE")) {
            metadata.note = note;
        }
        write_run_metadata(metadata);
        metadata_started = true;

        auto file_pointer = open_file(filename, std::ios::app);

        ResultSink csv_sink = [&file_pointer](const BenchmarkResult& result) {
            write_result_row(file_pointer, result);
            file_pointer.flush();

            if (!file_pointer) {
                throw std::runtime_error(
                    "Failed to write result for benchmark: " + result.name
                );
            }
        };


        std::unordered_map<std::string, BenchmarkResult> serial_benchmarks =
            run_serial_benchmarks(settings, csv_sink);
        std::vector<BenchmarkResult> results =
            run_non_serial_benchmarks(settings, serial_benchmarks, csv_sink);

        // A metadata problem must never turn a finished run into a failure.
        metadata.end_status = "completed";
        if (!append_run_end_metadata(metadata)) {
            std::cerr << "Warning: could not append the end section "
                      << "to the metadata file\n";
        }

        std::cout << "Serial benchmarks run: " << serial_benchmarks.size() << "\n";
        std::cout << "Total benchmarks run: " << results.size() << "\n";

        std::cout << "Results written to: " << filename << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";

        if (metadata_started) {
            metadata.end_status = std::string("failed: ") + e.what();
            append_run_end_metadata(metadata);
        }

        if (!filename.empty()) {
            std::cout << "Partial results written to: " << filename << "\n";
        }
        return 1;
    }
    return 0;
}