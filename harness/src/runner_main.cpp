#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>

#include "runner.h"
#include "result_writer.h"
#include "benchmark_settings.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif


namespace {

    constexpr int PARSE_SUCCESS = 0;
    constexpr int PARSE_ERROR = 1;
    constexpr int PARSE_HELP = 2;

    constexpr int MAX_REPETITIONS = 1000;
    constexpr int MAX_WARMUP = 1000;


    std::string generate_file_name() {
        auto now = std::chrono::system_clock::now();
        std::time_t time = std::chrono::system_clock::to_time_t(now);

        std::tm local_time = *std::localtime(&time);

        std::ostringstream timestamp;

        timestamp << std::put_time(&local_time, "%Y-%m-%d_%H-%M-%S");

        return "results/" + timestamp.str() + ".csv";
    }

    void merge_benchmarks(
        std::vector<BenchmarkResult>& results,
        const std::unordered_map<std::string, BenchmarkResult>& serial_benchmarks
    ) {
        for (const auto& [name, result] : serial_benchmarks) {
            results.push_back(result);
        }
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
                    << "  --help               Show this help message\n";

                return 2;
            }

            if (arg == "--repetitions" || arg == "--warmup") {
                const bool is_repetitions = (arg == "--repetitions");

                if (i + 1 >= argc) {
                    std::cerr
                        << "Error: Missing value for " << arg << "\n";
                    return 1;
                }

                int value;

                try {
                    value = parse_integer(argv[++i]);
                } catch (const std::exception& e) {
                    std::cerr
                        << "Error: Invalid value for " << arg
                        << ": " << e.what() << "\n";
                    return 1;
                }

                if (is_repetitions) {
                    if (value < 1 || value > MAX_REPETITIONS) {
                        std::cerr
                            << "Error: --repetitions must be between 1 and "
                            << MAX_REPETITIONS << ".\n";
                        return 1;
                    }

                    settings.repetitions = value;
                } else {
                    if (value < 0 || value > MAX_WARMUP) {
                        std::cerr
                            << "Error: --warmup must be between 0 and "
                            << MAX_WARMUP << ".\n";
                        return 1;
                    }

                    settings.warmup = value;
                }
            } else {
                std::cerr << "Error: Unknown argument: " << arg << "\n";
                std::cerr << "Use --help to see available options.\n";
                return 1;
            }
        }

        return 0; 
    }

}

int main(int argc, char* argv[]) {
    std::string filename;
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
        //merge_benchmarks(results, serial_benchmarks);

        std::cout << "Serial benchmarks run: " << serial_benchmarks.size() << "\n";
        std::cout << "Total benchmarks run: " << results.size() << "\n";

        std::cout << "Results written to: " << filename << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        std::cout << "Partial results written to: " << filename << "\n";
        return 1;
    }
    return 0;
}