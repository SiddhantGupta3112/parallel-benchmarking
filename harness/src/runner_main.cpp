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

int main() {
    std::string filename;
    try {
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
            run_serial_benchmarks(csv_sink);
        std::vector<BenchmarkResult> results =
            run_non_serial_benchmarks(serial_benchmarks, csv_sink);
        merge_benchmarks(results, serial_benchmarks);

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