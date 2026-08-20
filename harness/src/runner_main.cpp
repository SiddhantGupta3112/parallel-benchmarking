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
    try {
        std::unordered_map<std::string, BenchmarkResult> serial_benchmarks =
            run_serial_benchmarks();
        std::vector<BenchmarkResult> results =
            run_non_serial_benchmarks(serial_benchmarks);
        merge_benchmarks(results, serial_benchmarks);

        std::cout << "Serial benchmarks run: " << serial_benchmarks.size() << "\n";
        std::cout << "Total benchmarks run: " << results.size() << "\n";

        std::string filename = generate_file_name();
        write_csv(filename, results);

        std::cout << "Results written to: " << filename << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}