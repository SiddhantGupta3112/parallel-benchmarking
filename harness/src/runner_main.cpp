#include <iostream>
#include "runner.h"

int main(){
    std::unordered_map<std::string, BenchmarkResult> serial_benchmarks = run_serial_benchmarks();
    std::vector<BenchmarkResult> results = run_non_serial_benchmarks(serial_benchmarks);

    std::cout << "Serial benchmarks run: " << serial_benchmarks.size() << "\n";
    std::cout << "Non-serial benchmarks run: " << results.size() << "\n";

    return 0;
}