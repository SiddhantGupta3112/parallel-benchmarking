#include "result_writer.h"
#include "benchmark_result.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iomanip>
#include <limits>

namespace{
    std::string paradigm_to_string(Paradigm paradigm) {
        switch (paradigm) {
            case Paradigm::Serial:
                return "Serial";
            case Paradigm::OpenMP:
                return "OpenMP";
            case Paradigm::MPI:
                return "MPI";
            case Paradigm::CUDA:
                return "CUDA";
            default:
                return "UNKNOWN";
        }
    }

    std::string run_type_to_string(RunType type) {
    switch (type) {
        case RunType::Warmup: return "warmup";
        case RunType::Core:   return "core";
        default:              return "unset";
    }
}
}

void write_raw_data_header(std::ofstream& file) {
    file
        << "name,kernel_name,paradigm,run_type,rep_index,"
        << "problem_size,iterations,number_of_processors,"
        << "wall_time_ms,speedup,efficiency,achieved_flops,"
        << "arithmetic_intensity,cpu_time_ms,peak_rss_kb,"
        << "flop_count,bytes_moved,"
        << "gpu_utilization_pct,gpu_memory_used_mb,"
        << "gpu_memory_throughput_gbps,mpi_comm_overhead_fraction\n";

    if (!file) {
        throw std::runtime_error("Failed to write CSV header");
    }
}

void write_result_row(
    std::ofstream& file,
    const BenchmarkResult& result
) {
    file
        << "\"" << result.name << "\","
        << "\"" << result.kernel_name << "\","
        << "\"" << paradigm_to_string(result.paradigm) << "\","
        << "\"" << run_type_to_string(result.run_type) << "\","
        << result.rep_index << ","
        << result.problem_size << ","
        << result.iterations << ","
        << result.number_of_processors << ","
        << result.wall_time_ms << ","
        << result.speedup << ","
        << result.efficiency << ","
        << result.achieved_flops << ","
        << result.arithmetic_intensity << ","
        << result.cpu_time_ms << ","
        << result.peak_rss_kb << ","
        << result.flop_count << ","
        << result.bytes_moved << ",";

    if (result.gpu_utilization_pct.has_value()) {
        file << result.gpu_utilization_pct.value();
    }

    file << ",";

    if (result.gpu_memory_used_mb.has_value()) {
        file << result.gpu_memory_used_mb.value();
    }

    file << ",";

    if (result.gpu_memory_throughput_gbps.has_value()) {
        file << result.gpu_memory_throughput_gbps.value();
    }

    file << ",";

    if (result.mpi_comm_overhead_fraction.has_value()) {
        file << result.mpi_comm_overhead_fraction.value();
    }

    file << '\n';

    if (!file) {
        throw std::runtime_error(
            "Failed to write benchmark result: " + result.name
        );
    }
}

std::ofstream open_file(
    const std::string& filename,
    std::ios::openmode mode
) {
    std::ofstream file(filename, mode);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Failed to open output file: " + filename
        );
    }

    file << std::setprecision(
        std::numeric_limits<double>::max_digits10
    );

    return file;
}

void initialize_raw_data_csv(const std::string& filename) {
    try {
        std::filesystem::create_directories("results");
    } catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error(
            "Failed to create results directory: " +
            std::string(e.what())
        );
    }

    auto file = open_file(filename, std::ios::out);
    write_raw_data_header(file);
}

void write_raw_data_csv(
    const std::string& filename,
    const std::vector<BenchmarkResult>& results
) {
    auto file = open_file(filename, std::ios::out);
    write_raw_data_header(file);

    for (const auto& result : results) {
        write_result_row(file, result);
    }

    file.flush();

    if (!file) {
        throw std::runtime_error(
            "Failed to finish writing CSV: " + filename
        );
    }
}
