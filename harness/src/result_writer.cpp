#include <fstream>
#include <string>
#include <vector>
#include "benchmark_result.h"
#include <stdexcept>
#include "result_writer.h"

std::string paradigm_to_string(Paradigm paradigm){
    switch (paradigm)
    {
    case Paradigm::Serial :
        return "Serial";
    case Paradigm::OpenMP :
        return "OpenMP";
    case Paradigm::MPI :
        return "MPI";
    case Paradigm::CUDA :
        return "CUDA";
    default:
        return "UNKNOWN";
    }
}

void write_csv(
    const std::string& filename,
    const std::vector<BenchmarkResult>& results
) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + filename);
    }

    file << "name,paradigm,problem_size,iterations,number_of_processors,"
            "wall_time_ms,speedup,efficiency,achieved_flops,"
            "arithmetic_intensity,cpu_time_ms,peak_rss_kb,"
            "gpu_utilization_pct,gpu_memory_used_mb,"
            "gpu_memory_throughput_gbps,mpi_comm_overhead_fraction\n";

    for (const auto& result : results) {
        file
            << "\"" << result.name << "\","
            << "\"" << paradigm_to_string(result.paradigm) << "\","
            << result.problem_size << ","
            << result.iterations << ","
            << result.number_of_processors << ","
            << result.wall_time_ms << ","
            << result.speedup << ","
            << result.efficiency << ","
            << result.achieved_flops << ","
            << result.arithmetic_intensity << ","
            << result.cpu_time_ms << ","
            << result.peak_rss_kb << ",";

        if (result.gpu_utilization_pct.has_value())
            file << result.gpu_utilization_pct.value();
        else
            file << "";

        file << ",";

        if (result.gpu_memory_used_mb.has_value())
            file << result.gpu_memory_used_mb.value();
        else
            file << "";

        file << ",";

        if (result.gpu_memory_throughput_gbps.has_value())
            file << result.gpu_memory_throughput_gbps.value();
        else
            file << "";

        file << ",";

        //MPI COMM OVERHEAD does not exist yet
        file << "";

        file << "\n";
    }
}