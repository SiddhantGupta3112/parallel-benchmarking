#ifndef BENCHMARK_RESULT_H
#define BENCHMARK_RESULT_H
#include<string>
#include<optional>

enum class Paradigm {
    Serial,
    OpenMP,
    MPI,
    CUDA
};

struct BenchmarkResult{
    //config fields
    std::string name; 
    Paradigm paradigm; 
    int problem_size;
    int iterations;
    int number_of_processors;

    //compulsory fields;
    double wall_time_ms = 0.0;
    double speedup = 0.0;
    double efficiency = 0.0;
    double achieved_flops = 0.0;
    double arithmetic_intensity = 0.0;
    double cpu_time_ms = 0.0;
    long peak_rss_kb = 0;

    //optional fields;
    std::optional<double> gpu_utilization_pct;
    std::optional<double> gpu_memory_used_mb;
    std::optional<double> gpu_memory_throughput_gbps;
    std::optional<double> mpi_comm_overhead_fraction;
};


#endif