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
    double wall_time_ms;
    double speedup;
    double efficiency;
    double achieved_flops;
    double arithmetic_intensity;
    double cpu_time_ms;
    long peak_rss_kb;

    //optional fields;
    std::optional<double> gpu_utilization_pct;
    std::optional<double> gpu_memory_used_mb;
    std::optional<double> gpu_memory_throughput_gbps;
    std::optional<double> mpi_comm_overhead_fraction;
};


#endif