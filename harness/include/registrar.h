#ifndef REGISTRAR_H
#define REGISTRAR_H

#include <string>
#include "benchmark_result.h"
#include "benchmark_context.h"
#include <functional>
#include <vector>

struct RegisteredBenchmark {
    std::string name;
    std::string kernel_name;
    Paradigm paradigm;
    std::function<void(BenchmarkContext&)> callable;
};

class Registry {
private:
    std::vector<RegisteredBenchmark> registered_benchmarks;
public:
    void register_benchmark(std::string name, std::string kernel_name, Paradigm paradigm, std::function<void(BenchmarkContext&)> callable);
    std::vector<RegisteredBenchmark>& access_registered_benchmarks();
};

class Registrar {
public:
    Registrar(std::string name, std::string kernel_name, Paradigm paradigm, std::function<void(BenchmarkContext&)> callable);
};

Registry& get_registry();

#define BENCHMARK_TEST(name, kernel_name, paradigm) \
    void name##_impl(BenchmarkContext&); \
    static Registrar name##_registrar(#name, #kernel_name, paradigm, name##_impl); \
    void name##_impl(BenchmarkContext& ctx)



#endif