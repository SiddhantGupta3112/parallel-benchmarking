#ifndef REGISTRAR_H
#define REGISTRAR_H

#include <string>
#include "benchmark_result.h"
#include <functional>
#include <vector>

struct RegisteredBenchmark {
    std::string name;
    Paradigm paradigm;
    std::function<void()> callable;
};

class Registry {
private:
    std::vector<RegisteredBenchmark> registered_benchmarks;
public:
    void register_benchmark(std::string name, Paradigm paradigm, std::function<void()> callable);
    std::vector<RegisteredBenchmark>& access_registered_benchmarks();
};

class Registrar {
public:
    Registrar(std::string name, Paradigm paradigm, std::function<void()> callable);
};

Registry& get_registry();

#define BENCHMARK_TEST(name, paradigm) \
    void name##_impl(); \
    static Registrar name##_registrar(#name, paradigm, name##_impl); \
    void name##_impl()



#endif