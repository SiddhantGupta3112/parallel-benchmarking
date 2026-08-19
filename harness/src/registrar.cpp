#include "registrar.h"

Registry& get_registry() {
    static Registry instance;  
    return instance;
}

void Registry::register_benchmark(std::string name, std::string kernel_name, Paradigm paradigm, std::function<void(BenchmarkContext&)> callable){
    RegisteredBenchmark new_benchmark;
    new_benchmark.name = name;
    new_benchmark.kernel_name = kernel_name;
    new_benchmark.paradigm = paradigm;
    new_benchmark.callable = callable;

    registered_benchmarks.push_back(new_benchmark);
}

std::vector<RegisteredBenchmark>& Registry::access_registered_benchmarks(){
    return registered_benchmarks;
}

Registrar::Registrar(std::string name, std::string kernel_name, Paradigm paradigm, std::function<void(BenchmarkContext&)> callable){
    get_registry().register_benchmark(name, kernel_name, paradigm, callable);
}

