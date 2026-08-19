#ifndef BENCHMARK_CONTEXT_H
#define BENCHMARK_CONTEXT_H

struct BenchmarkContext {
    double flop_count = 0.0;
    double bytes_moved = 0.0;
    int problem_size;
    int number_of_processors;
    int iterations;
};

#endif