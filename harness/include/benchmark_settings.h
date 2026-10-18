#ifndef BENCHMARK_SETTINGS_H
#define BENCHMARK_SETTINGS_H

struct BenchmarkSettings{
    static constexpr int DEFAULT_REPETITIONS = 1;
    static constexpr int DEFAULT_WARMUP = 0;

    int repetitions = DEFAULT_REPETITIONS;
    int warmup = DEFAULT_WARMUP;
};


#endif