#ifndef LIVE_MONITOR_H
#define LIVE_MONITOR_H

#include <thread>
#include <atomic>
#include <functional>
#include "benchmark_result.h"
#include "cuda_metrics.h"

constexpr double POLLING_INTERVAL_SEC = 0.05;

class LiveMonitor {
private:
    double peak_gpu_memory_mb;
    std::thread monitor_thread;
    std::atomic<bool> keep_running;
    std::function<double()> sample_source;

    void monitor();

public:
    explicit LiveMonitor(std::function<double()> source = read_current_gpu_memory_mb);
    ~LiveMonitor();

    void start();
    void stop();
    void populate(BenchmarkResult& result) const;
};

#endif