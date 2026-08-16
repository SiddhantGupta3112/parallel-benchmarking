#include "live_monitor.h"
#include <chrono>
#include <algorithm>

LiveMonitor::LiveMonitor(std::function<double()> source)
    : peak_gpu_memory_mb(0.0), keep_running(false), sample_source(source) {}

void LiveMonitor::monitor() {
    while (keep_running) {
        double current = sample_source();
        peak_gpu_memory_mb = std::max(peak_gpu_memory_mb, current);

        std::this_thread::sleep_for(
            std::chrono::duration<double>(POLLING_INTERVAL_SEC)
        );
    }
}

void LiveMonitor::start() {
    peak_gpu_memory_mb = 0.0;
    keep_running = true;
    monitor_thread = std::thread(&LiveMonitor::monitor, this);
}

void LiveMonitor::stop() {
    keep_running = false;
    if (monitor_thread.joinable()) {
        monitor_thread.join();
    }
}

void LiveMonitor::populate(BenchmarkResult& result) const {
    result.gpu_memory_used_mb = peak_gpu_memory_mb;
}

LiveMonitor::~LiveMonitor() {
    stop();
}