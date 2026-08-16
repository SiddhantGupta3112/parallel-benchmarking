#include "live_monitor.h"
#include <catch2/catch_test_macros.hpp>
#include "benchmark_result.h"
#include <thread>
#include <chrono>

TEST_CASE("LiveMonitor tracks the peak value, not the most recent one") {
    bool first_call = true;

    auto fake_source = [first_call]() mutable -> double {
        if (first_call) {
            first_call = false;
            return 100.0;
        }
        return 10.0;
    };

    LiveMonitor monitor(fake_source);
    monitor.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    monitor.stop();

    BenchmarkResult result;
    monitor.populate(result);

    REQUIRE(result.gpu_memory_used_mb == 100.0);
}