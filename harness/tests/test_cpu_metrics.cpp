#include "cpu_metrics.h"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("CPU time greater than 0"){
    CpuMetrics metrics;

    long long sum = 0;
    long n = 100000000;
    metrics.start();
    for(int i = 0; i <= n; i++){
        sum += i;
    }

    metrics.stop();

    REQUIRE(sum == n * (n + 1) / 2);
    REQUIRE(metrics.get_cpu_time_ms() > 0.0);
    REQUIRE(metrics.get_peak_rss_kb() > 0);
}