#ifndef CPU_METRICS_H
#define CPU_METRICS_H

#include <sys/resource.h>

class CpuMetrics {
private:
    struct rusage start_metrics;
    struct rusage stop_metrics;

public:
    void start();
    void stop();

    double get_cpu_time_ms() const;
    long get_peak_rss_kb() const;
};

#endif