#include "cpu_metrics.h"

void CpuMetrics::start() {
    getrusage(RUSAGE_SELF, &this->start_metrics);
}

void CpuMetrics::stop() {
    getrusage(RUSAGE_SELF, &this->stop_metrics);
}

double CpuMetrics::get_cpu_time_ms() const {
    double start_user =
        start_metrics.ru_utime.tv_sec * 1000.0 +
        start_metrics.ru_utime.tv_usec / 1000.0;

    double start_system =
        start_metrics.ru_stime.tv_sec * 1000.0 +
        start_metrics.ru_stime.tv_usec / 1000.0;

    double stop_user =
        stop_metrics.ru_utime.tv_sec * 1000.0 +
        stop_metrics.ru_utime.tv_usec / 1000.0;

    double stop_system =
        stop_metrics.ru_stime.tv_sec * 1000.0 +
        stop_metrics.ru_stime.tv_usec / 1000.0;

    return (stop_user + stop_system) - (start_user + start_system);
}

long CpuMetrics::get_peak_rss_kb() const {
    return stop_metrics.ru_maxrss;
}