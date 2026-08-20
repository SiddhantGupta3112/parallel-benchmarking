#include "derived_metrics.h"
#include <stdexcept>

void compute_achieved_metrics(
    BenchmarkResult& result,
    double flop_count,
    double bytes_moved
) {
    if (result.wall_time_ms <= 0.0) {
        throw std::runtime_error(
            "Wall-clock time must be greater than zero"
        );
    }

    if (flop_count < 0.0) {
        throw std::runtime_error(
            "FLOP count cannot be negative"
        );
    }

    if (bytes_moved <= 0.0) {
        throw std::runtime_error(
            "Bytes moved must be greater than zero"
        );
    }

    result.achieved_flops =
        flop_count / (result.wall_time_ms / 1000.0);

    result.arithmetic_intensity =
        flop_count / bytes_moved;
}

void compute_derived_metrics(
    BenchmarkResult& result,
    const BenchmarkResult& serial_baseline,
    double flop_count,
    double bytes_moved
) {
    if (result.problem_size != serial_baseline.problem_size ||
        result.iterations != serial_baseline.iterations) {
        throw std::runtime_error(
            "Baseline and result configuration fields mismatch"
        );
    }

    if (result.wall_time_ms <= 0.0 ||
        serial_baseline.wall_time_ms <= 0.0) {
        throw std::runtime_error(
            "Wall-clock time must be greater than zero"
        );
    }

    if (result.number_of_processors <= 0) {
        throw std::runtime_error(
            "Number of processors must be greater than zero"
        );
    }

    if (flop_count < 0.0) {
        throw std::runtime_error(
            "FLOP count cannot be negative"
        );
    }

    if (bytes_moved <= 0.0) {
        throw std::runtime_error(
            "Bytes moved must be greater than zero"
        );
    }

    result.speedup =
        serial_baseline.wall_time_ms / result.wall_time_ms;

    result.efficiency =
        result.speedup / result.number_of_processors;
}