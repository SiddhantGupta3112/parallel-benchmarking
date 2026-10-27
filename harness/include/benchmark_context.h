#ifndef BENCHMARK_CONTEXT_H
#define BENCHMARK_CONTEXT_H

#include <functional>
#include <utility>
#include <stdexcept>

class BenchmarkContext {
private:
    std::function<void()> start_measurement_hook;
    std::function<void()> stop_measurement_hook;
    bool measurement_stopped = false;
    bool measurement_started = false;

public:
    double flop_count = 0.0;
    double bytes_moved = 0.0;
    int problem_size = 0;
    int number_of_processors = 0;
    int iterations = 0;
    double mpi_communication_time_ms = 0.0;

    void assign_measurement_functions(
        std::function<void()> start_func,
        std::function<void()> stop_func
    ) {
        start_measurement_hook = std::move(start_func);
        stop_measurement_hook = std::move(stop_func);
    }

    void start_measurement() {
        if(!start_measurement_hook) {
            throw std::runtime_error("Start measurement function not assigned.");
        }
        if (start_measurement_hook && !measurement_stopped) {
            start_measurement_hook();
            measurement_started = true;
        }
    }

    bool is_measurement_stopped() const {
        return measurement_stopped;
    }

    void stop_measurement() {
        if(!stop_measurement_hook) {
            throw std::runtime_error("Stop measurement function not assigned.");
        }
        if(!measurement_started){
            throw std::runtime_error("Attempted to stop measurement before starting it.");
        }
        if (stop_measurement_hook && measurement_started && !measurement_stopped) {
            stop_measurement_hook();
            measurement_stopped = true;
        }
    }
};

#endif