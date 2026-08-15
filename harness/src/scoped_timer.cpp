#include "scoped_timer.h"

void Timer::start() {
    this->start_time = std::chrono::steady_clock::now();
}

void Timer::stop() {
    this->end_time = std::chrono::steady_clock::now();
}

double Timer::get_elapsed_time() {
    return std::chrono::duration<double, std::milli>(this->end_time - this->start_time).count();
}