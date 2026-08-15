#ifndef SCOPED_TIMER_H
#define SCOPED_TIMER_H

#include<chrono>

class Timer{
private:
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;

public:
    void start();
    void stop();
    double get_elapsed_time();
};

#endif