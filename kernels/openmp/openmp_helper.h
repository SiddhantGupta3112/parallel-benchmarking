#ifndef OPENMP_HELPER_H
#define OPENMP_HELPER_H

#include <omp.h>

// ---------------------------------------------------------------------------
// Thread-count sweep.
//
// Every OpenMP benchmark is registered once per thread count listed here, as
//     <kernel>_OpenMP_<size>_t<threads>
// with the same kernel_name as the serial baseline. This is the ONLY place the
// list of thread counts lives: edit it here to change the sweep.
//
// Usage:  OPENMP_THREAD_SWEEP(SOME_BENCHMARK_MACRO, arg1, arg2, ...)
// expands to  SOME_BENCHMARK_MACRO(arg1, arg2, ..., <threads>)  per count.
// ---------------------------------------------------------------------------
#define OPENMP_THREAD_SWEEP(MACRO, ...) \
    MACRO(__VA_ARGS__, 1)  \
    MACRO(__VA_ARGS__, 2)  \
    MACRO(__VA_ARGS__, 4)  \
    MACRO(__VA_ARGS__, 8)  \
    MACRO(__VA_ARGS__, 12) \
    MACRO(__VA_ARGS__, 16) \
    MACRO(__VA_ARGS__, 20)

// Sets the OpenMP thread count with omp_set_num_threads() for the lifetime of
// the object and restores the previous value on destruction, so a kernel that
// runs hundreds of times in one process leaves no changed global state behind.
// Throws std::invalid_argument if num_threads <= 0.
class OpenMPThreadScope {
public:
    explicit OpenMPThreadScope(int num_threads);
    ~OpenMPThreadScope();

    OpenMPThreadScope(const OpenMPThreadScope&) = delete;
    OpenMPThreadScope& operator=(const OpenMPThreadScope&) = delete;

private:
    int previous_num_threads;
};

// Number of threads a parallel region actually gets when `num_threads` is
// requested (the runtime may give fewer, e.g. under OMP_THREAD_LIMIT). Runs a
// tiny parallel region, so call it OUTSIDE the measured window. This is the
// value reported in BenchmarkContext::number_of_processors.
int get_number_of_threads(int num_threads);

#endif // OPENMP_HELPER_H
