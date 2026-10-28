#include "openmp_helper.h"

#include <stdexcept>

OpenMPThreadScope::OpenMPThreadScope(int num_threads)
    : previous_num_threads(omp_get_max_threads()) {
    if (num_threads <= 0) {
        throw std::invalid_argument("Number of threads must be a positive integer.");
    }

    omp_set_num_threads(num_threads);
}

OpenMPThreadScope::~OpenMPThreadScope() {
    omp_set_num_threads(previous_num_threads);
}

int get_number_of_threads(int num_threads) {
    OpenMPThreadScope scope(num_threads);

    int actual_threads = 1;

    #pragma omp parallel
    {
        #pragma omp single
        actual_threads = omp_get_num_threads();
    }

    return actual_threads;
}
