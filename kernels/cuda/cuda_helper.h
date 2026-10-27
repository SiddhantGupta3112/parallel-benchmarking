#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <algorithm>

int get_number_of_processors(int thread_count);

inline void cuda_check(
    cudaError_t status,
    const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(status));
    }
}

#endif // CUDA_HELPER_H