#include "helper.h"
#include<cuda_runtime.h>

int get_number_of_processors(int thread_count) {
    int count = 0;
    cudaGetDeviceCount(&count);

    if (count == 0) {
        throw std::runtime_error("No CUDA Device found"); 
    }

    int current_device = 0;
    cudaGetDevice(&current_device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);

    int max_hardware_threads = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;

    return std::min(thread_count, max_hardware_threads);
}