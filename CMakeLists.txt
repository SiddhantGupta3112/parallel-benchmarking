cmake_minimum_required(VERSION 3.14)

project(parallel_benchmarking LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.2
)

FetchContent_MakeAvailable(Catch2)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(Threads REQUIRED)


# CUDA is optional
option(
    ENABLE_CUDA
    "Enable CUDA benchmarking support"
    ON
)

include(CheckLanguage)

if(ENABLE_CUDA)

    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)

        set(
            CMAKE_CUDA_ARCHITECTURES
            86
            CACHE STRING
            "CUDA architectures to build for"
        )

        enable_language(CUDA)

        find_package(CUDAToolkit REQUIRED)

        set(HAS_CUDA TRUE)

        message(STATUS "CUDA benchmarking support enabled")
        message(STATUS "CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")

    else()

        set(HAS_CUDA FALSE)

        message(
            STATUS
            "CUDA compiler not found. CUDA support disabled."
        )

    endif()

else()

    set(HAS_CUDA FALSE)

    message(
        STATUS
        "CUDA benchmarking support disabled"
    )

endif()


add_subdirectory(harness)
add_subdirectory(harness/tests)
add_subdirectory(kernels/openmp)
add_subdirectory(kernels/mpi)
add_subdirectory(kernels/serial)
add_subdirectory(kernels/tests)
if(HAS_CUDA)
    add_subdirectory(kernels/cuda)
endif()

add_executable(benchmark_runner harness/src/runner_main.cpp)

target_link_libraries(benchmark_runner
    PRIVATE
        harness
        openmp_kernels
        mpi_kernels
        serial_kernels
)

if(HAS_CUDA)
    target_link_libraries(benchmark_runner PRIVATE cuda_kernels)
endif()