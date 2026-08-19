add_library(harness STATIC
    src/registrar.cpp
    src/derived_metrics.cpp
    src/cpu_metrics.cpp
    src/scoped_timer.cpp
    src/live_monitor.cpp
    src/cuda_metrics.cpp
    src/runner.cpp
)

target_include_directories(
    harness
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    harness
    PRIVATE
        CUDA::nvml
)

add_executable(
    benchmark_runner
    src/runner_main.cpp
)
find_package(Threads REQUIRED)

target_link_libraries(
    benchmark_runner
    PRIVATE
        harness
        Threads::Threads
)