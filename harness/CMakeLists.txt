add_library(harness STATIC
    src/registrar.cpp
    src/derived_metrics.cpp
    src/cpu_metrics.cpp
    src/scoped_timer.cpp
    src/live_monitor.cpp
    src/runner.cpp
    src/result_writer.cpp
)

target_include_directories(
    harness
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(
    harness
    PRIVATE
        Threads::Threads
)

if(HAS_CUDA)
    target_sources(
        harness
        PRIVATE
            src/cuda_metrics.cpp
    )

    target_link_libraries(
        harness
        PRIVATE
            CUDA::nvml
    )

    target_compile_definitions(
        harness
        PRIVATE
            HAS_CUDA
    )
endif()


