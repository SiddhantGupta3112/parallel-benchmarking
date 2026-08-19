add_executable(
    harness_tests
    test_benchmark_result.cpp
    test_registrar.cpp
    test_derived_metrics.cpp
    test_cpu_metrics.cpp
    test_timer.cpp
    test_live_monitor.cpp
    test_runner.cpp
)

target_link_libraries(
    harness_tests
    PRIVATE
    harness
    Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(harness_tests)