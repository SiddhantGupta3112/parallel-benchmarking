add_executable(
    harness_tests
    test_benchmark_result.cpp
    test_registrar.cpp
    test_derived_metrics.cpp
    test_cpu_metrics.cpp
    test_timer.cpp
    ../src/registrar.cpp
    ../src/derived_metrics.cpp
    ../src/cpu_metrics.cpp
    ../src/scoped_timer.cpp
)

target_include_directories(
    harness_tests
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(
    harness_tests
    PRIVATE
    Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(harness_tests)