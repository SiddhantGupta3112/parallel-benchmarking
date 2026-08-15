add_executable(
    harness_tests
    test_benchmark_result.cpp
    test_registrar.cpp
    ../src/registrar.cpp
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