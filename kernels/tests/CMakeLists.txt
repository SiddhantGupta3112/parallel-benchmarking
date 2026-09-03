add_executable(
    kernel_tests
    test_serial_monte_carlo.cpp
    test_matrix_multiply.cpp
)

target_link_libraries(
    kernel_tests
    PRIVATE
    serial_kernels
    Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(kernel_tests)