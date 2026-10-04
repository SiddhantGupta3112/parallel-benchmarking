add_executable(
    kernel_tests
    test_serial_monte_carlo.cpp
    test_serial_matrix_multiply.cpp
    test_serial_stencil.cpp
)

target_link_libraries(
    kernel_tests
    PRIVATE
    serial_kernels
    Catch2::Catch2WithMain
)

if(HAS_CUDA)
    target_sources(kernel_tests PRIVATE test_cuda_monte_carlo.cpp)
    target_link_libraries(kernel_tests PRIVATE cuda_kernels)
endif()

include(Catch)
catch_discover_tests(kernel_tests)