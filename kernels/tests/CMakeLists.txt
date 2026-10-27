add_executable(
    kernel_tests
    test_serial_monte_carlo.cpp
    test_serial_matrix_multiply.cpp
    test_serial_stencil.cpp
    test_build_matrix_helper.cpp
    test_build_stencil_input.cpp
    test_openmp_helper.cpp
    test_openmp_monte_carlo.cpp
    test_openmp_matrix_multiply.cpp
    test_openmp_stencil.cpp
)

target_link_libraries(
    kernel_tests
    PRIVATE
    serial_kernels
    openmp_kernels
    helper
    Catch2::Catch2WithMain
)

if(HAS_CUDA)
    target_sources(kernel_tests PRIVATE 
    test_cuda_monte_carlo.cpp
    test_cuda_matrix_multiply.cpp
    test_cuda_stencil.cpp
    )
    target_link_libraries(kernel_tests PRIVATE cuda_kernels)
endif()

include(Catch)
catch_discover_tests(kernel_tests)
