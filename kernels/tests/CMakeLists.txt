add_executable(
    kernel_tests
    test_serial_monte_carlo.cpp
    test_serial_matrix_multiply.cpp
    test_serial_stencil.cpp
    test_build_matrix_helper.cpp
    test_build_stencil_input.cpp
)

target_link_libraries(
    kernel_tests
    PRIVATE
    serial_kernels
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
add_executable(
    mpi_matrix_multiply_tests
    test_mpi_matrix_multiply.cpp
)

target_link_libraries(
    mpi_matrix_multiply_tests
    PRIVATE
        mpi_kernels
        serial_kernels
        helper
        Catch2::Catch2
        MPI::MPI_CXX
)
add_executable(
    mpi_monte_carlo_tests
    test_mpi_monte_carlo.cpp
)

target_link_libraries(
    mpi_monte_carlo_tests
    PRIVATE
        mpi_kernels
        Catch2::Catch2
        MPI::MPI_CXX
)

add_executable(
    mpi_stencil_tests
    test_mpi_stencil.cpp
)

target_link_libraries(
    mpi_stencil_tests
    PRIVATE
        mpi_kernels
        serial_kernels
        Catch2::Catch2
        MPI::MPI_CXX
)
