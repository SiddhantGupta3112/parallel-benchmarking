add_executable(
    kernel_tests
    test_serial_monte_carlo.cpp
    test_serial_matrix_multiply.cpp
    test_serial_stencil.cpp
    test_build_matrix_helper.cpp
    test_build_stencil_input.cpp
    test_openmp_helper.cpp
    test_openmp_matrix_multiply.cpp
    test_openmp_monte_carlo.cpp
    test_openmp_stencil.cpp
)

target_link_libraries(
    kernel_tests
    PRIVATE
        serial_kernels
        helper
        Catch2::Catch2WithMain
        openmp_kernels
)

if(HAS_CUDA)
    target_sources(
        kernel_tests
        PRIVATE
            test_cuda_monte_carlo.cpp
            test_cuda_matrix_multiply.cpp
            test_cuda_stencil.cpp
    )

    target_link_libraries(kernel_tests PRIVATE cuda_kernels)
endif()

include(Catch)
catch_discover_tests(kernel_tests)

set(MPI_TEST_KERNELS
    matrix_multiply
    monte_carlo
    stencil
)

foreach(kernel IN LISTS MPI_TEST_KERNELS)
    set(test_target mpi_${kernel}_tests)

    add_executable(
        ${test_target}
        test_mpi_${kernel}.cpp
    )

    target_link_libraries(
        ${test_target}
        PRIVATE
            mpi_kernels
            serial_kernels
            Catch2::Catch2
            MPI::MPI_CXX
    )

    if(kernel STREQUAL "matrix_multiply")
        target_link_libraries(${test_target} PRIVATE helper)
    endif()

    foreach(np IN ITEMS 1 2 3 4 7 8)
        add_test(
            NAME mpi_${kernel}_np${np}
            COMMAND
                ${MPIEXEC_EXECUTABLE}
                ${MPIEXEC_NUMPROC_FLAG} ${np}
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:${test_target}>
                ${MPIEXEC_POSTFLAGS}
        )
    endforeach()
endforeach()