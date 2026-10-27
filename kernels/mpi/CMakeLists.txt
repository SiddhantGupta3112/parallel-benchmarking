add_library(mpi_kernels OBJECT
    matrix_multiply.cpp
    monte_carlo.cpp
    stencil.cpp
)

target_link_libraries(mpi_kernels
    PUBLIC
        harness
        MPI::MPI_CXX
        helper
)
target_include_directories(mpi_kernels
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
