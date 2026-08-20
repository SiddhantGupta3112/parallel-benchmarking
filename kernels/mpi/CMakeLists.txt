add_library(mpi_kernels OBJECT
    matrix_multiply.cpp
)

target_link_libraries(mpi_kernels
    PUBLIC
        harness
        MPI::MPI_CXX
)