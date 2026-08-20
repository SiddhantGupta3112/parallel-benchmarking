add_library(openmp_kernels OBJECT
    matrix_multiply.cpp
)

target_link_libraries(openmp_kernels
    PUBLIC
        harness
        OpenMP::OpenMP_CXX
)