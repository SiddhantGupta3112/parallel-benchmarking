add_library(openmp_kernels OBJECT
    matrix_multiply.cpp
    monte_carlo.cpp
    openmp_helper.cpp
    stencil.cpp
)

target_include_directories(openmp_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(openmp_kernels
    PUBLIC
        harness
        helper
        OpenMP::OpenMP_CXX
)
