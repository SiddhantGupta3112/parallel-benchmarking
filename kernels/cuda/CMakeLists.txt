add_library(cuda_kernels OBJECT
    matrix_multiply.cu
    monte_carlo.cu
    helper.cu
    stencil.cu
)

target_include_directories(cuda_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
    cuda_kernels
    PUBLIC
        harness
)