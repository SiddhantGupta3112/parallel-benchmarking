add_library(cuda_kernels OBJECT
    matrix_multiply.cu
)

target_link_libraries(
        cuda_kernels
        PUBLIC
            harness      
    )