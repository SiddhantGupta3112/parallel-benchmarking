add_library(serial_kernels OBJECT
    matrix_multiply.cpp
)

target_link_libraries(serial_kernels
    PUBLIC
        harness
)