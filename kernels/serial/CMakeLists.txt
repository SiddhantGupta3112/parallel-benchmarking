add_library(serial_kernels OBJECT
    monte_carlo.cpp
    matrix_multiply.cpp
    stencil.cpp
)

target_include_directories(serial_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(serial_kernels
    PUBLIC
        harness
)