add_library(helper STATIC
    build_matrix.cpp
    build_stencil_input.cpp
)

target_include_directories(helper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})