#include "build_stencil_input.h"
#include <random>
#include <stdexcept>

std::unique_ptr<std::vector<double>> build_stencil_input(
    int r,
    int c,
    uint32_t seed
) {
    if (r < 3 || c < 3) {
        throw std::runtime_error(
            "Stencil dimensions must be at least 3x3"
        );
    }

    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    double boundary_value = dis(gen);

    auto matrix = std::make_unique<std::vector<double>>(
        static_cast<std::size_t>(r) * c, 0.0
    );

    for (int j = 0; j < c; ++j) {
        (*matrix)[j] = boundary_value;
        (*matrix)[(r - 1) * c + j] = boundary_value;
    }

    for (int i = 0; i < r; ++i) {
        (*matrix)[i * c] = boundary_value;
        (*matrix)[i * c + (c - 1)] = boundary_value;
    }

    return matrix;
}