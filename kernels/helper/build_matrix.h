#ifndef BUILD_MATRIX_H
#define BUILD_MATRIX_H

#include <vector>
#include <memory>
#include <cstdint>

std::unique_ptr<std::vector<double>> build_matrix(int r, int c, uint32_t seed);

#endif // BUILD_MATRIX_H