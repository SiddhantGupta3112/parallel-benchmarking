#ifndef BUILD_STENCIL_INPUT_H
#define BUILD_STENCIL_INPUT_H

#include <vector>
#include <memory>
#include <cstdint>

std::unique_ptr<std::vector<double>> build_stencil_input(int r, int c, uint32_t seed);

#endif