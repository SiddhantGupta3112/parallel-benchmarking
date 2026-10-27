#include "build_matrix.h"
#include <random>

std::unique_ptr<std::vector<double>> build_matrix(int r, int c, uint32_t seed) {
    int total_elements = r * c;

    std::mt19937 gen(seed);
    
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    std::unique_ptr<std::vector<double>> heapVector = std::make_unique<std::vector<double>>(total_elements);

    for(int i = 0; i < total_elements; i++) {
        (*heapVector)[i] = dis(gen);
    }

    return heapVector;
}
