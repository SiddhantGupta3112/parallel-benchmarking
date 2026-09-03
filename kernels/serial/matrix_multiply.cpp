#include "matrix_multiply.h"
#include <stdexcept>

std::vector<double> matrix_multiply(
    BenchmarkContext &ctx,
    const std::vector<double>& A,
    const std::vector<double>& B,
    int r1,
    int c1,
    int r2,
    int c2
) {
    if (r1 <= 0 || c1 <= 0 || r2 <= 0 || c2 <= 0) {
        throw std::runtime_error("Empty matrix");
    }

    if (c1 != r2) {
        throw std::runtime_error("Invalid matrix sizes");
    }

    if (A.size() != static_cast<std::size_t>(r1 * c1) ||
        B.size() != static_cast<std::size_t>(r2 * c2)) {
        throw std::runtime_error("Matrix dimensions do not match data");
    }

    std::vector<double> C(r1 * c2, 0.0);

    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            for (int k = 0; k < c1; k++) {
                C[i * c2 + j] +=
                    A[i * c1 + k] *
                    B[k * c2 + j];
            }
        }
    }

    ctx.iterations = 1;
    ctx.problem_size = r1 * c2;
    ctx.number_of_processors = 1;
    ctx.bytes_moved =
        (static_cast<double>(r1) * c1 + static_cast<double>(c1) * c2 + static_cast<double>(r1) * c2) * sizeof(double);
    ctx.flop_count =
        2.0 * r1 * c2 * c1;

    return C;
}