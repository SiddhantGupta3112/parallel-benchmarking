#include "matrix_multiply.h"
#include <stdexcept>
#include "build_matrix.h"
#include "seed_values.h"
#include "registrar.h"

std::vector<double> matrix_multiply(
    const std::vector<double>& A,
    const std::vector<double>& B,
    int r1,
    int c1,
    int r2,
    int c2
){
    if (r1 <= 0 || c1 <= 0 || r2 <= 0 || c2 <= 0) {
        throw std::runtime_error("Empty matrix");
    }

    if (c1 != r2) {
        throw std::runtime_error("Invalid matrix sizes");
    }

    if (A.size() != static_cast<std::size_t>(r1) * c1 ||
        B.size() != static_cast<std::size_t>(r2) * c2) {
        throw std::runtime_error("Matrix dimensions do not match data");
    }

    std::vector<double> C(
        static_cast<std::size_t>(r1) * c2, 0.0
    );

    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            for (int k = 0; k < c1; k++) {
                C[i * c2 + j] +=
                    A[i * c1 + k] *
                    B[k * c2 + j];
            }
        }
    }

    return C;
}

std::vector<double> matrix_multiplication_wrapper(
    BenchmarkContext &ctx,
    int r1,
    int c1,
    int r2,
    int c2
) {
    
    std::vector<double> A = std::move(*build_matrix(r1, c1, MATRIX_MULTIPLY_A));
    std::vector<double> B = std::move(*build_matrix(r2, c2, MATRIX_MULTIPLY_B));

    ctx.start_measurement();

    std::vector<double> C = matrix_multiply(A, B, r1, c1, r2, c2);

    ctx.stop_measurement();

    ctx.iterations = 1;
    ctx.problem_size = r1 * c2;
    ctx.number_of_processors = 1;
    ctx.bytes_moved =
        (static_cast<double>(r1) * c1 + static_cast<double>(c1) * c2 + static_cast<double>(r1) * c2) * sizeof(double);
    ctx.flop_count =
        2.0 * r1 * c2 * c1;

    return C;
}

BENCHMARK_TEST(matrix_multiplication_serial_n32x32, matrix_multiplication_n32x32, Paradigm::Serial) {
    matrix_multiplication_wrapper(ctx, 32, 32, 32, 32);
}

BENCHMARK_TEST(matrix_multiplication_serial_n64x64, matrix_multiplication_n64x64, Paradigm::Serial) {
    matrix_multiplication_wrapper(ctx, 64, 64, 64, 64);
}

BENCHMARK_TEST(matrix_multiplication_serial_n128x128, matrix_multiplication_n128x128, Paradigm::Serial) {
    matrix_multiplication_wrapper(ctx, 128, 128, 128, 128);
}

BENCHMARK_TEST(matrix_multiplication_serial_n256x256, matrix_multiplication_n256x256, Paradigm::Serial) {
    matrix_multiplication_wrapper(ctx, 256, 256, 256, 256);
}

BENCHMARK_TEST(matrix_multiplication_serial_n512x512, matrix_multiplication_n512x512, Paradigm::Serial) {
    matrix_multiplication_wrapper(ctx, 512, 512, 512, 512);
}

BENCHMARK_TEST(matrix_multiplication_serial_rect_128x256_256x64, matrix_multiplication_rect_128x256_256x64, Paradigm::Serial) {
    matrix_multiplication_wrapper(ctx, 128, 256, 256, 64);
}

BENCHMARK_TEST(matrix_multiplication_serial_rect_64x128_128x256, matrix_multiplication_rect_64x128_128x256, Paradigm::Serial) {
    matrix_multiplication_wrapper(ctx, 64, 128, 128, 256);
}

BENCHMARK_TEST(matrix_multiplication_serial_rect_256x64_64x128, matrix_multiplication_rect_256x64_64x128, Paradigm::Serial) {
    matrix_multiplication_wrapper(ctx, 256, 64, 64, 128);
}

BENCHMARK_TEST(matrix_multiplication_serial_rect_32x256_256x32, matrix_multiplication_rect_32x256_256x32, Paradigm::Serial) {
    matrix_multiplication_wrapper(ctx, 32, 256, 256, 32);
}
