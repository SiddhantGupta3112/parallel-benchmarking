#include "openmp_matrix_multiply.h"

#include <stdexcept>

#include "build_matrix.h"
#include "openmp_helper.h"
#include "registrar.h"
#include "seed_values.h"

std::vector<double> openmp_matrix_multiply(
    const std::vector<double>& A,
    const std::vector<double>& B,
    int r1,
    int c1,
    int r2,
    int c2,
    int num_threads
) {
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

    OpenMPThreadScope thread_scope(num_threads);

    // Same i-k-j loop order as the serial kernel. Only the outer i loop is
    // parallel: each thread owns whole rows of C, so no two threads ever write
    // the same element and no synchronisation is needed. A, B and C are shared,
    // i/k/j are private (declared inside / loop variables).
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < r1; ++i) {
        for (int k = 0; k < c1; ++k) {
            for (int j = 0; j < c2; ++j) {
                C[i * c2 + j] +=
                    A[i * c1 + k] * B[k * c2 + j];
            }
        }
    }

    return C;
}

namespace {

std::vector<double> openmp_matrix_multiplication_wrapper(
    BenchmarkContext &ctx,
    int r1,
    int c1,
    int r2,
    int c2,
    int num_threads
) {
    std::vector<double> A = std::move(*build_matrix(r1, c1, MATRIX_MULTIPLY_A_SEED));
    std::vector<double> B = std::move(*build_matrix(r2, c2, MATRIX_MULTIPLY_B_SEED));

    // Untimed: find out how many threads a parallel region really gets.
    const int threads = get_number_of_threads(num_threads);

    ctx.start_measurement();

    std::vector<double> C = openmp_matrix_multiply(A, B, r1, c1, r2, c2, num_threads);

    ctx.stop_measurement();

    ctx.iterations = 1;
    ctx.problem_size = r1 * c2;
    ctx.number_of_processors = threads;
    ctx.bytes_moved =
        (static_cast<double>(r1) * c1 + static_cast<double>(c1) * c2 + static_cast<double>(r1) * c2) * sizeof(double);
    ctx.flop_count =
        2.0 * r1 * c2 * c1;

    return C;
}

} // namespace

// One registration per thread count (see OPENMP_THREAD_SWEEP in
// openmp_helper.h). kernel_name matches the serial benchmark exactly.
#define OPENMP_MATMUL_BENCHMARK(size_tag, r1, c1, r2, c2, T)                      \
    BENCHMARK_TEST(matrix_multiplication_OpenMP_##size_tag##_t##T,                \
                   matrix_multiplication_##size_tag, Paradigm::OpenMP) {          \
        openmp_matrix_multiplication_wrapper(ctx, r1, c1, r2, c2, T);             \
    }

OPENMP_THREAD_SWEEP(OPENMP_MATMUL_BENCHMARK, n32x32, 32, 32, 32, 32)
OPENMP_THREAD_SWEEP(OPENMP_MATMUL_BENCHMARK, n64x64, 64, 64, 64, 64)
OPENMP_THREAD_SWEEP(OPENMP_MATMUL_BENCHMARK, n128x128, 128, 128, 128, 128)
OPENMP_THREAD_SWEEP(OPENMP_MATMUL_BENCHMARK, n256x256, 256, 256, 256, 256)
OPENMP_THREAD_SWEEP(OPENMP_MATMUL_BENCHMARK, n384x384, 384, 384, 384, 384)
OPENMP_THREAD_SWEEP(OPENMP_MATMUL_BENCHMARK, n512x512, 512, 512, 512, 512)
OPENMP_THREAD_SWEEP(OPENMP_MATMUL_BENCHMARK, n768x768, 768, 768, 768, 768)
OPENMP_THREAD_SWEEP(OPENMP_MATMUL_BENCHMARK, n1024x1024, 1024, 1024, 1024, 1024)
OPENMP_THREAD_SWEEP(OPENMP_MATMUL_BENCHMARK, n1536x1536, 1536, 1536, 1536, 1536)
OPENMP_THREAD_SWEEP(OPENMP_MATMUL_BENCHMARK, n2048x2048, 2048, 2048, 2048, 2048)

OPENMP_THREAD_SWEEP(OPENMP_MATMUL_BENCHMARK, rect_128x256_256x64, 128, 256, 256, 64)
OPENMP_THREAD_SWEEP(OPENMP_MATMUL_BENCHMARK, rect_64x128_128x256, 64, 128, 128, 256)
OPENMP_THREAD_SWEEP(OPENMP_MATMUL_BENCHMARK, rect_256x64_64x128, 256, 64, 64, 128)
OPENMP_THREAD_SWEEP(OPENMP_MATMUL_BENCHMARK, rect_32x256_256x32, 32, 256, 256, 32)
