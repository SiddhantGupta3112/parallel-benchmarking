
#include <random>
#include <stdexcept>

#include "monte_carlo.h"
#include "registrar.h"
#include "seed_values.h"

double monte_carlo(int N) {
    if (N <= 0) {
        throw std::invalid_argument("N must be a positive integer.");
    }

    std::mt19937 gen(MONTE_CARLO_SEED);
    std::uniform_real_distribution<double> generator(0.0, 1.0);

    int count_inside = 0;

    for (int i = 0; i < N; i++) {
        double x = generator(gen);
        double y = generator(gen);

        if (x * x + y * y <= 1.0) {
            count_inside++;
        }
    }

    return static_cast<double>(count_inside) / N * 4.0;
}

double monte_carlo_wrapper(BenchmarkContext& ctx, int N) {
    if (N <= 0) {
        throw std::invalid_argument("N must be a positive integer.");
    }

    ctx.problem_size = N;
    ctx.iterations = 1;
    ctx.number_of_processors = 1;

    ctx.start_measurement();

    double pi = monte_carlo(N);

    ctx.stop_measurement();

    ctx.flop_count = 3.0 * N;
    ctx.bytes_moved = 2.0 * N * sizeof(double);

    return pi;
}

BENCHMARK_TEST(monte_carlo_serial_n10000, monte_carlo_n10000, Paradigm::Serial) {
    monte_carlo_wrapper(ctx, 10'000);
}

BENCHMARK_TEST(monte_carlo_serial_n20000, monte_carlo_n20000, Paradigm::Serial) {
    monte_carlo_wrapper(ctx, 20'000);
}

BENCHMARK_TEST(monte_carlo_serial_n50000, monte_carlo_n50000, Paradigm::Serial) {
    monte_carlo_wrapper(ctx, 50'000);
}

BENCHMARK_TEST(monte_carlo_serial_n100000, monte_carlo_n100000, Paradigm::Serial) {
    monte_carlo_wrapper(ctx, 100'000);
}

BENCHMARK_TEST(monte_carlo_serial_n200000, monte_carlo_n200000, Paradigm::Serial) {
    monte_carlo_wrapper(ctx, 200'000);
}

BENCHMARK_TEST(monte_carlo_serial_n500000, monte_carlo_n500000, Paradigm::Serial) {
    monte_carlo_wrapper(ctx, 500'000);
}

BENCHMARK_TEST(monte_carlo_serial_n1000000, monte_carlo_n1000000, Paradigm::Serial) {
    monte_carlo_wrapper(ctx, 1'000'000);
}

BENCHMARK_TEST(monte_carlo_serial_n2000000, monte_carlo_n2000000, Paradigm::Serial) {
    monte_carlo_wrapper(ctx, 2'000'000);
}

BENCHMARK_TEST(monte_carlo_serial_n5000000, monte_carlo_n5000000, Paradigm::Serial) {
    monte_carlo_wrapper(ctx, 5'000'000);
}

BENCHMARK_TEST(monte_carlo_serial_n10000000, monte_carlo_n10000000, Paradigm::Serial) {
    monte_carlo_wrapper(ctx, 10'000'000);
}

BENCHMARK_TEST(monte_carlo_serial_n20000000, monte_carlo_n20000000, Paradigm::Serial) {
    monte_carlo_wrapper(ctx, 20'000'000);
}

BENCHMARK_TEST(monte_carlo_serial_n50000000, monte_carlo_n50000000, Paradigm::Serial) {
    monte_carlo_wrapper(ctx, 50'000'000);
}

BENCHMARK_TEST(monte_carlo_serial_n100000000, monte_carlo_n100000000, Paradigm::Serial) {
    monte_carlo_wrapper(ctx, 100'000'000);
}