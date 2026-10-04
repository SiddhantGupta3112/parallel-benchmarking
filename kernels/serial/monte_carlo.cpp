#include <random>

#include "monte_carlo.h"
#include "registrar.h"

double monte_carlo(BenchmarkContext &ctx, int N){
    ctx.problem_size = N;
    ctx.iterations = 1;
    ctx.number_of_processors = 1;

    std::random_device rd;

    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> generator(0, 1);

    int count_inside = 0;
    for(int i = 0; i < N; i++){
        double x = generator(gen);
        double y = generator(gen);

        if(x * x + y * y <= 1){
            count_inside ++;
        }
    }

    double pi = static_cast<double>(count_inside) / N * 4;

    ctx.flop_count = 3 * N;
    ctx.bytes_moved = 2.0 * N * sizeof(double);
    return pi;
}

BENCHMARK_TEST(monte_carlo_serial_n10000, monte_carlo_n10000, Paradigm::Serial) {
    monte_carlo(ctx, 10'000);
}

BENCHMARK_TEST(monte_carlo_serial_n100000, monte_carlo_n100000, Paradigm::Serial) {
    monte_carlo(ctx, 100'000);
}

BENCHMARK_TEST(monte_carlo_serial_n1000000, monte_carlo_n1000000, Paradigm::Serial) {
    monte_carlo(ctx, 1'000'000);
}

BENCHMARK_TEST(monte_carlo_serial_n10000000, monte_carlo_n10000000, Paradigm::Serial) {
    monte_carlo(ctx, 10'000'000);
}



