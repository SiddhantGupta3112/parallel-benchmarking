#include <random>

#include "monte_carlo.h"


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