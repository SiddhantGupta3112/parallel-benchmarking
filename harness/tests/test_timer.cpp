#include "scoped_timer.h"
#include <thread>
#include <chrono>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Timer reports longer duration for longer sleep") {
    Timer t1;
    t1.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    t1.stop();

    Timer t2;
    t2.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    t2.stop();

    REQUIRE(t2.get_elapsed_time() > t1.get_elapsed_time());
}