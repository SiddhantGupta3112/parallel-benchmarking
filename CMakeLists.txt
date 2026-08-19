cmake_minimum_required(VERSION 3.14)
project(parallel_benchmarking LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.2
)
FetchContent_MakeAvailable(Catch2)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)
add_subdirectory(harness)

add_subdirectory(harness/tests)