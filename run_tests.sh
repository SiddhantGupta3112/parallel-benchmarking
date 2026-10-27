#!/bin/bash
set -euo pipefail

cmake -S . -B build -DENABLE_CUDA=off
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
