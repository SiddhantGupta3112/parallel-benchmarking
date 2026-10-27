#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASELINE="${1:-results/2026-10-20_00-38-03.csv}"
WARMUP="${2:-10}"
REPETITIONS="${3:-50}"

RUNNER="./build/mpi_benchmark_runner"
RANK_COUNTS=(1 2 4 8 12 16 20)

if [[ ! -f "$BASELINE" ]]; then
    echo "Error: Baseline CSV not found: $BASELINE" >&2
    exit 1
fi

if [[ ! -x "$RUNNER" ]]; then
    echo "Error: MPI runner not found. Build it first with:" >&2
    echo "  cmake --build build -j\"\$(nproc)\"" >&2
    exit 1
fi

echo "MPI Rank Sweep"
echo "Baseline: $BASELINE"
echo "Warm-ups: $WARMUP"
echo "Measured repetitions: $REPETITIONS"
echo "Rank counts: ${RANK_COUNTS[*]}"

for ranks in "${RANK_COUNTS[@]}"; do
    echo
    echo "=========================================="
    echo "Running MPI benchmarks with $ranks ranks"
    echo "=========================================="

    PB_NOTE="mpi_ranks=${ranks}; mpirun_flags=--oversubscribe --bind-to none; warmup=${WARMUP}; repetitions=${REPETITIONS}" \
        mpirun --oversubscribe --bind-to none -np "$ranks" \
        "$RUNNER" \
        --baseline "$BASELINE" \
        --warmup "$WARMUP" \
        --repetitions "$REPETITIONS"
done

echo
echo "All MPI rank-sweep runs completed successfully."
