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
    echo "Error: MPI runner not found. Build it first." >&2
    exit 1
fi

echo "Discovering registered MPI benchmarks..."

mapfile -t ALL_BENCHMARKS < <(
    mpirun --oversubscribe -np 1 "$RUNNER" --list |
        sed -n 's/^  \(.*\) (kernel: .*)$/\1/p'
)

if [[ ${#ALL_BENCHMARKS[@]} -eq 0 ]]; then
    echo "Error: No MPI benchmarks were discovered." >&2
    exit 1
fi

echo "Discovered ${#ALL_BENCHMARKS[@]} registrations."
echo "Warm-ups: $WARMUP"
echo "Measured repetitions: $REPETITIONS"
echo "Each benchmark configuration will run in a fresh MPI launch."

for ranks in "${RANK_COUNTS[@]}"; do
    mapfile -t RANK_BENCHMARKS < <(
        printf '%s\n' "${ALL_BENCHMARKS[@]}" |
            grep -E "_np${ranks}$" || true
    )

    if [[ ${#RANK_BENCHMARKS[@]} -eq 0 ]]; then
        echo "Error: No benchmark registrations found for ${ranks} ranks." >&2
        exit 1
    fi

    echo
    echo "========== MPI ranks: $ranks =========="
    echo "Configurations: ${#RANK_BENCHMARKS[@]}"

    for benchmark in "${RANK_BENCHMARKS[@]}"; do
        echo "Running: $benchmark"

        PB_NOTE="cold_start=true; mpi_ranks=${ranks}; configuration=${benchmark}; warmup=${WARMUP}; repetitions=${REPETITIONS}" \
            mpirun --oversubscribe --bind-to none -np "$ranks" \
            "$RUNNER" \
            --baseline "$BASELINE" \
            --only "$benchmark" \
            --warmup "$WARMUP" \
            --repetitions "$REPETITIONS"
    done
done

echo
echo "All cold-start MPI benchmark runs completed successfully."
