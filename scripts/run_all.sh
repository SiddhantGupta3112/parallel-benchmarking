#!/usr/bin/env bash
# run_all.sh - runs the whole benchmark chain unattended.
#
#   scripts/run_all.sh              full run
#   SMOKE=1 scripts/run_all.sh      tiny repetition counts, to test the chain first
#
# Chain:
#   1. benchmark_runner      serial + OpenMP (+ CUDA if built in); its CSV becomes the baseline
#   2. mpi_benchmark_runner  one mpirun per rank count, against that baseline CSV
#   3. summary of exit codes and the CSV files produced
#
# No "set -e": a failing step is logged and the chain carries on.

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR" || exit 1

# ---------------------------------------------------------------- configuration
BASE_WARMUP="${BASE_WARMUP:-20}"
BASE_REPS="${BASE_REPS:-200}"
MPI_WARMUP="${MPI_WARMUP:-10}"
MPI_REPS="${MPI_REPS:-50}"
RANKS="${RANKS:-1 2 4 8 12 16 20}"
RUNNER_MAIN="${RUNNER_MAIN:-./build/benchmark_runner}"
RUNNER_MPI="${RUNNER_MPI:-./build/mpi_benchmark_runner}"
MPI_ONLY="${MPI_ONLY:-}"                 # optional name filter for the MPI runner
CHAIN_NOTE="${CHAIN_NOTE:-run_all chain}"
ALLOW_NO_CUDA="${ALLOW_NO_CUDA:-0}"

if [[ "${SMOKE:-0}" == "1" ]]; then
    BASE_WARMUP=1; BASE_REPS=2; MPI_WARMUP=1; MPI_REPS=2
    CHAIN_NOTE="${CHAIN_NOTE} (SMOKE TEST)"
fi

# ---------------------------------------------------------------- logging
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="logs/run_all_${STAMP}"
mkdir -p "$LOG_DIR" results
MARKER="$LOG_DIR/.start"
touch "$MARKER"
STATUS_FILE="$LOG_DIR/.statuses"
: > "$STATUS_FILE"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/chain.log"; }
record() { echo "$1 | exit code $2 | ${3}s" >> "$STATUS_FILE"; }

# ---------------------------------------------------------------- preflight
log "Chain started. Logs in $LOG_DIR"
log "Baseline run: warm-up=$BASE_WARMUP repetitions=$BASE_REPS"
log "MPI sweep:    warm-up=$MPI_WARMUP repetitions=$MPI_REPS ranks=[$RANKS] only='${MPI_ONLY}'"

for exe in "$RUNNER_MAIN" "$RUNNER_MPI"; do
    if [[ ! -x "$exe" ]]; then
        log "ABORT: $exe not found or not executable. Build first."
        exit 1
    fi
done
command -v mpirun >/dev/null 2>&1 || { log "ABORT: mpirun not found in PATH"; exit 1; }

if [[ -f build/CMakeCache.txt ]]; then
    cuda_state="$(grep -E '^ENABLE_CUDA:BOOL=' build/CMakeCache.txt | cut -d= -f2)"
    log "build/CMakeCache.txt: ENABLE_CUDA=${cuda_state:-not set}"
    case "${cuda_state,,}" in
        off|false|no|n|0)
            if [[ "$ALLOW_NO_CUDA" != "1" ]]; then
                log "ABORT: CUDA is switched off in this build (set ALLOW_NO_CUDA=1 to run without it)."
                exit 1
            fi ;;
    esac
    if ! grep -qE '^CMAKE_CUDA_COMPILER:[A-Z]+=.+' build/CMakeCache.txt; then
        log "WARNING: no CUDA compiler recorded in build/CMakeCache.txt; CUDA may not be built in."
    fi
fi

if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
    log "WARNING: git working tree is dirty (the run metadata will record this)."
fi
log "Load average at start: $(cut -d' ' -f1-3 /proc/loadavg)"

# ---------------------------------------------------------------- step 1: baseline run
log "STEP 1: benchmark_runner (serial + OpenMP + CUDA)"
t0=$SECONDS
PB_NOTE="${CHAIN_NOTE}; step=1 baseline run; warmup=${BASE_WARMUP}; repetitions=${BASE_REPS}" \
    "$RUNNER_MAIN" --warmup "$BASE_WARMUP" --repetitions "$BASE_REPS" 2>&1 \
    | tee "$LOG_DIR/step1_baseline.log"
rc1=${PIPESTATUS[0]}
record "step 1 baseline run" "$rc1" "$((SECONDS - t0))"
log "STEP 1 finished with exit code $rc1"

# find the CSV that step 1 wrote
BASELINE="$(grep -h 'Results written to:' "$LOG_DIR/step1_baseline.log" | tail -1 \
            | sed 's/^Results written to: *//' | tr -d '\r')"
if [[ -z "$BASELINE" || ! -f "$BASELINE" ]]; then
    BASELINE="$(find results -maxdepth 1 -name '*.csv' -newer "$MARKER" -printf '%T@ %p\n' \
                | sort -n | tail -1 | cut -d' ' -f2-)"
fi
log "Baseline CSV: ${BASELINE:-NOT FOUND}"

# ---------------------------------------------------------------- step 2: MPI sweep
if [[ -n "$BASELINE" && -f "$BASELINE" ]]; then
    for ranks in $RANKS; do
        log "STEP 2: MPI with $ranks ranks"
        t0=$SECONDS
        PB_NOTE="${CHAIN_NOTE}; step=2 mpi; mpi_ranks=${ranks}; baseline=${BASELINE}; warmup=${MPI_WARMUP}; repetitions=${MPI_REPS}" \
            mpirun --oversubscribe --bind-to none -np "$ranks" \
            "$RUNNER_MPI" --baseline "$BASELINE" \
            --warmup "$MPI_WARMUP" --repetitions "$MPI_REPS" \
            ${MPI_ONLY:+--only "$MPI_ONLY"} 2>&1 \
            | tee "$LOG_DIR/step2_mpi_np${ranks}.log"
        rc=${PIPESTATUS[0]}
        record "step 2 mpi ranks=${ranks}" "$rc" "$((SECONDS - t0))"
        log "STEP 2 ($ranks ranks) finished with exit code $rc"
    done
else
    log "STEP 2 SKIPPED: no baseline CSV."
    record "step 2 mpi (skipped, no baseline)" "1" "0"
fi

# ---------------------------------------------------------------- step 3: summary
failures="$(grep -vc 'exit code 0 ' "$STATUS_FILE")"
{
    echo "=============== run_all chain summary ==============="
    echo "started : $STAMP"
    echo "finished: $(date +%Y-%m-%d_%H-%M-%S)"
    echo "baseline CSV: ${BASELINE:-NOT FOUND}"
    if [[ -n "$BASELINE" && -f "$BASELINE" ]]; then
        echo "rows per paradigm in the baseline CSV (CUDA missing here means it did not run):"
        tail -n +2 "$BASELINE" | cut -d, -f3 | tr -d '"' | sort | uniq -c | sed 's/^/  /'
    fi
    echo
    echo "steps:"
    sed 's/^/  /' "$STATUS_FILE"
    echo
    echo "CSV files written during this chain:"
    find results -maxdepth 1 -name '*.csv' -newer "$MARKER" | sort | sed 's/^/  /'
    echo
    if [[ "$failures" -eq 0 ]]; then
        echo "ALL STEPS EXITED WITH CODE 0"
    else
        echo "FINISHED WITH $failures FAILED STEP(S) - check the logs in $LOG_DIR"
    fi
} | tee "$LOG_DIR/summary.txt"

[[ "$failures" -eq 0 ]]