# Parallel Benchmarking Harness — Spec

This document is what you need to read before writing a benchmark. It covers the data model, the registration API, how metrics get collected, the output format, and how to build and add new kernels. If you're adding an OpenMP, MPI, or CUDA kernel, skip to **"Adding a new benchmark."**

---

## 1. Architecture overview

The project uses two benchmark runners that share the harness, result schema,
metrics infrastructure, and serial-baseline loading logic.

- `benchmark_runner` executes serial benchmarks and the registered OpenMP
  and CUDA benchmarks. CUDA support is conditional on the build configuration.
  When `--baseline <file>` is supplied, it loads existing serial core
  measurements instead of rerunning the serial benchmarks.
- `mpi_benchmark_runner` is a separate MPI application. It initializes MPI,
  selects benchmarks compatible with the launched MPI rank count, and runs
  the selected MPI configurations against their serial baselines.

The architecture is:

```text
kernels/serial/*.cpp
        |
        v
   Serial baselines
        |
        +------------------------------+
        |                              |
        v                              v
 benchmark_runner              mpi_benchmark_runner
        |                              |
        +-- OpenMP                      +-- MPI matrix multiplication
        +-- CUDA (when enabled)         +-- MPI Monte Carlo
                                       +-- MPI stencil
        |                              |
        +--------------+---------------+
                       |
                       v
              Shared harness
                       |
                       v
             CSV results + metadata
```
## 2. The data model — `BenchmarkResult`

One struct represents every benchmark's result, regardless of paradigm. Fields that don't apply to a given paradigm (e.g. GPU metrics for an OpenMP run) are `std::optional` and left empty rather than using a sentinel value like `-1` or `0`.

```cpp
enum class Paradigm { Serial, OpenMP, MPI, CUDA };

struct BenchmarkResult {
    // Identity / config
    std::string name;              // e.g. "matrix_multiply_openmp"
    Paradigm paradigm;
    int problem_size;
    int iterations;
    int number_of_processors;

    // Always populated
    double wall_time_ms;
    double speedup;                // serial_baseline.wall_time_ms / this.wall_time_ms
    double efficiency;              // speedup / number_of_processors
    double achieved_flops;
    double arithmetic_intensity;    // FLOPs per byte moved — roofline x-axis
    double cpu_time_ms;
    long   peak_rss_kb;

    // Paradigm-specific, nullable
    std::optional<double> gpu_utilization_pct;
    std::optional<double> gpu_memory_used_mb;
    std::optional<double> gpu_memory_throughput_gbps;
    std::optional<double> mpi_comm_overhead_fraction;  // max per-rank MPI-call time / wall time; see §7
};
```

**Why one struct, not one per paradigm:** every downstream consumer (the runner, the CSV writer, eventual dashboard) works against one type. A CUDA-specific struct would mean every consumer needs paradigm-specific branching.

**Why `std::optional`, not a sentinel:** a value like `-1` for "not applicable" is ambiguous the moment a metric could legitimately be negative or zero. `std::optional` makes "no value" a distinct, unambiguous state.

---

## 3. Registering a benchmark — `BENCHMARK_TEST`

```cpp
#include "registrar.h"

BENCHMARK_TEST(matrix_multiply_openmp, matrix_multiply, Paradigm::OpenMP) {
    // your code here — ctx is available implicitly
    ctx.problem_size = N;
    ctx.iterations = 1;
    ctx.number_of_processors = omp_get_max_threads();
    ctx.flop_count = 2.0 * N * N * N;
    ctx.bytes_moved = 3.0 * N * N * sizeof(double);

    // ... actual kernel code ...
}
```

**Arguments:**
- `matrix_multiply_openmp` — the benchmark's unique name. Becomes `BenchmarkResult::name`.
- `matrix_multiply` — the **kernel tag**. This is how the runner finds this benchmark's serial baseline. Every paradigm's version of the same kernel (`matrix_multiply_serial`, `matrix_multiply_openmp`, `matrix_multiply_mpi`, `matrix_multiply_cuda`) must use the **same kernel tag**, or the runner will throw at runtime ("no serial benchmark found for ...").
- `Paradigm::OpenMP` — one of `Paradigm::Serial`, `Paradigm::OpenMP`, `Paradigm::MPI`, `Paradigm::CUDA`.

**How it works, mechanically:** the macro expands into a function definition plus a `static Registrar` object at global/namespace scope. C++ guarantees every global object's constructor runs before `main()` — the `Registrar`'s constructor is what actually adds this benchmark to the global registry, as a side effect of simply being declared. You never call anything to "register" — it happens automatically the moment the compiled object file is linked into the final binary.

**Important:** a benchmark file that is *compiled* but not *linked into `benchmark_runner`* silently registers nothing — there's no error. If your benchmark isn't showing up, check that its filename is listed in the relevant `kernels/<paradigm>/CMakeLists.txt`'s `add_library(... OBJECT ...)` source list.

---

## 4. Reporting data via `BenchmarkContext`

Some values — FLOP count, bytes moved, problem size, iteration count, processor count — are only known to the kernel's own code, not to the runner. `BENCHMARK_TEST` gives your function body an implicit `BenchmarkContext& ctx` to report them:

```cpp
struct BenchmarkContext {
    double flop_count = 0.0;
    double bytes_moved = 0.0;
    int problem_size;
    int number_of_processors;
    int iterations;
};
```

Set these fields **during your benchmark's execution**, in whatever order makes sense — the runner reads them after your function returns. This pattern is modeled on Google Benchmark's `State` object.

**Every benchmark must set all five fields.** `problem_size`/`iterations`/`number_of_processors` have no default values — an uninitialized `int` is indeterminate, not zero.

---

## 5. What the runner does with each benchmark

For every registered benchmark, in order:

1. Construct a `BenchmarkContext`, a `Timer`, a `CpuMetrics`, and (CUDA only) a `LiveMonitor`.
2. Start all collectors.
3. Call your benchmark's function, passing the context.
4. Stop all collectors.
5. Fill `wall_time_ms`, `cpu_time_ms`, `peak_rss_kb` from the collectors.
6. Compute `achieved_flops`/`arithmetic_intensity` from your reported `flop_count`/`bytes_moved` (works for both serial and parallel results).
7. **Non-serial only:** look up the matching serial baseline (by kernel tag) and compute `speedup`/`efficiency` against it. **Throws `std::runtime_error` if no matching serial baseline was ever registered** — every kernel needs a `Paradigm::Serial` version registered, even if it's a naive, unoptimized implementation.
8. **CUDA only:** populate `gpu_utilization_pct` and `gpu_memory_used_mb` (peak-over-time, via a background polling thread — a single end-of-run NVML snapshot would miss a memory spike mid-run).

**Timing specifics:**

- CPU and wall-clock timing use `std::chrono::steady_clock` and `getrusage()`.
- The standard runner executes serial benchmarks first and uses their results as baselines for non-serial benchmarks, unless an existing baseline CSV is supplied with `--baseline <file>`.
- The MPI runner is a separate application. It initializes MPI, selects registrations compatible with the launched rank count, and compares their results against serial baselines.
- **MPI wall-clock time:** Communication performed inside the measured kernel, including scatter, broadcast, gather, and stencil halo exchange, contributes to the measured elapsed time.
- **MPI CPU time:** The reported CPU time aggregates CPU consumption across MPI ranks. It represents aggregate process CPU time, not exclusively useful computation.
- **MPI peak RSS:** The reported value represents the maximum per-process peak RSS across the ranks. RSS is a process-lifetime high-water mark, so an earlier memory-intensive benchmark may affect later measurements made within the same runner process.
- **Monte Carlo randomness:** MPI ranks use rank-specific random-number streams. Results may therefore differ between process counts even when the total sample count is unchanged.
- **CUDA timing:** CUDA event-based timing using `cudaEventRecord()` and `cudaEventElapsedTime()` is not yet implemented. CUDA wall-clock results should therefore be interpreted with this limitation in mind.

---

## 6. Output — CSV

Every run produces `results/<timestamp>.csv`, one row per `BenchmarkResult`, columns matching the struct field order. `Paradigm` serializes as its string name (`"OpenMP"`, not an integer). Empty optional fields are empty cells (reads as `NaN` in pandas).
Each runner invocation writes a CSV file under `results/` using a timestamp and process ID, for example `results/2026-10-28_00-03-24_pid15358.csv`. A metadata sidecar records the run settings and execution status. Each benchmark execution produces one row: warm-up runs use `run_type=warmup`, and measured runs use `run_type=core`. The `kernel_name` field identifies the computation and is used to match serial baselines.

**Interpreting empty GPU-metric cells:**
- Row's paradigm is **not** CUDA → empty is expected, means "not applicable."
- Row's paradigm **is** CUDA → empty means the NVML query failed during collection. Worth investigating, not ignoring.

**Column list:**

```text
name,kernel_name,paradigm,run_type,rep_index,problem_size,iterations,
number_of_processors,wall_time_ms,speedup,efficiency,achieved_flops,
arithmetic_intensity,cpu_time_ms,peak_rss_kb,flop_count,bytes_moved,
gpu_utilization_pct,gpu_memory_used_mb,gpu_memory_throughput_gbps,
mpi_comm_overhead_fraction
```

The standard `benchmark_runner` supports `--baseline <file>`, `--repetitions <n>`, `--warmup <n>`, and `--help`. The MPI-specific `mpi_benchmark_runner` additionally supports `--list` and `--only <pattern>`. The `--only` option matches benchmark names containing the supplied pattern. Run each executable with `--help` for usage details.

---

## 7. Known gaps — read before assuming a metric works
- **`mpi_comm_overhead_fraction` is populated for MPI results.** For each rank, the runner divides the accumulated elapsed time inside instrumented MPI calls by that rank's measured kernel wall time, then reports the maximum fraction across ranks. Instrumented calls cover communication and collective operations in the matrix multiplication, Monte Carlo, and stencil kernels. Because blocking calls can include waiting for other ranks, this metric reflects communication and synchronization overhead, not pure network-transfer time.
- **CUDA kernel-event timing (`cudaEventRecord`/`cudaEventElapsedTime`) is not implemented.** Currently, a CUDA benchmark's `wall_time_ms` comes from the same host-side `Timer` every paradigm uses — which, per §5, is explicitly the wrong way to time an async GPU kernel launch. This needs a small stateful class (similar shape to `Timer`) before CUDA timing numbers can be trusted.
- **GPU memory throughput (`gpu_memory_throughput_gbps`) and CUDA occupancy are not collected.** Throughput is derivable (bytes moved ÷ kernel time vs. hardware peak bandwidth) once event timing exists. Occupancy was deliberately scoped out — it needs profiler-level introspection, not worth the complexity for the current project stage.
- **Load imbalance (per-thread/process variance) was deliberately cut from scope** — real instrumentation cost for a secondary metric, not core to the speedup/efficiency/roofline story.

---

## 8. Build system

- Root `CMakeLists.txt` fetches Catch2, finds OpenMP/MPI (required — every teammate has both), and conditionally enables CUDA (`option(ENABLE_CUDA)`, defaults on, silently disables if no CUDA compiler is found — **no teammate is blocked by not having a GPU**).
- `harness/` builds as a static library (`harness`), containing the registry, all metrics collectors, the runner logic, and the CSV writer.
- Each `kernels/<paradigm>/` directory is an `OBJECT` library (not `STATIC` — object libraries always include every compiled file, unconditionally; static libraries can silently drop object files the linker doesn't see a direct call into, which would mean a real, registered benchmark silently vanishing from the final binary with no error).
- `benchmark_runner` links `harness` + all four kernel libraries (CUDA's conditional on `HAS_CUDA`).

### Adding a new benchmark
1. Write a `.cpp` (or `.cu` for CUDA) file in the right `kernels/<paradigm>/` folder, using `BENCHMARK_TEST`.
2. Add its filename to that folder's `CMakeLists.txt`, in the `add_library(..._kernels OBJECT ...)` source list.
3. If it's a new kernel (not an existing one's new paradigm), make sure a `Paradigm::Serial` version with the same kernel tag exists somewhere too.
4. Rebuild: `cmake --build build`.
5. Run `./build/benchmark_runner` and check the generated CSV.

You never need to edit anything under `harness/` to add an ordinary benchmark.

### Building without CUDA
```
cmake -S . -B build -DENABLE_CUDA=off
```
Verified to configure, build, and pass the full test suite cleanly with CUDA disabled.

---

## 9. Testing conventions, if you're touching harness internals

- Harness logic (registry, metrics collectors' arithmetic, derived-metrics computation) has real Catch2 test coverage in `harness/tests/`. If you change harness behavior, update or add tests.
- Anything using the global registry (`get_registry()`) in a test needs an **isolated local `Registry`** instead — construct one directly, register benchmarks via `Registry::register_benchmark(...)` (not `BENCHMARK_TEST`, which is hardcoded to the global registry), and pass it explicitly to whichever function you're testing (`get_serial_benchmarks`, `run_serial_benchmarks`, `run_non_serial_benchmarks` all accept an optional `Registry&` for exactly this reason).
- Hardware-dependent collectors (NVML, CUDA) are hard to unit test directly. `LiveMonitor` solves this via dependency injection — its constructor takes a `std::function<double()>` value source, defaulting to the real NVML reader, swappable for a fake in tests. Follow this pattern for any new hardware-dependent collector.

---