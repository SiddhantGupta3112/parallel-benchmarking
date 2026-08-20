# Parallel Benchmarking Harness — Spec

This document is what you need to read before writing a benchmark. It covers the data model, the registration API, how metrics get collected, the output format, and how to build and add new kernels. If you're adding an OpenMP, MPI, or CUDA kernel, skip to **"Adding a new benchmark."**

---

## 1. Architecture overview

Every benchmark — regardless of paradigm — self-registers into one global registry at program startup, using the `BENCHMARK_TEST` macro. A single runner binary (`benchmark_runner`) discovers everything registered, runs each benchmark through a common metrics-collection pipeline, computes derived metrics (speedup, efficiency, roofline inputs) against a matching serial baseline, and writes everything to a CSV file.

```
kernels/serial/*.cpp   ─┐
kernels/openmp/*.cpp   ─┼─ each self-registers via BENCHMARK_TEST
kernels/mpi/*.cpp      ─┤
kernels/cuda/*.cu      ─┘
        │
        ▼
   benchmark_runner
        │
        ├─ runs every Serial benchmark first (baselines)
        ├─ runs every non-Serial benchmark, measuring against its baseline
        └─ writes results/<timestamp>.csv
```

You only ever write code in `kernels/<paradigm>/`. You never need to touch `harness/` to add a new benchmark.

---

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
    std::optional<double> mpi_comm_overhead_fraction;  // NOT YET IMPLEMENTED — see §7
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
- CPU/wall-clock timing uses `std::chrono::steady_clock` and `getrusage()`.
- **CUDA kernel timing must use CUDA Events** (`cudaEventRecord`/`cudaEventElapsedTime`), not wall-clock — wall-clock around an async kernel launch does not correctly measure GPU execution time. *(Not yet wired into a CUDA kernel — see §7.)*

---

## 6. Output — CSV

Every run produces `results/<timestamp>.csv`, one row per `BenchmarkResult`, columns matching the struct field order. `Paradigm` serializes as its string name (`"OpenMP"`, not an integer). Empty optional fields are empty cells (reads as `NaN` in pandas).

**Interpreting empty GPU-metric cells:**
- Row's paradigm is **not** CUDA → empty is expected, means "not applicable."
- Row's paradigm **is** CUDA → empty means the NVML query failed during collection. Worth investigating, not ignoring.

**Column list:**
```
name,paradigm,problem_size,iterations,number_of_processors,
wall_time_ms,speedup,efficiency,achieved_flops,arithmetic_intensity,
cpu_time_ms,peak_rss_kb,
gpu_utilization_pct,gpu_memory_used_mb,gpu_memory_throughput_gbps,
mpi_comm_overhead_fraction
```

JSON output and CLI flags (`--output=json`, `--only=<name>`) are planned but not yet implemented — see §8.

---

## 7. Known gaps — read before assuming a metric works

- **`mpi_comm_overhead_fraction` is never populated by anything yet.** It exists in the struct for schema stability, but no MPI kernel currently reports communication time. When you write the stencil/Jacobi kernel (the one that actually generates real communication overhead via halo exchange), you'll need to extend `BenchmarkContext` with a way to report time spent in `MPI_Send`/`MPI_Recv` separately from total compute time — this hasn't been designed yet.
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