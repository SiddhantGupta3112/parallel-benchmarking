#include "metadata_collector.h"

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------------------
// Running shell commands
// ---------------------------------------------------------------------------

struct CommandResult {
    bool ok = false;          // started, and exited with status 0
    std::string output;       // '\r' removed, trailing newlines removed
};

std::string clean_output(std::string text) {
    text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
    while (!text.empty() && text.back() == '\n') {
        text.pop_back();
    }
    return text;
}

std::string first_line(const std::string& text) {
    std::string line = text.substr(0, text.find('\n'));
    if (line.size() > 160) {
        line = line.substr(0, 160) + "...";
    }
    return line;
}

std::string head_lines(const std::string& text, int count) {
    std::istringstream in(text);
    std::ostringstream out;
    std::string line;
    for (int i = 0; i < count && std::getline(in, line); ++i) {
        out << line << '\n';
    }
    return clean_output(out.str());
}

std::string drop_lines_starting_with(const std::string& text,
                                     const std::vector<std::string>& prefixes) {
    std::istringstream in(text);
    std::ostringstream out;
    std::string line;
    while (std::getline(in, line)) {
        bool drop = false;
        for (const auto& prefix : prefixes) {
            if (line.rfind(prefix, 0) == 0) {
                drop = true;
            }
        }
        if (!drop) {
            out << line << '\n';
        }
    }
    return clean_output(out.str());
}

// The command is wrapped in a subshell so that "2>&1" also covers pipelines
// such as "tool --version | head -n 1".
CommandResult run_command(const std::string& command, bool merge_stderr = true) {
    CommandResult result;
    const std::string full =
        "(" + command + ") " + (merge_stderr ? "2>&1" : "2>/dev/null");

    FILE* pipe = popen(full.c_str(), "r");
    if (!pipe) {
        return result;
    }

    std::array<char, 4096> buffer{};
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe)) {
        result.output += buffer.data();
    }

    const int status = pclose(pipe);
    result.ok = status != -1 && WIFEXITED(status) && WEXITSTATUS(status) == 0;
    result.output = clean_output(std::move(result.output));
    return result;
}

// Success with empty output is a legitimate answer (for example a clean
// "git status --porcelain"), so it is reported as "(empty)", not as a failure.
std::string value_of(const CommandResult& result) {
    if (result.ok) {
        return result.output.empty() ? "(empty)" : result.output;
    }
    if (result.output.empty()) {
        return "Unavailable";
    }
    return "Unavailable (" + first_line(result.output) + ")";
}

std::string value_of_head(CommandResult result, int lines) {
    if (result.ok) {
        result.output = head_lines(result.output, lines);
    }
    return value_of(result);
}

std::string shell_quote(const std::string& text) {
    std::string quoted = "'";
    for (const char c : text) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

std::string upper(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return text;
}

bool is_unavailable(const std::string& value) {
    return value.rfind("Unavailable", 0) == 0 || value == "(empty)";
}

std::string env_value(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string("(unset)");
}

bool is_wsl() {
    std::ifstream file("/proc/version");
    std::string text((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text.find("microsoft") != std::string::npos;
}

std::string executable_path() {
    std::error_code ec;
    const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    return ec ? std::string("Unavailable") : exe.string();
}

std::string metadata_filename(const std::string& csv_filename) {
    const fs::path path(csv_filename);
    return (path.parent_path() / (path.stem().string() + ".meta.txt")).string();
}

void kv(std::ostream& out, const std::string& key, const std::string& value) {
    out << key << '=' << value << '\n';
}

void block(std::ostream& out, const std::string& title, const std::string& text) {
    out << title << ":\n";
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        out << "  " << line << '\n';
    }
}

void section(std::ostream& out, const std::string& name) {
    out << "\n[" << name << "]\n";
}

// ---------------------------------------------------------------------------
// CMake information
// ---------------------------------------------------------------------------

// Cache lines look like "KEY:TYPE=value".
std::string read_cache_value(const fs::path& cache_path, const std::string& key) {
    std::ifstream file(cache_path);
    if (!file) {
        return "Unavailable (CMakeCache.txt not found)";
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#' || line.compare(0, 2, "//") == 0) {
            continue;
        }
        const auto colon = line.find(':');
        const auto equals = line.find('=');
        if (colon == std::string::npos || equals == std::string::npos ||
            colon > equals) {
            continue;
        }
        if (line.substr(0, colon) == key) {
            const std::string value = line.substr(equals + 1);
            return value.empty() ? "(empty)" : value;
        }
    }
    return "Unavailable (not set in cache)";
}

// The flags actually used to compile each target live in
// <build>/**/CMakeFiles/<target>.dir/flags.make (Makefile generator).
std::string collect_target_flags(const fs::path& build_dir) {
    static const std::vector<std::string> targets = {
        "benchmark_runner", "harness",        "serial_kernels",
        "cuda_kernels",     "openmp_kernels", "mpi_kernels",
        "helper"};
    static const std::vector<std::string> wanted = {
        "CXX_FLAGS", "CXX_DEFINES", "CUDA_FLAGS", "CUDA_DEFINES"};

    std::error_code ec;
    if (!fs::is_directory(build_dir, ec)) {
        return "Unavailable (build directory not found)";
    }

    std::map<std::string, std::vector<std::string>> lines_by_target;

    try {
        for (const auto& entry : fs::recursive_directory_iterator(build_dir)) {
            if (!entry.is_regular_file() ||
                entry.path().filename() != "flags.make") {
                continue;
            }
            const std::string dir = entry.path().parent_path().filename().string();
            for (const auto& target : targets) {
                if (dir != target + ".dir") {
                    continue;
                }
                std::ifstream file(entry.path());
                std::string line;
                while (std::getline(file, line)) {
                    for (const auto& key : wanted) {
                        if (line.rfind(key + " =", 0) == 0) {
                            lines_by_target[target].push_back(line);
                        }
                    }
                }
            }
        }
    } catch (const fs::filesystem_error&) {
        // keep whatever was collected before the error
    }

    if (lines_by_target.empty()) {
        // Ninja and other generators: ask the build tool for the compile commands.
        std::error_code exists_ec;
        if (fs::exists(build_dir / "build.ninja", exists_ec)) {
            const CommandResult commands = run_command(
                "ninja -C " + shell_quote(build_dir.string()) +
                    " -t commands benchmark_runner",
                false);
            if (commands.ok && !commands.output.empty()) {
                std::istringstream in(commands.output);
                std::ostringstream shortened;
                std::string line;
                int kept = 0;
                while (kept < 40 && std::getline(in, line)) {
                    shortened << (line.size() > 500 ? line.substr(0, 500) + "..." : line)
                              << '\n';
                    ++kept;
                }
                return "(from ninja -t commands benchmark_runner, lines shortened)\n" +
                       clean_output(shortened.str());
            }
        }
        return "Unavailable (no flags.make files and no usable build.ninja)";
    }

    std::ostringstream out;
    for (const auto& [target, lines] : lines_by_target) {
        for (const auto& line : lines) {
            out << target << ": " << line << '\n';
        }
    }
    return clean_output(out.str());
}

// ---------------------------------------------------------------------------
// Git information
// ---------------------------------------------------------------------------

bool is_source_file(const fs::path& path) {
    static const std::vector<std::string> extensions = {
        ".cpp", ".cc", ".h", ".hpp", ".cu", ".cuh", ".cmake"};
    if (path.filename() == "CMakeLists.txt") {
        return true;
    }
    return std::find(extensions.begin(), extensions.end(),
                     path.extension().string()) != extensions.end();
}

// A binary older than a tracked source file may not match the sources the
// recorded commit describes.
std::string stale_binary_check(const std::string& git_prefix,
                               const fs::path& source_dir) {
    std::error_code ec;
    const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    if (ec) {
        return "Unavailable";
    }
    const auto exe_time = fs::last_write_time(exe, ec);
    if (ec) {
        return "Unavailable";
    }

    const CommandResult files = run_command(git_prefix + "ls-files", false);
    if (!files.ok) {
        return "Unavailable";
    }

    fs::path newest_path;
    fs::file_time_type newest_time = fs::file_time_type::min();

    std::istringstream lines(files.output);
    std::string line;
    while (std::getline(lines, line)) {
        const fs::path path = source_dir / line;
        if (!is_source_file(path)) {
            continue;
        }
        std::error_code file_ec;
        const auto time = fs::last_write_time(path, file_ec);
        if (file_ec) {
            continue;  // deleted or unreadable
        }
        if (time > newest_time) {
            newest_time = time;
            newest_path = path;
        }
    }

    if (newest_path.empty()) {
        return "Unavailable";
    }
    return newest_time > exe_time
               ? "Yes (newer source file: " + newest_path.string() + ")"
               : "No";
}

// ---------------------------------------------------------------------------
// GPU / CUDA runtime
// ---------------------------------------------------------------------------

std::string cuda_runtime_info() {
#ifdef HAS_CUDA
    std::ostringstream out;

    int runtime_version = 0;
    int driver_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);

    auto format = [](int v) {
        return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
    };
    out << "runtime_version=" << format(runtime_version) << '\n';
    out << "driver_supports_cuda=" << format(driver_version) << '\n';

    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        out << "devices=Unavailable (" << cudaGetErrorString(err) << ")\n";
        return clean_output(out.str());
    }
    out << "device_count=" << count << '\n';

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, i) != cudaSuccess) {
            out << "device " << i << ": Unavailable\n";
            continue;
        }
        out << "device " << i << ": " << p.name << '\n';
        out << "  compute_capability=" << p.major << '.' << p.minor << '\n';
        out << "  multiprocessor_count=" << p.multiProcessorCount << '\n';
        out << "  max_threads_per_multiprocessor="
            << p.maxThreadsPerMultiProcessor << '\n';
        out << "  total_global_memory_mib="
            << p.totalGlobalMem / (1024 * 1024) << '\n';
        out << "  l2_cache_bytes=" << p.l2CacheSize << '\n';
        out << "  memory_bus_width_bits=" << p.memoryBusWidth << '\n';
        out << "  ecc_enabled=" << p.ECCEnabled << '\n';
    }
    return clean_output(out.str());
#else
    return "Not built with CUDA support";
#endif
}

const char* const kGpuStateQuery =
    "nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,"
    "pstate,utilization.gpu,memory.used --format=csv";

}  // namespace

// ---------------------------------------------------------------------------
// Public functions
// ---------------------------------------------------------------------------

std::string current_timestamp() {
    const std::time_t now = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    std::tm tm{};
    if (!localtime_r(&now, &tm)) {
        return "Unavailable";
    }
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%d %H:%M:%S %z");
    return out.str();
}

void write_run_metadata(const RunMetadata& metadata) {
    const fs::path csv_path(metadata.csv_filename);
    const fs::path meta_path(metadata_filename(metadata.csv_filename));

    if (!meta_path.parent_path().empty()) {
        fs::create_directories(meta_path.parent_path());
    }

    const fs::path source_dir(
        metadata.source_directory.empty() ? "." : metadata.source_directory);
    const fs::path build_dir(
        metadata.build_directory.empty() ? "build" : metadata.build_directory);
    const fs::path cache_path = build_dir / "CMakeCache.txt";

    // ---- git (all commands run against the source directory) ----
    const std::string git = "git -C " + shell_quote(source_dir.string()) + " ";

    const CommandResult inside = run_command(git + "rev-parse --is-inside-work-tree");
    const bool git_available = inside.ok && inside.output == "true";

    CommandResult commit, branch, describe, commit_date, subject, status;
    if (git_available) {
        commit = run_command(git + "rev-parse HEAD");
        branch = run_command(git + "rev-parse --abbrev-ref HEAD");
        describe = run_command(git + "describe --tags --always --dirty");
        commit_date = run_command(git + "log -1 --format=%cI");
        subject = run_command(git + "log -1 --format=%s");
        status = run_command(git + "status --porcelain", false);
    }

    int untracked_count = 0;
    int tracked_change_count = 0;
    {
        std::istringstream lines(status.output);
        std::string line;
        while (std::getline(lines, line)) {
            if (line.empty()) {
                continue;
            }
            if (line.rfind("??", 0) == 0) {
                ++untracked_count;
            } else {
                ++tracked_change_count;
            }
        }
    }
    const bool dirty = git_available && status.ok && !status.output.empty();

    // Tracked changes are saved as a patch; untracked files are only listed.
    std::string patch_result = "Not required";
    const fs::path patch_path =
        fs::absolute(meta_path.parent_path() / (csv_path.stem().string() + ".patch"));
    if (git_available && commit.ok && tracked_change_count > 0) {
        const std::string command = git + "diff --binary HEAD > " +
                                    shell_quote(patch_path.string()) + " 2>/dev/null";
        const int rc = std::system(command.c_str());
        std::error_code ec;
        const auto size = fs::file_size(patch_path, ec);
        if (rc != 0 || ec) {
            fs::remove(patch_path, ec);
            patch_result = "Failed";
        } else if (size == 0) {
            fs::remove(patch_path, ec);
            patch_result = "Not generated (no tracked changes)";
        } else {
            patch_result = patch_path.filename().string();
        }
    } else if (git_available && untracked_count > 0) {
        patch_result = "Not generated (only untracked files)";
    }

    // ---- write the file ----
    std::ofstream out(meta_path);
    if (!out) {
        throw std::runtime_error("Failed to write metadata file: " +
                                 meta_path.string());
    }

    out << "=== Parallel Benchmarking Run Metadata ===\n";

    section(out, "Run");
    kv(out, "schema_version", metadata.schema_version);
    kv(out, "csv_filename", metadata.csv_filename);
    kv(out, "warmup_count", std::to_string(metadata.warmup_count));
    kv(out, "repetitions", std::to_string(metadata.repetitions));
    kv(out, "command_line",
       metadata.command_line.empty() ? "Not supplied" : metadata.command_line);
    kv(out, "start_timestamp",
       metadata.start_timestamp.empty() ? current_timestamp()
                                        : metadata.start_timestamp);
    kv(out, "hostname", value_of(run_command("hostname")));
    kv(out, "pid", std::to_string(static_cast<long>(getpid())));
    {
        std::error_code ec;
        kv(out, "working_directory", fs::current_path(ec).string());
    }
    kv(out, "executable", executable_path());
    kv(out, "note", metadata.note.empty() ? "(none)" : metadata.note);

    section(out, "Git");
    kv(out, "repository_available", git_available ? "Yes" : "No");
    kv(out, "commit", git_available ? value_of(commit) : "Unavailable");
    kv(out, "branch", git_available ? value_of(branch) : "Unavailable");
    kv(out, "describe", git_available ? value_of(describe) : "Unavailable");
    kv(out, "commit_date", git_available ? value_of(commit_date) : "Unavailable");
    kv(out, "commit_subject", git_available ? value_of(subject) : "Unavailable");
    kv(out, "working_tree",
       !git_available ? "Unavailable" : (dirty ? "Dirty" : "Clean"));
    kv(out, "tracked_changes", std::to_string(tracked_change_count));
    kv(out, "untracked_files", std::to_string(untracked_count));
    kv(out, "patch_file", patch_result);
    kv(out, "binary_older_than_sources",
       git_available ? stale_binary_check(git, source_dir) : "Unavailable");
    if (dirty) {
        block(out, "status_porcelain", status.output);
    }
    kv(out, "note",
       "Untracked files are listed in status_porcelain but are not part of the patch");

    section(out, "Build");
    kv(out, "source_directory", source_dir.string());
    kv(out, "build_directory", build_dir.string());
    const std::string build_type = read_cache_value(cache_path, "CMAKE_BUILD_TYPE");
    kv(out, "CMAKE_BUILD_TYPE", build_type);
    kv(out, "CMAKE_GENERATOR", read_cache_value(cache_path, "CMAKE_GENERATOR"));
    kv(out, "ENABLE_CUDA", read_cache_value(cache_path, "ENABLE_CUDA"));
    kv(out, "CMAKE_CXX_COMPILER", read_cache_value(cache_path, "CMAKE_CXX_COMPILER"));
    kv(out, "CMAKE_CXX_FLAGS", read_cache_value(cache_path, "CMAKE_CXX_FLAGS"));
    if (!is_unavailable(build_type)) {
        kv(out, "CMAKE_CXX_FLAGS_" + upper(build_type),
           read_cache_value(cache_path, "CMAKE_CXX_FLAGS_" + upper(build_type)));
    }
    kv(out, "CMAKE_CUDA_COMPILER", read_cache_value(cache_path, "CMAKE_CUDA_COMPILER"));
    kv(out, "CMAKE_CUDA_FLAGS", read_cache_value(cache_path, "CMAKE_CUDA_FLAGS"));
    if (!is_unavailable(build_type)) {
        kv(out, "CMAKE_CUDA_FLAGS_" + upper(build_type),
           read_cache_value(cache_path, "CMAKE_CUDA_FLAGS_" + upper(build_type)));
    }
    kv(out, "CMAKE_CUDA_ARCHITECTURES",
       read_cache_value(cache_path, "CMAKE_CUDA_ARCHITECTURES"));
    kv(out, "OpenMP_CXX_FLAGS", read_cache_value(cache_path, "OpenMP_CXX_FLAGS"));
    kv(out, "OpenMP_CXX_SPEC_DATE", read_cache_value(cache_path, "OpenMP_CXX_SPEC_DATE"));
    block(out, "flags_used_per_target",
          collect_target_flags(build_dir));

    section(out, "Toolchain");
    {
        const std::string cxx = read_cache_value(cache_path, "CMAKE_CXX_COMPILER");
        const std::string cxx_cmd = is_unavailable(cxx) ? "c++" : shell_quote(cxx);
        kv(out, "cxx_compiler", value_of_head(run_command(cxx_cmd + " --version"), 1));

        const std::string nvcc = read_cache_value(cache_path, "CMAKE_CUDA_COMPILER");
        const std::string nvcc_cmd = is_unavailable(nvcc) ? "nvcc" : shell_quote(nvcc);
        block(out, "nvcc", value_of(run_command(nvcc_cmd + " --version")));

        kv(out, "cmake", value_of_head(run_command("cmake --version"), 1));

        const std::string generator = read_cache_value(cache_path, "CMAKE_GENERATOR");
        if (generator.find("Ninja") != std::string::npos) {
            kv(out, "ninja", value_of(run_command("ninja --version")));
        } else if (generator.find("Makefiles") != std::string::npos) {
            kv(out, "make", value_of_head(run_command("make --version"), 1));
        }

        const std::string mpiexec = read_cache_value(cache_path, "MPIEXEC_EXECUTABLE");
        const std::string mpi_cmd = is_unavailable(mpiexec) ? "mpirun" : shell_quote(mpiexec);
        kv(out, "mpi", value_of_head(run_command(mpi_cmd + " --version"), 1));
    }

    section(out, "Environment Variables");
    for (const char* name :
         {"OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_SCHEDULE",
          "OMP_DYNAMIC", "CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING",
          "CUDA_LAUNCH_BLOCKING", "CUDA_DEVICE_MAX_CONNECTIONS"}) {
        kv(out, name, env_value(name));
    }

    section(out, "CPU and Memory");
    {
        CommandResult lscpu = run_command("lscpu");
        if (lscpu.ok) {
            lscpu.output = drop_lines_starting_with(lscpu.output, {"Flags", "Vulnerability"});
        }
        block(out, "lscpu", value_of(lscpu));
    }
    kv(out, "nproc", value_of(run_command("nproc")));
    kv(out, "cpu_affinity",
       value_of(run_command("taskset -cp " + std::to_string(static_cast<long>(getpid())))));
    kv(out, "cpu0_governor",
       value_of(run_command("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")));
    block(out, "cpu_mhz_sample",
          value_of_head(run_command("grep 'cpu MHz' /proc/cpuinfo"), 4));
    block(out, "memory",
          value_of(run_command("grep -E 'MemTotal|MemAvailable|SwapTotal' /proc/meminfo")));

    section(out, "System Load at Start");
    kv(out, "loadavg", value_of(run_command("cat /proc/loadavg")));
    block(out, "top_processes_by_cpu",
          value_of_head(run_command("ps -eo pcpu,pmem,comm --sort=-pcpu"), 8));

    section(out, "GPU");
    block(out, "device",
          value_of(run_command("nvidia-smi --query-gpu=name,driver_version,"
                               "memory.total,power.limit --format=csv")));
    block(out, "compute_capability",
          value_of(run_command("nvidia-smi --query-gpu=compute_cap --format=csv")));
    block(out, "max_clocks",
          value_of(run_command("nvidia-smi --query-gpu=clocks.max.sm,clocks.max.mem --format=csv")));
    block(out, "state_at_start", value_of(run_command(kGpuStateQuery)));
    block(out, "throttle_reasons_at_start",
          value_of(run_command("nvidia-smi -q -d PERFORMANCE")));
    block(out, "nvidia_smi", value_of(run_command("nvidia-smi")));
    if (metadata.query_cuda_runtime) {
        block(out, "cuda_runtime", cuda_runtime_info());
    } else {
        kv(out, "cuda_runtime", "Skipped (query_cuda_runtime=false)");
    }

    section(out, "Operating System");
    kv(out, "uname", value_of(run_command("uname -a")));
    kv(out, "distribution",
       value_of(run_command("grep PRETTY_NAME /etc/os-release")));
    const bool wsl = is_wsl();
    kv(out, "wsl", wsl ? "Yes" : "No");

    section(out, "Power and Host");
    block(out, "linux_power_supply",
          value_of(run_command(R"SH(for d in /sys/class/power_supply/*; do [ -d "$d" ] || continue; echo "$d type=$(cat "$d/type" 2>/dev/null) online=$(cat "$d/online" 2>/dev/null) status=$(cat "$d/status" 2>/dev/null) capacity=$(cat "$d/capacity" 2>/dev/null)"; done)SH")));
    if (wsl) {
        // Run from /mnt/c so cmd.exe does not complain about a UNC working directory.
        kv(out, "windows_version",
           value_of(run_command("cd /mnt/c && cmd.exe /c ver")));
        kv(out, "windows_active_power_plan",
           value_of(run_command("cd /mnt/c && powercfg.exe /getactivescheme")));
    }

    out.flush();
    if (!out) {
        throw std::runtime_error("Failed while writing metadata file: " +
                                 meta_path.string());
    }
}

bool append_run_end_metadata(const RunMetadata& metadata) {
    try {
        std::ofstream out(metadata_filename(metadata.csv_filename), std::ios::app);
        if (!out) {
            return false;
        }

        section(out, "End");
        kv(out, "end_timestamp",
           metadata.end_timestamp.empty() ? current_timestamp() : metadata.end_timestamp);
        kv(out, "loadavg_after_run", value_of(run_command("cat /proc/loadavg")));
        block(out, "gpu_state_after_run", value_of(run_command(kGpuStateQuery)));
        block(out, "throttle_reasons_after_run",
              value_of(run_command("nvidia-smi -q -d PERFORMANCE")));

        out.flush();
        return static_cast<bool>(out);
    } catch (...) {
        return false;
    }
}