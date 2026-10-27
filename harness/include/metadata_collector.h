#ifndef METADATA_COLLECTOR_H
#define METADATA_COLLECTOR_H

#include <string>

// Everything a reader needs to know about how a result CSV was produced.
// One text file is written next to each CSV (same base name, ".meta.txt").
// A ".patch" file is written next to it as well when the git tree has
// tracked, uncommitted changes.
struct RunMetadata {
    // ---- run settings (fill these in from the runner) ----
    std::string schema_version;     // version of the CSV layout
    std::string csv_filename;       // path of the CSV this file describes
    int warmup_count = 0;
    int repetitions = 0;
    std::string command_line;       // the full command line as typed
    std::string start_timestamp;    // empty -> taken when the file is written
    std::string end_timestamp;      // empty -> taken when the end section is written
    std::string end_status;         // e.g. "completed" or "failed: <message>"

    // ---- locations (best baked in at configure time, see the CMake note) ----
    std::string build_directory = "build";   // directory holding CMakeCache.txt
    std::string source_directory = ".";      // git repository root

    // ---- manual facts no tool can see ----
    // Example: "plugged in; Windows best-performance mode; browsers closed"
    std::string note;

    // Querying the CUDA runtime can initialise it, which would hide a true
    // cold start. Set this to false for cold-start experiments.
    bool query_cuda_runtime = true;
};

// Current local time as "YYYY-MM-DD HH:MM:SS +ZZZZ".
std::string current_timestamp();

// Writes <csv basename>.meta.txt (and .patch when the tree has tracked
// changes). Call it BEFORE the benchmarks start so a crashed run still has
// metadata. Throws std::runtime_error if the file cannot be written.
void write_run_metadata(const RunMetadata& metadata);

// Appends an [End] section (end time, GPU state right after the run, load).
// Never throws; returns false if the section could not be written, so a
// finished run is never turned into a failure by a metadata problem.
bool append_run_end_metadata(const RunMetadata& metadata);

#endif