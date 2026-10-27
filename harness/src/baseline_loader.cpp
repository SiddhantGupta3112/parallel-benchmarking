#include "baseline_loader.h"
#include "derived_metrics.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool inside_quotes = false;

    for (std::size_t i = 0; i < line.size(); ++i) {
        char character = line[i];

        if (character == '"') {
            if (inside_quotes && i + 1 < line.size() &&
                line[i + 1] == '"') {
                field += '"';
                ++i;
            } else {
                inside_quotes = !inside_quotes;
            }
        } else if (character == ',' && !inside_quotes) {
            fields.push_back(field);
            field.clear();
        } else {
            field += character;
        }
    }

    if (inside_quotes) {
        throw std::runtime_error(
            "Malformed CSV: unterminated quoted field"
        );
    }

    fields.push_back(field);
    return fields;
}

const std::string& csv_field(
    const std::vector<std::string>& fields,
    const std::unordered_map<std::string, std::size_t>& columns,
    const std::string& name
) {
    auto column = columns.find(name);

    if (column == columns.end()) {
        throw std::runtime_error("Missing CSV column: " + name);
    }

    if (column->second >= fields.size()) {
        throw std::runtime_error("CSV row is missing column: " + name);
    }

    return fields[column->second];
}

double parse_csv_double(const std::string& value) {
    std::size_t position = 0;
    double result = std::stod(value, &position);

    if (position != value.size() || !std::isfinite(result)) {
        throw std::runtime_error("Invalid numeric CSV value: " + value);
    }

    return result;
}

int parse_csv_integer(const std::string& value) {
    std::size_t position = 0;
    int result = std::stoi(value, &position);

    if (position != value.size()) {
        throw std::runtime_error("Invalid integer CSV value: " + value);
    }

    return result;
}

long parse_csv_long(const std::string& value) {
    std::size_t position = 0;
    long result = std::stol(value, &position);

    if (position != value.size()) {
        throw std::runtime_error("Invalid integer CSV value: " + value);
    }

    return result;
}

void remove_carriage_return(std::string& line) {
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
}

}  // namespace

BaselineMap load_serial_baselines(const std::string& filename) {
    std::ifstream file(filename);

    if (!file) {
        throw std::runtime_error(
            "Cannot open baseline CSV: " + filename
        );
    }

    std::string line;

    if (!std::getline(file, line)) {
        throw std::runtime_error("Baseline CSV is empty");
    }

    remove_carriage_return(line);
    const auto header = parse_csv_line(line);

    std::unordered_map<std::string, std::size_t> columns;

    for (std::size_t i = 0; i < header.size(); ++i) {
        columns[header[i]] = i;
    }

    using BaselineSamples =
        std::unordered_map<std::string, std::vector<BenchmarkResult>>;

    BaselineSamples samples;

    while (std::getline(file, line)) {
        remove_carriage_return(line);

        if (line.empty()) {
            continue;
        }

        const auto fields = parse_csv_line(line);

        const std::string& paradigm =
            csv_field(fields, columns, "paradigm");
        const std::string& run_type =
            csv_field(fields, columns, "run_type");

        if (paradigm != "Serial" || run_type != "core") {
            continue;
        }

        BenchmarkResult result;

        result.name = csv_field(fields, columns, "name");
        result.kernel_name = csv_field(fields, columns, "kernel_name");
        result.paradigm = Paradigm::Serial;
        result.run_type = RunType::Core;

        result.rep_index = parse_csv_integer(
            csv_field(fields, columns, "rep_index")
        );
        result.problem_size = parse_csv_integer(
            csv_field(fields, columns, "problem_size")
        );
        result.iterations = parse_csv_integer(
            csv_field(fields, columns, "iterations")
        );
        result.number_of_processors = parse_csv_integer(
            csv_field(fields, columns, "number_of_processors")
        );
        result.wall_time_ms = parse_csv_double(
            csv_field(fields, columns, "wall_time_ms")
        );
        result.cpu_time_ms = parse_csv_double(
            csv_field(fields, columns, "cpu_time_ms")
        );
        result.peak_rss_kb = parse_csv_long(
            csv_field(fields, columns, "peak_rss_kb")
        );
        result.flop_count = parse_csv_double(
            csv_field(fields, columns, "flop_count")
        );
        result.bytes_moved = parse_csv_double(
            csv_field(fields, columns, "bytes_moved")
        );

        if (result.wall_time_ms <= 0.0 ||
            result.problem_size <= 0 ||
            result.number_of_processors <= 0) {
            throw std::runtime_error(
                "Invalid Serial baseline for " + result.kernel_name
            );
        }

        samples[result.kernel_name].push_back(result);
    }

    if (samples.empty()) {
        throw std::runtime_error(
            "No Serial core measurements found in baseline CSV"
        );
    }

    BaselineMap baselines;

    for (auto& entry : samples) {
        const std::string& kernel_name = entry.first;
        auto& repetitions = entry.second;

        std::vector<double> times;
        times.reserve(repetitions.size());

        const BenchmarkResult& first = repetitions.front();

        for (const auto& repetition : repetitions) {
            if (repetition.problem_size != first.problem_size ||
                repetition.iterations != first.iterations) {
                throw std::runtime_error(
                    "Inconsistent Serial configurations for " + kernel_name
                );
            }

            times.push_back(repetition.wall_time_ms);
        }

        std::sort(times.begin(), times.end());

        const std::size_t middle = times.size() / 2;
        double median;

        if (times.size() % 2 == 0) {
            median = (times[middle - 1] + times[middle]) / 2.0;
        } else {
            median = times[middle];
        }

        BenchmarkResult baseline = first;
        baseline.wall_time_ms = median;
        baseline.speedup = 1.0;
        baseline.efficiency = 1.0;

        compute_achieved_metrics(
            baseline,
            baseline.flop_count,
            baseline.bytes_moved
        );

        baselines.emplace(kernel_name, baseline);
    }

    return baselines;
}
