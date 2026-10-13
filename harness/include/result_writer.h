#ifndef RESULT_WRITER_H
#define RESULT_WRITER_H

#include <fstream> 
#include <string> 
#include <vector> 

#include "benchmark_result.h"

void write_raw_data_header(std::ofstream& file);

void write_result_row(std::ofstream& file, const BenchmarkResult& result);

std::ofstream open_file(
    const std::string& filename,
    std::ios::openmode mode
);

void initialize_raw_data_csv(const std::string& filename);

void write_raw_data_csv(
    const std::string& filename,
    const std::vector<BenchmarkResult>& results
);

#endif