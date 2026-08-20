#ifndef RESULT_WRITER_H
#define RESULT_WRITER_H

void write_csv(
    const std::string& filename,
    const std::vector<BenchmarkResult>& results
);

#endif