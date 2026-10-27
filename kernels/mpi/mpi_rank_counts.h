#ifndef MPI_RANK_COUNTS_H
#define MPI_RANK_COUNTS_H

#include <array>

inline constexpr std::array<int, 7> MPI_RANK_COUNTS = {
    1, 2, 4, 8, 12, 16, 20
};

#endif
