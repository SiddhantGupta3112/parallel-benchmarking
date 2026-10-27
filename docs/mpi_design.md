# MPI Kernel Design Note

## Overview

The MPI implementation provides distributed versions of matrix
multiplication, Monte Carlo estimation of π, and the stencil computation.
The kernels use MPI processes to divide work while preserving the
mathematical behavior of the corresponding serial implementations.

## Matrix Multiplication

Matrix multiplication distributes rows of matrix A among MPI ranks.
Matrix B is broadcast to the participating ranks, which independently
compute their assigned rows of the output matrix C. The root rank gathers
the local output rows to construct the final result.

The row distribution supports uneven workloads and cases where the number
of ranks exceeds the number of rows. Input validation and input preparation
are coordinated through MPI communication. The measured runtime includes
communication and computation, as well as the root-side copy of matrix B.

## Monte Carlo Estimation

The Monte Carlo kernel divides the requested sample count among MPI ranks.
Each rank generates random samples and counts how many points fall inside
the quarter-circle. A reduction combines the local counts, and the final
estimate is calculated from the total number of points inside the circle.

Ranks use rank-specific random-number streams. Consequently, estimates can
vary with the process count even when the total sample count is unchanged.
Uneven and small sample counts are handled by the work distribution.

## Stencil Computation

The stencil kernel distributes grid rows across MPI ranks. Each iteration
exchanges halo rows between neighboring active ranks so that boundary
updates have access to neighboring values from adjacent partitions.
The ranks update their assigned grid regions, and the root gathers the
final grid.

Grid boundaries remain unchanged. The root rank creates the initial input
and communicates input preparation status to the other ranks, avoiding
unnecessary full-grid copies on non-root ranks.

## Measurement and Limitations

Wall-clock measurements include the communication operations performed
inside the measured kernel, including broadcasts, scatter/gather operations,
reductions, and stencil halo exchanges.

Reported CPU time is aggregated across MPI ranks. Peak RSS represents the
maximum per-process peak and is a process-lifetime high-water mark, so an
earlier memory-intensive benchmark may affect later rows in the same run.

The MPI benchmark runner selects configurations for the launched rank
count and compares results against serial core baselines. Baselines use
the median wall-clock time of the available serial core repetitions.

The `mpi_comm_overhead_fraction` field is not yet populated. Separately
measuring communication time and computation time remains future work.
