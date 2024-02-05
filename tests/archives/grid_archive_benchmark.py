"""Benchmarks for the GridArchive."""
import numpy as np

from ribs.archives import GridArchive


def benchmark_add_10k(benchmark, benchmark_data_10k):
    _, solution_batch, objective_batch, measures_batch = benchmark_data_10k

    def setup():
        archive = GridArchive(solution_dim=solution_batch.shape[1],
                              dims=(64, 64),
                              ranges=[(-1, 1), (-1, 1)])

        # Let numba compile.
        archive.add_single(solution_batch[0], objective_batch[0],
                           measures_batch[0])

        return (archive,), {}

    def add_10k(archive):
        archive.add(solution_batch, objective_batch, measures_batch)

    benchmark.pedantic(add_10k, setup=setup, rounds=5, iterations=1)
