"""Benchmarks for the GridArchive."""
import numpy as np

from ribs.archives import GridArchive


def benchmark_add_10k(benchmark, benchmark_data_10k):
    n, solutions, objective_values, behavior_values = benchmark_data_10k

    def setup():
        archive = GridArchive(solution_dim=solutions.shape[1],
                              dims=(64, 64),
                              ranges=[(-1, 1), (-1, 1)])

        # Let numba compile.
        archive.add(solutions[0], objective_values[0], behavior_values[0])

        return (archive,), {}

    def add_10k(archive):
        for i in range(n):
            archive.add(solutions[i], objective_values[i], behavior_values[i])

    benchmark.pedantic(add_10k, setup=setup, rounds=5, iterations=1)


def benchmark_as_pandas_2025_items(benchmark):
    dim = 45
    archive = GridArchive(solution_dim=10,
                          dims=(dim, dim),
                          ranges=[(-1, 1), (-1, 1)])

    for x in np.linspace(-1, 1, dim):
        for y in np.linspace(-1, 1, dim):
            sol = np.random.random(10)
            sol[0] = x
            sol[1] = y
            archive.add(sol, 1.0, np.array([x, y]))

    # Archive should be full.
    assert len(archive) == dim * dim

    benchmark(archive.as_pandas)
