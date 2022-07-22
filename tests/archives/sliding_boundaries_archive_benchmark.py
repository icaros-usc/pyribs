"""Benchmarks for the SlidingBoundariesArchive."""
import numpy as np
import pytest

from ribs.archives import SlidingBoundariesArchive


@pytest.mark.skip
def benchmark_add_10k(benchmark, benchmark_data_10k):
    n, solutions, objective_values, behavior_values = benchmark_data_10k

    def setup():
        archive = SlidingBoundariesArchive(solutions.shape[1], [10, 20],
                                           [(-1, 1), (-2, 2)],
                                           remap_frequency=100,
                                           buffer_capacity=1000)

        # Let numba compile.
        archive.add(solutions[0], objective_values[0], behavior_values[0])

        return (archive,), {}

    def add_10k(archive):
        for i in range(n):
            archive.add(solutions[i], objective_values[i], behavior_values[i])

    benchmark.pedantic(add_10k, setup=setup, rounds=5, iterations=1)


@pytest.mark.skip
def benchmark_as_pandas_2048_elements(benchmark):
    # TODO (btjanaka): Make this size smaller so that we do a remap.
    archive = SlidingBoundariesArchive(10, [32, 64], [(-1, 1), (-2, 2)],
                                       remap_frequency=20000,
                                       buffer_capacity=20000)

    for x in np.linspace(-1, 1, 100):
        for y in np.linspace(-2, 2, 100):
            sol = np.random.random(10)
            sol[0] = x
            sol[1] = y
            archive.add(sol, -(x**2 + y**2), np.array([x, y]))

    # Archive should be full.
    assert len(archive) == 32 * 64

    benchmark(archive.as_pandas)
