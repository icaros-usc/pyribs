"""Benchmarks for the SlidingBoundariesArchive."""
import numpy as np

from ribs.archives import SlidingBoundariesArchive


def benchmark_add_10k(benchmark, benchmark_data_10k):
    n, solution_batch, objective_batch, measures_batch = benchmark_data_10k

    def setup():
        archive = SlidingBoundariesArchive(solution_dim=solution_batch.shape[1],
                                           dims=[10, 20],
                                           ranges=[(-1, 1), (-2, 2)],
                                           remap_frequency=100,
                                           buffer_capacity=1000)

        # Let numba compile.
        archive.add_single(solution_batch[0], objective_batch[0],
                           measures_batch[0])

        return (archive,), {}

    def add_10k(archive):
        archive.add(solution_batch, objective_batch, measures_batch)

    benchmark.pedantic(add_10k, setup=setup, rounds=5, iterations=1)


def benchmark_as_pandas_2048_elements(benchmark):
    # TODO (btjanaka): Make this size smaller so that we do a remap.
    archive = SlidingBoundariesArchive(solution_dim=10,
                                       dims=[32, 64],
                                       ranges=[(-1, 1), (-2, 2)],
                                       remap_frequency=20000,
                                       buffer_capacity=20000)

    for x in np.linspace(-1, 1, 100):
        for y in np.linspace(-2, 2, 100):
            sol = np.random.random(10)
            sol[0] = x
            sol[1] = y
            archive.add_single(sol, -(x**2 + y**2), np.array([x, y]))

    # Archive should be full.
    assert len(archive) == 32 * 64

    benchmark(archive.as_pandas)
