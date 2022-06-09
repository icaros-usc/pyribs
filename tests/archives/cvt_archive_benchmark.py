"""Benchmarks for the CVTArhive."""
import numpy as np

from ribs.archives import CVTArchive


def benchmark_init(use_kd_tree, benchmark):
    """Construction includes k-means clustering and building a kd-tree."""

    def init():
        archive = CVTArchive(solution_dim=2,
                             cells=1000,
                             ranges=[(-1, 1), (-1, 1)],
                             samples=20_000,
                             use_kd_tree=use_kd_tree)

    benchmark(init)


def benchmark_add_10k(use_kd_tree, benchmark, benchmark_data_10k):
    n, solutions, objective_values, behavior_values = benchmark_data_10k

    def setup():
        archive = CVTArchive(solution_dim=solutions.shape[1],
                             cells=1000,
                             ranges=[(-1, 1), (-1, 1)],
                             samples=20_000,
                             use_kd_tree=use_kd_tree)

        # Let numba compile.
        archive.add(solutions[0], objective_values[0], behavior_values[0])

        return (archive,), {}

    def add_10k(archive):
        for i in range(n):
            archive.add(solutions[i], objective_values[i], behavior_values[i])

    benchmark.pedantic(add_10k, setup=setup, rounds=5, iterations=1)


def benchmark_as_pandas_2000_items(benchmark):
    cells = 2000
    archive = CVTArchive(solution_dim=10,
                         cells=cells,
                         ranges=[(-1, 1), (-1, 1)],
                         use_kd_tree=True,
                         samples=50_000)

    for x, y in archive.centroids:
        sol = np.random.random(10)
        sol[0] = x
        sol[1] = y
        archive.add(sol, 1.0, np.array([x, y]))

    # Archive should be full.
    assert len(archive) == cells

    benchmark(archive.as_pandas)
