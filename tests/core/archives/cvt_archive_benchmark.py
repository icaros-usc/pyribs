"""Benchmarks for the CVTArhive."""
import numpy as np

from ribs.archives import CVTArchive

# pylint: disable = invalid-name, unused-variable


def benchmark_init(use_kd_tree, benchmark):
    """Construction includes k-means clustering and building a kd-tree."""

    @benchmark
    def init():
        archive = CVTArchive([(-1, 1), (-1, 1)],
                             1000,
                             samples=20_000,
                             use_kd_tree=use_kd_tree)
        archive.initialize(solution_dim=2)


def benchmark_add_10k(use_kd_tree, benchmark, benchmark_data_10k):
    n, solutions, objective_values, behavior_values = benchmark_data_10k

    def setup():
        archive = CVTArchive([(-1, 1), (-1, 1)],
                             1000,
                             samples=20_000,
                             use_kd_tree=use_kd_tree)
        archive.initialize(solutions.shape[1])

        # Let numba compile.
        archive.add(solutions[0], objective_values[0], behavior_values[0])

        return (archive,), {}

    def add_10k(archive):
        for i in range(n):
            archive.add(solutions[i], objective_values[i], behavior_values[i])

    benchmark.pedantic(add_10k, setup=setup, rounds=5, iterations=1)


def benchmark_get_10k_random_elites(use_kd_tree, benchmark, benchmark_data_10k):
    n, solutions, objective_values, behavior_values = benchmark_data_10k
    archive = CVTArchive([(-1, 1), (-1, 1)],
                         1000,
                         samples=20_000,
                         use_kd_tree=use_kd_tree)
    archive.initialize(solutions.shape[1])
    for i in range(n):
        archive.add(solutions[i], objective_values[i], behavior_values[i])

    @benchmark
    def get_elites():
        for i in range(n):
            sol, obj, beh = archive.get_random_elite()


def benchmark_as_pandas_2000_items(benchmark):
    bins = 2000
    archive = CVTArchive([(-1, 1), (-1, 1)],
                         bins,
                         use_kd_tree=True,
                         samples=50_000)
    archive.initialize(10)

    for x, y in archive.centroids:
        sol = np.random.random(10)
        sol[0] = x
        sol[1] = y
        archive.add(sol, 1.0, np.array([x, y]))

    # Archive should be full.
    assert len(archive.as_pandas()) == bins

    benchmark(archive.as_pandas)
