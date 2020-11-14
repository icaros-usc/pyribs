"""Benchmarks for the CVTArhive."""
from ribs.archives import CVTArchive

# pylint: disable = invalid-name, unused-variable


def benchmark_construction(use_kd_tree, benchmark):
    """Construction includes k-means clustering and building a kd-tree."""

    @benchmark
    def construct():
        archive = CVTArchive([(-1, 1), (-1, 1)],
                             1000,
                             samples=10_000,
                             use_kd_tree=use_kd_tree)
        archive.initialize(solution_dim=2)


def benchmark_10k_additions(use_kd_tree, benchmark, benchmark_data_100k):
    n, solutions, objective_values, behavior_values = benchmark_data_100k

    def setup():
        archive = CVTArchive([(-1, 1), (-1, 1)],
                             1000,
                             samples=10_000,
                             use_kd_tree=use_kd_tree)
        archive.initialize(solutions.shape[1])
        return (archive,), {}

    def add_100k(archive):
        for i in range(10_000):
            archive.add(solutions[i], objective_values[i], behavior_values[i])

    benchmark.pedantic(add_100k, setup=setup, rounds=5, iterations=1)


def benchmark_get_100k_random_elites(use_kd_tree, benchmark,
                                     benchmark_data_100k):
    n, solutions, objective_values, behavior_values = benchmark_data_100k
    archive = CVTArchive([(-1, 1), (-1, 1)],
                         1000,
                         samples=10_000,
                         use_kd_tree=use_kd_tree)
    archive.initialize(solutions.shape[1])
    for i in range(n):
        archive.add(solutions[i], objective_values[i], behavior_values[i])

    @benchmark
    def get_elites():
        for i in range(n):
            sol, obj, beh = archive.get_random_elite()
