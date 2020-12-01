"""Benchmarks for the GridArchive."""

from ribs.archives import GridArchive

# pylint: disable = invalid-name, unused-variable


def benchmark_add_100k(benchmark, benchmark_data_100k):
    n, solutions, objective_values, behavior_values = benchmark_data_100k

    def setup():
        archive = GridArchive((64, 64), [(-1, 1), (-1, 1)])
        archive.initialize(solutions.shape[1])

        # Let numba compile.
        archive.add(solutions[0], objective_values[0], behavior_values[0])

        return (archive,), {}

    def add_100k(archive):
        for i in range(n):
            archive.add(solutions[i], objective_values[i], behavior_values[i])

    benchmark.pedantic(add_100k, setup=setup, rounds=5, iterations=1)


def benchmark_get_100k_random_elites(benchmark, benchmark_data_100k):
    n, solutions, objective_values, behavior_values = benchmark_data_100k
    archive = GridArchive((64, 64), [(-1, 1), (-1, 1)])
    archive.initialize(solutions.shape[1])
    for i in range(n):
        archive.add(solutions[i], objective_values[i], behavior_values[i])

    @benchmark
    def get_elites():
        for i in range(n):
            sol, obj, beh = archive.get_random_elite()
