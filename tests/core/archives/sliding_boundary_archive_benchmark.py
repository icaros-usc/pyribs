"""Benchmarks for the GridArchive."""

from ribs.archives import SlidingBoundaryArchive

# pylint: disable = invalid-name, unused-variable


def benchmark_construct_and_add_10k(benchmark, benchmark_data_100k):
    _, solutions, objective_values, behavior_values = benchmark_data_100k
    n = int(1e4)
    @benchmark
    def insert_hundred_thousand_entries():
        archive = SlidingBoundaryArchive((64, 64), [(-1, 1), (-1, 1)])
        for i in range(n):
            archive.add(solutions[i], objective_values[i], behavior_values[i])


def benchmark_get_10k_random_elites(benchmark, benchmark_data_100k):
    _, solutions, objective_values, behavior_values = benchmark_data_100k
    n = int(1e4)
    archive = SlidingBoundaryArchive((64, 64), [(-1, 1), (-1, 1)])
    for i in range(n):
        archive.add(solutions[i], objective_values[i], behavior_values[i])

    @benchmark
    def get_elites():
        for i in range(n):
            sol, obj, beh = archive.get_random_elite()
