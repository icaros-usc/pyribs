"""Benchmarks for the GridArchive."""

# pylint: disable = invalid-name, unused-variable, missing-function-docstring

from ribs.archives import GridArchive

# See conftest.py for benchmark_data_100k.


def benchmark_construct_and_add_100k(benchmark, benchmark_data_100k):
    n, solutions, objective_values, behavior_values = benchmark_data_100k

    @benchmark
    def insert_hundred_thousand_entries():
        archive = GridArchive((64, 64), [(-1, 1), (-1, 1)])
        for i in range(n):
            archive.add(solutions[i], objective_values[i], behavior_values[i])


def benchmark_get_100k_random_elites(benchmark, benchmark_data_100k):
    n, solutions, objective_values, behavior_values = benchmark_data_100k
    archive = GridArchive((64, 64), [(-1, 1), (-1, 1)])
    for i in range(n):
        archive.add(solutions[i], objective_values[i], behavior_values[i])

    @benchmark
    def get_hundred_thousand_elites():
        for i in range(n):
            ind = archive.get_random_elite()
