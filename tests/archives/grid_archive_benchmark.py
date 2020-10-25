"""Benchmarks for the GridArchive."""

# pylint: disable = invalid-name, unused-variable, missing-function-docstring

from ribs.archives import GridArchive


def benchmark_add_hundred_thousand_entries(benchmark, benchmark_data):
    n, solutions, objective_values, behavior_values = benchmark_data

    @benchmark
    def insert_hundred_thousand_entries():
        archive = GridArchive((64, 64), [(-1, 1), (-1, 1)])
        assert archive.is_empty()
        for i in range(n):
            archive.add(solutions[i], objective_values[i], behavior_values[i])
