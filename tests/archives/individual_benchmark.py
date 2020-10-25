"""Tests for the Individual."""

from ribs.archives import Individual

from .utils import benchmark_data_100k

# pylint: disable = invalid-name, unused-variable, missing-function-docstring


def benchmark_individual_construction(benchmark, benchmark_data_100k):
    n, solutions, objective_values, behavior_values = benchmark_data_100k

    @benchmark
    def construct_individuals():
        for i in range(n):
            Individual(objective_values[i], behavior_values[i], solutions[i])
