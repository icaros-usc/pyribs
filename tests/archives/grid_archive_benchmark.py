"""Tests for the GridArchive."""
import numpy as np

from ribs.archives import GridArchive

# pylint: disable = invalid-name, unused-variable, missing-function-docstring


def benchmark_add_hundred_thousand_entries(benchmark):
    archive = GridArchive([64, 64], [(-1, 1), (-1, 1)])

    itrs = int(1e5)

    solutions = np.random.uniform(-1, 1, (itrs, 2))
    behavior_values = solutions
    objective_values = np.random.uniform(-10, 10, itrs)

    @benchmark
    def insert_hundred_thousand_entries():
        for i in range(itrs):
            archive.add(solutions[i], objective_values[i], behavior_values[i])
