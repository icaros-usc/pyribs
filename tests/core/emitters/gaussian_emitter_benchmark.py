"""Benchmarks for the GaussianEmitter."""

import numpy as np

from ribs.emitters import GaussianEmitter

# pylint: disable = invalid-name, unused-variable


def benchmark_ask_tell_100k(benchmark, fake_archive_fixture):
    archive, x0 = fake_archive_fixture
    sigma0 = 1
    batch_size = 2
    emitter = GaussianEmitter(x0, sigma0, archive, batch_size=batch_size)

    objective_values = np.full(batch_size, 1.)
    behavior_values = np.array([[-1, -1], [0, 0], [1, 1]])

    # Let numba compile.
    temp_sol = emitter.ask()
    emitter.tell(temp_sol, objective_values, behavior_values)

    @benchmark
    def ask_and_tell():
        for i in range(int(1e5)):
            solutions = emitter.ask()
            emitter.tell(solutions, objective_values, behavior_values)
