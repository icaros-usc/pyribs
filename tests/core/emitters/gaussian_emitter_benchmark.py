"""Benchmarks for the GaussianEmitter."""

import numpy as np

from ribs.emitters import GaussianEmitter

# pylint: disable = invalid-name, unused-variable


def benchmark_ask_tell_100k(benchmark, fake_archive_fixture):
    archive, x0 = fake_archive_fixture
    sigma0 = 1
    batch_size = 32
    emitter = GaussianEmitter(x0, sigma0, archive, batch_size=batch_size)
    n = 100_000

    np.random.seed(0)

    objective_values = np.random.rand(batch_size)
    behavior_values = np.random.rand(3, batch_size)

    # Let numba compile.
    temp_sol = emitter.ask()
    emitter.tell(temp_sol, objective_values, behavior_values)

    obj_vals = np.random.rand(n, batch_size)
    behavior_vals = np.random.rand(n, 3, batch_size)

    @benchmark
    def ask_and_tell():
        for i in range(n):
            solutions = emitter.ask()
            objective_values = obj_vals[i]
            behavior_values = behavior_vals[i]
            emitter.tell(solutions, objective_values, behavior_values)
