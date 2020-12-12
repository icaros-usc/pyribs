"""Benchmarks for the IsoLineEmitter."""

import numpy as np

from ribs.emitters import IsoLineEmitter

# pylint: disable = invalid-name, unused-variable


def benchmark_ask_tell_100k(benchmark, fake_archive_fixture):
    archive, x0 = fake_archive_fixture
    batch_size = 2
    emitter = IsoLineEmitter(x0, archive, batch_size=batch_size)

    np.random.seed(0)

    objective_values = np.random.rand(batch_size)
    behavior_values = np.random.rand(3, 2)

    # Let numba compile.
    temp_sol = emitter.ask()
    emitter.tell(temp_sol, objective_values, behavior_values)

    obj_vals = np.random.rand(int(1e5), batch_size)
    behavior_vals = np.random.rand(int(1e5), 3, 2)

    @benchmark
    def ask_and_tell():
        for i in range(int(1e5)):
            solutions = emitter.ask()
            objective_values = obj_vals[i]
            behavior_values = behavior_vals[i]
            emitter.tell(solutions, objective_values, behavior_values)
