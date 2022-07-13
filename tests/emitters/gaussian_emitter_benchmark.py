"""Benchmarks for the GaussianEmitter."""

import numpy as np

from ribs.emitters import GaussianEmitter


def benchmark_ask_tell_100k(benchmark, fake_archive_fixture):
    archive, x0 = fake_archive_fixture
    sigma0 = 1
    batch_size = 32
    emitter = GaussianEmitter(archive, x0, sigma0, batch_size=batch_size)
    n = 100_000

    np.random.seed(0)

    objective_batch = np.random.rand(batch_size)
    measures_batch = np.random.rand(batch_size, 2)

    # Let numba compile.
    temp_sol = emitter.ask()

    # Add solutions to the archive.
    # TODO Replace with add_batch
    status_batch = []
    value_batch = []
    for (sol, obj, mea) in zip(temp_sol, objective_batch, measures_batch):
        status, value = archive.add(sol, obj, mea)
        status_batch.append(status)
        value_batch.append(value)
    status_batch = np.asarray(status_batch)
    value_batch = np.asarray(value_batch)

    emitter.tell(temp_sol, objective_batch, measures_batch, status_batch,
                 value_batch)

    obj_vals = np.random.rand(n, batch_size)
    behavior_vals = np.random.rand(n, batch_size, 2)

    def ask_and_tell():
        for i in range(n):
            solution_batch = emitter.ask()
            objective_batch = obj_vals[i]
            measure_batch = behavior_vals[i]
            # Add solutions to the archive.
            # TODO Replace with add_batch
            status_batch = []
            value_batch = []
            for (sol, obj, mea) in zip(solution_batch, objective_batch,
                                       measures_batch):
                status, value = archive.add(sol, obj, mea)
                status_batch.append(status)
                value_batch.append(value)
            status_batch = np.asarray(status_batch)
            value_batch = np.asarray(value_batch)
            emitter.tell(solution_batch, objective_batch, measure_batch,
                         status_batch, value_batch)

    benchmark(ask_and_tell)
