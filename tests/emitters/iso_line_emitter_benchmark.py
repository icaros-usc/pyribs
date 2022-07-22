"""Benchmarks for the IsoLineEmitter."""

import numpy as np

from ribs.emitters import IsoLineEmitter


def benchmark_ask_tell_100k(benchmark, fake_archive_fixture):
    archive, x0 = fake_archive_fixture
    batch_size = 32
    emitter = IsoLineEmitter(archive, x0, batch_size=batch_size)
    n = 100_000

    np.random.seed(0)

    objective_batch = np.random.rand(batch_size)
    measures_batch = np.random.rand(batch_size, 2)

    # Let numba compile.
    temp_sol = emitter.ask()
    status_batch, value_batch = archive.add(temp_sol, objective_batch,
                                            measures_batch)
    emitter.tell(temp_sol, objective_batch, measures_batch, status_batch,
                 value_batch)

    all_objective_batch = np.random.rand(n, batch_size)
    all_measures_batch = np.random.rand(n, batch_size, 2)

    def ask_and_tell():
        for i in range(n):
            solution_batch = emitter.ask()
            objective_batch = all_objective_batch[i]
            measures_batch = all_measures_batch[i]

            status_batch, value_batch = archive.add(solution_batch,
                                                    objective_batch,
                                                    measures_batch)
            emitter.tell(solution_batch, objective_batch, measures_batch,
                         status_batch, value_batch)

    benchmark(ask_and_tell)
