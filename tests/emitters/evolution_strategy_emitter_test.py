"""Tests for EvolutionStrategyEmitter."""
import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.emitters.rankers import get_ranker


def test_auto_batch_size():
    ranker = get_ranker("obj")
    archive = GridArchive(10, [20, 20], [(-1.0, 1.0)] * 2)
    emitter = EvolutionStrategyEmitter(archive, np.zeros(10), 1.0, ranker)
    assert emitter.batch_size is not None
    assert isinstance(emitter.batch_size, int)


def test_list_as_initial_solution():
    ranker = get_ranker("obj")
    archive = GridArchive(10, [20, 20], [(-1.0, 1.0)] * 2)
    emitter = EvolutionStrategyEmitter(archive, [0.0] * 10, 1.0, ranker)

    # The list was passed in but should be converted to a numpy array.
    assert isinstance(emitter.x0, np.ndarray)
    assert (emitter.x0 == np.zeros(10)).all()


@pytest.mark.parametrize("dtype", [np.float64, np.float32],
                         ids=["float64", "float32"])
def test_dtypes(dtype):
    ranker = get_ranker("obj")
    archive = GridArchive(10, [20, 20], [(-1.0, 1.0)] * 2, dtype=dtype)
    emitter = EvolutionStrategyEmitter(archive, np.zeros(10), 1.0, ranker)
    assert emitter.x0.dtype == dtype

    # Try running with the negative sphere function for a few iterations.
    for _ in range(10):
        solution_batch = emitter.ask()
        objective_batch = -np.sum(np.square(solution_batch), axis=1)
        measures_batch = solution_batch[:, :2]

        status_batch, value_batch = archive.add(solution_batch, objective_batch,
                                                measures_batch)
        emitter.tell(solution_batch, objective_batch, measures_batch,
                     status_batch, value_batch)


def test_simga0_vector():
    ranker = get_ranker("obj")
    archive = GridArchive(10, [20, 20], [(-1.0, 1.0)] * 2)
    emitter = EvolutionStrategyEmitter(archive, np.zeros(10), [1.0, 1.0],
                                       ranker)

    # Try running with the negative sphere function for a few iterations.
    for _ in range(10):
        solution_batch = emitter.ask()
        objective_batch = -np.sum(np.square(solution_batch), axis=1)
        measures_batch = solution_batch[:, :2]

        status_batch, value_batch = archive.add(solution_batch, objective_batch,
                                                measures_batch)
        emitter.tell(solution_batch, objective_batch, measures_batch,
                     status_batch, value_batch)
