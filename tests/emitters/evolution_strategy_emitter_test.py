"""Tests for EvolutionStrategyEmitter."""

import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter

# pylint: disable = redefined-outer-name

RANKER_LIST = ["imp", "2imp", "rd", "2rd", "obj", "2obj"]
ES_LIST = ["cma_es", "sep_cma_es", "lm_ma_es", "openai_es"]


@pytest.fixture
def emitter_fixture(request):
    """Creates an EvolutionStrategyEmitter for a particular ranker and
    evolution strategy.

    Returns:
        Tuple of (archive, emitter, batch_size, x0).
    """
    x0 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    archive = GridArchive(
        solution_dim=len(x0), dims=[10, 10], ranges=[(-1, 1), (-1, 1)]
    )
    emitter = EvolutionStrategyEmitter(
        archive, x0=x0, sigma0=1.0, ranker=request.param[0], es=request.param[1]
    )

    return emitter, archive, x0


@pytest.mark.parametrize(
    "emitter_fixture",
    [("imp", es) for es in ES_LIST],
    ids=[f"imp-{es}" for es in ES_LIST],
    indirect=True,
)
def test_auto_batch_size(emitter_fixture):
    emitter, *_ = emitter_fixture
    assert emitter.batch_size is not None
    assert isinstance(emitter.batch_size, int)


def test_list_as_initial_solution():
    archive = GridArchive(solution_dim=10, dims=[20, 20], ranges=[(-1.0, 1.0)] * 2)
    emitter = EvolutionStrategyEmitter(archive, x0=[0.0] * 10, sigma0=1.0)

    # The list was passed in but should be converted to a numpy array.
    assert isinstance(emitter.x0, np.ndarray)
    assert (emitter.x0 == np.zeros(10)).all()


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
def test_dtypes(dtype):
    archive = GridArchive(
        solution_dim=10, dims=[20, 20], ranges=[(-1.0, 1.0)] * 2, dtype=dtype
    )
    emitter = EvolutionStrategyEmitter(archive, x0=np.zeros(10), sigma0=1.0)
    assert emitter.x0.dtype == dtype


def test_seed_sequence():
    archive = GridArchive(
        solution_dim=10,
        dims=[20, 20],
        ranges=[(-1.0, 1.0)] * 2,
    )
    EvolutionStrategyEmitter(
        archive,
        x0=np.zeros(10),
        sigma0=1.0,
        # Passing a SeedSequence here used to throw a TypeError.
        seed=np.random.SeedSequence(42),
    )


@pytest.mark.parametrize("es", ES_LIST)
def test_sphere(es):
    archive = GridArchive(solution_dim=10, dims=[20, 20], ranges=[(-1.0, 1.0)] * 2)
    emitter = EvolutionStrategyEmitter(archive, x0=np.zeros(10), sigma0=1.0, es=es)

    # Try running with the negative sphere function for a few iterations.
    for _ in range(5):
        solution_batch = emitter.ask()
        objective_batch = -np.sum(np.square(solution_batch), axis=1)
        measures_batch = solution_batch[:, :2]
        add_info = archive.add(solution_batch, objective_batch, measures_batch)
        emitter.tell(solution_batch, objective_batch, measures_batch, add_info)


if __name__ == "__main__":
    # For testing bounds handling. Run this file:
    # python tests/emitters/evolution_strategy_emitter_test.py
    # The below code should show the resampling warning indicating that the ES
    # resampled too many times. This test cannot be included in pytest because
    # it is designed to hang. Comment out the different emitters to test
    # different ESs.

    archive = GridArchive(solution_dim=31, dims=[20, 20], ranges=[(-1.0, 1.0)] * 2)
    emitter = EvolutionStrategyEmitter(
        archive, x0=np.zeros(31), sigma0=1.0, bounds=[(0, 1.0)] * 31, es="cma_es"
    )
    #  emitter = EvolutionStrategyEmitter(archive,
    #                                     x0=np.zeros(31),
    #                                     sigma0=1.0,
    #                                     bounds=[(0, 1.0)] * 31,
    #                                     es="sep_cma_es")
    #  emitter = EvolutionStrategyEmitter(archive,
    #                                     x0=np.zeros(31),
    #                                     sigma0=1.0,
    #                                     bounds=[(0, 1.0)] * 31,
    #                                     es="lm_ma_es")
    #  emitter = EvolutionStrategyEmitter(archive,
    #                                     x0=np.zeros(31),
    #                                     sigma0=1.0,
    #                                     bounds=[(0, 1.0)] * 31,
    #                                     es="openai_es",
    #                                     es_kwargs={"mirror_sampling": False})

    emitter.ask()
