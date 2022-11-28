"""Tests for EvolutionStrategyEmitter."""
import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter

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
    archive = GridArchive(solution_dim=len(x0),
                          dims=[10, 10],
                          ranges=[(-1, 1), (-1, 1)])
    emitter = EvolutionStrategyEmitter(archive,
                                       x0=x0,
                                       sigma0=1.0,
                                       ranker=request.param[0],
                                       es=request.param[1])

    return emitter, archive, x0


@pytest.mark.parametrize("emitter_fixture", [("imp", es) for es in ES_LIST],
                         ids=[f"imp-{es}" for es in ES_LIST],
                         indirect=True)
def test_auto_batch_size(emitter_fixture):
    emitter, *_ = emitter_fixture
    assert emitter.batch_size is not None
    assert isinstance(emitter.batch_size, int)


def test_list_as_initial_solution():
    archive = GridArchive(solution_dim=10,
                          dims=[20, 20],
                          ranges=[(-1.0, 1.0)] * 2)
    emitter = EvolutionStrategyEmitter(archive,
                                       x0=[0.0] * 10,
                                       sigma0=1.0,
                                       ranker="obj")

    # The list was passed in but should be converted to a numpy array.
    assert isinstance(emitter.x0, np.ndarray)
    assert (emitter.x0 == np.zeros(10)).all()


@pytest.mark.parametrize("dtype", [np.float64, np.float32],
                         ids=["float64", "float32"])
def test_dtypes(dtype):
    archive = GridArchive(solution_dim=10,
                          dims=[20, 20],
                          ranges=[(-1.0, 1.0)] * 2,
                          dtype=dtype)
    emitter = EvolutionStrategyEmitter(archive,
                                       x0=np.zeros(10),
                                       sigma0=1.0,
                                       ranker="obj")
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
