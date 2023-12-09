"""Tests for GradientArborescenceEmitter."""
import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.emitters import GradientArborescenceEmitter

ES_LIST = ["cma_es", "sep_cma_es", "lm_ma_es", "openai_es"]


def test_auto_batch_size():
    archive = GridArchive(solution_dim=10,
                          dims=[20, 20],
                          ranges=[(-1.0, 1.0)] * 2)
    emitter = GradientArborescenceEmitter(archive,
                                          x0=np.zeros(10),
                                          sigma0=1.0,
                                          lr=1.0)
    assert emitter.batch_size is not None
    assert isinstance(emitter.batch_size, int)


def test_list_as_initial_solution():
    archive = GridArchive(solution_dim=10,
                          dims=[20, 20],
                          ranges=[(-1.0, 1.0)] * 2)
    emitter = GradientArborescenceEmitter(archive,
                                          x0=[0.0] * 10,
                                          sigma0=1.0,
                                          lr=1.0)

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
    emitter = GradientArborescenceEmitter(archive,
                                          x0=np.zeros(10),
                                          sigma0=1.0,
                                          lr=1.0)
    assert emitter.x0.dtype == dtype


def test_bounds_must_be_none():
    bound = [(-1, 1)]
    batch_size = 1
    archive = GridArchive(solution_dim=1, dims=[10], ranges=[(-1.0, 1.0)])

    with pytest.raises(ValueError):
        GradientArborescenceEmitter(archive,
                                    x0=np.array([0]),
                                    sigma0=1.0,
                                    lr=1.0,
                                    normalize_grad=False,
                                    bounds=bound,
                                    batch_size=batch_size)


def test_ask_dqd_must_be_called_before_ask():
    archive = GridArchive(solution_dim=1, dims=[10], ranges=[(-1.0, 1.0)])
    with pytest.raises(RuntimeError):
        emitter = GradientArborescenceEmitter(archive,
                                              x0=np.array([0]),
                                              sigma0=1.0,
                                              lr=1.0)
        # Must call ask_dqd() before calling ask() to set the jacobian.
        emitter.ask()


def test_tell_dqd_must_be_called_before_tell():
    archive = GridArchive(solution_dim=1, dims=[10], ranges=[(-1.0, 1.0)])
    with pytest.raises(RuntimeError):
        emitter = GradientArborescenceEmitter(archive,
                                              x0=np.array([0]),
                                              sigma0=1.0,
                                              lr=1.0)
        # Must call ask_dqd() before calling ask() to set the jacobian.
        emitter.tell([[0]], [0], [[0]], {"status": [0], "value": [0]})


@pytest.mark.parametrize("es", ES_LIST)
def test_sphere(es):
    archive = GridArchive(solution_dim=10,
                          dims=[20, 20],
                          ranges=[(-1.0, 1.0)] * 2)
    emitter = GradientArborescenceEmitter(
        archive,
        x0=np.zeros(10),
        sigma0=1.0,
        lr=1.0,
        # Must be 2 to accommodate restrictions from LM-MA-ES.
        batch_size=2,
        es=es,
    )

    # Try running with the negative sphere function for a few iterations.
    for _ in range(5):
        solution = emitter.ask_dqd()
        objective = -np.sum(np.square(solution), axis=1)
        measures = solution[:, :2]
        jacobian = np.random.uniform(-1, 1, (1, 3, 10))
        add_info = archive.add(solution, objective, measures)
        emitter.tell_dqd(solution, objective, measures, jacobian, add_info)

        solution = emitter.ask()
        objective = -np.sum(np.square(solution), axis=1)
        measures = solution[:, :2]
        add_info = archive.add(solution, objective, measures)
        emitter.tell(solution, objective, measures, add_info)
