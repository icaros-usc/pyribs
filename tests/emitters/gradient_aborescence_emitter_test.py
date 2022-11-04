"""Tests for GradientAborescenceEmitter."""
import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.emitters import GradientAborescenceEmitter


def test_auto_batch_size():
    archive = GridArchive(solution_dim=10,
                          dims=[20, 20],
                          ranges=[(-1.0, 1.0)] * 2)
    emitter = GradientAborescenceEmitter(archive,
                                         x0=np.zeros(10),
                                         sigma0=1.0,
                                         step_size=1.0)
    assert emitter.batch_size is not None
    assert isinstance(emitter.batch_size, int)


def test_list_as_initial_solution():
    archive = GridArchive(solution_dim=10,
                          dims=[20, 20],
                          ranges=[(-1.0, 1.0)] * 2)
    emitter = GradientAborescenceEmitter(archive,
                                         x0=[0.0] * 10,
                                         sigma0=1.0,
                                         step_size=1.0)

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
    emitter = GradientAborescenceEmitter(archive,
                                         x0=np.zeros(10),
                                         sigma0=1.0,
                                         step_size=1.0)
    assert emitter.x0.dtype == dtype


def test_adhere_to_solution_bounds():
    bound = [(-1,1)]
    batch_size = 3
    archive = GridArchive(solution_dim=1,
                          dims=[10],
                          ranges=[(-1.0, 1.0)])
    emitter = GradientAborescenceEmitter(archive,
                                         x0=np.array([0]),
                                         sigma0=1.0,
                                         step_size=1.0,
                                         normalize_grad=False,
                                         bounds=bound,
                                         batch_size=batch_size)

    # Set jacobian so tell_dqd doesn't crash.
    jacobian = np.full((batch_size, 2, 1), 2)
    emitter.tell_dqd([0], [0], [0], jacobian, [0], [0])

    # This might take a while because it needs to resample.
    sol = emitter.ask()

    assert np.all(np.logical_and(sol >= bound[0][0], sol <= bound[0][1]))
