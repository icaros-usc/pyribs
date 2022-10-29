"""Tests for GradientAborescenceEmitter."""
import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.emitters import GradientAborescenceEmitter


def test_auto_batch_size():
    archive = GridArchive(10, [20, 20], [(-1.0, 1.0)] * 2)
    emitter = GradientAborescenceEmitter(archive, np.zeros(10), 1.0, 1.0)
    assert emitter.batch_size is not None
    assert isinstance(emitter.batch_size, int)


def test_list_as_initial_solution():
    archive = GridArchive(10, [20, 20], [(-1.0, 1.0)] * 2)
    emitter = GradientAborescenceEmitter(archive, [0.0] * 10, 1.0, 1.0)

    # The list was passed in but should be converted to a numpy array.
    assert isinstance(emitter.x0, np.ndarray)
    assert (emitter.x0 == np.zeros(10)).all()


@pytest.mark.parametrize("dtype", [np.float64, np.float32],
                         ids=["float64", "float32"])
def test_dtypes(dtype):
    archive = GridArchive(10, [20, 20], [(-1.0, 1.0)] * 2, dtype=dtype)
    emitter = GradientAborescenceEmitter(archive, np.zeros(10), 1.0, 1.0)
    assert emitter.x0.dtype == dtype


def test_adhere_to_solution_bounds():
    bound = [(-1,1)]
    archive = GridArchive(10, [20, 20], [(-1.0, 1.0)] * 2)
    emitter = GradientAborescenceEmitter(archive,
                                         np.zeros(2),
                                         1.0,
                                         1.0,
                                         normalize_grad=False,
                                         bounds=bound * 2)

    # Set jacobian so tell_dqd doesn't crash.
    jacobian = np.full((7, 3, 2), 1.5)
    emitter.tell_dqd([0], [0], [0], jacobian, [0], [0])

    # This will take a while because it needs to resample.
    sol = emitter.ask()

    assert np.all(np.logical_and(sol >= bound[0][0], sol <= bound[0][1]))
