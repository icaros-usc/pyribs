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
