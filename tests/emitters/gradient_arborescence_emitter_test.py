"""Tests for GradientArborescenceEmitter."""
import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.emitters import GradientArborescenceEmitter


def test_auto_batch_size():
    archive = GridArchive(solution_dim=10,
                          dims=[20, 20],
                          ranges=[(-1.0, 1.0)] * 2)
    emitter = GradientArborescenceEmitter(archive,
                                          x0=np.zeros(10),
                                          sigma0=1.0,
                                          step_size=1.0)
    assert emitter.batch_size is not None
    assert isinstance(emitter.batch_size, int)


def test_list_as_initial_solution():
    archive = GridArchive(solution_dim=10,
                          dims=[20, 20],
                          ranges=[(-1.0, 1.0)] * 2)
    emitter = GradientArborescenceEmitter(archive,
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
    emitter = GradientArborescenceEmitter(archive,
                                          x0=np.zeros(10),
                                          sigma0=1.0,
                                          step_size=1.0)
    assert emitter.x0.dtype == dtype


def test_adhere_to_solution_bounds():
    bound = [(-1, 1)]
    batch_size = 1
    archive = GridArchive(solution_dim=1, dims=[10], ranges=[(-1.0, 1.0)])
    emitter = GradientArborescenceEmitter(archive,
                                          x0=np.array([0]),
                                          sigma0=1.0,
                                          step_size=1.0,
                                          normalize_grad=False,
                                          bounds=bound,
                                          batch_size=batch_size)

    # Set jacobian so tell_dqd doesn't crash.
    jacobian = np.full(
        (
            1,  # Only one solution.
            2,  # Two gradients -- one objective, one measures.
            1,  # One solution dimension for each gradient.
        ),
        2,  # Each value is 2.0.
    )
    emitter.tell_dqd(
        np.zeros((batch_size, archive.solution_dim)),
        np.zeros(batch_size),
        np.zeros((batch_size, archive.measure_dim)),
        jacobian,
        np.zeros(batch_size),
        np.zeros(batch_size),
    )

    # This might take a while because it needs to resample.
    sol = emitter.ask()

    assert np.all(np.logical_and(sol >= bound[0][0], sol <= bound[0][1]))
