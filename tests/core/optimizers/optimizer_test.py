"""Tests for the Optimizer."""

import pytest

from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.optimizers import Optimizer

# pylint: disable = invalid-name


@pytest.fixture
def _optimizer_fixture():
    """Returns an Optimizer with GridArchive and one GaussianEmitter."""
    solution_dim = 2
    archive = GridArchive([100, 100], [(-1, 1), (-1, 1)])
    emitters = [
        GaussianEmitter([0.0, 0.0], 1, archive, config={"batch_size": 4})
    ]
    return Optimizer(archive, emitters), solution_dim


def test_init_fails_with_no_emitters():
    archive = GridArchive([100, 100], [(-1, 1), (-1, 1)])
    emitters = []
    with pytest.raises(RuntimeError):
        Optimizer(archive, emitters)


def test_init_fails_with_mismatched_emitters():
    archive = GridArchive([100, 100], [(-1, 1), (-1, 1)])
    emitters = [
        # Emits 2D solutions.
        GaussianEmitter([0.0, 0.0], 1, archive),
        # Mismatch -- emits 3D solutions rather than 2D solutions.
        GaussianEmitter([0.0, 0.0, 0.0], 1, archive),
    ]
    with pytest.raises(RuntimeError):
        Optimizer(archive, emitters)


def test_ask_returns_correct_solution_shape(_optimizer_fixture):
    optimizer, solution_dim = _optimizer_fixture
    solutions = optimizer.ask()
    assert solutions.shape == (optimizer.emitters[0].batch_size, solution_dim)


def test_ask_fails_when_called_twice(_optimizer_fixture):
    optimizer, _ = _optimizer_fixture
    with pytest.raises(RuntimeError):
        optimizer.ask()
        optimizer.ask()


def test_tell_inserts_solutions_into_archive(_optimizer_fixture):
    optimizer, _ = _optimizer_fixture
    _ = optimizer.ask()  # Ignore the actual values of the solutions.

    # Batch size is 4, so we need to pass in 4 objective values and behavior
    # values. Since the behavior values are all different, all 4 solutions
    # should go into the archive.
    optimizer.tell(
        objective_values=[1.0, 1.0, 1.0, 1.0],
        behavior_values=[[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]],
    )

    num_solutions = optimizer.emitters[0].batch_size
    assert len(optimizer.archive.as_pandas()) == num_solutions


def test_tell_inserts_solutions_with_multiple_emitters(_optimizer_fixture):
    archive = GridArchive([100, 100], [(-1, 1), (-1, 1)])
    emitters = [
        GaussianEmitter([0.0, 0.0], 1, archive, config={"batch_size": 1}),
        GaussianEmitter([0.5, 0.5], 1, archive, config={"batch_size": 2}),
        GaussianEmitter([-0.5, -0.5], 1, archive, config={"batch_size": 3}),
    ]
    optimizer = Optimizer(archive, emitters)

    _ = optimizer.ask()

    # The sum of all the emitters' batch sizes is 6.
    optimizer.tell(
        objective_values=[1.0] * 6,
        behavior_values=[[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0],
                         [0.0, 0.0], [0.0, 1.0]],
    )
    assert len(optimizer.archive.as_pandas()) == 6


def test_tell_fails_when_ask_not_called(_optimizer_fixture):
    optimizer, _ = _optimizer_fixture
    with pytest.raises(RuntimeError):
        optimizer.tell(None, None)
