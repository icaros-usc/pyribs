"""Tests for the Optimizer."""
import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.optimizers import Optimizer

# pylint: disable = redefined-outer-name


@pytest.fixture
def optimizer_fixture():
    """Returns an Optimizer with GridArchive and one GaussianEmitter."""
    solution_dim = 2
    num_solutions = 4
    archive = GridArchive([100, 100], [(-1, 1), (-1, 1)])
    emitters = [
        GaussianEmitter(archive, [0.0, 0.0], 1, batch_size=num_solutions)
    ]
    return Optimizer(archive, emitters), solution_dim, num_solutions


def test_init_fails_with_no_emitters():
    archive = GridArchive([100, 100], [(-1, 1), (-1, 1)])
    emitters = []
    with pytest.raises(ValueError):
        Optimizer(archive, emitters)


def test_init_fails_on_non_unique_emitter_instances():
    archive = GridArchive([100, 100], [(-1, 1), (-1, 1)])

    # All emitters are the same instance. This is bad because the same emitter
    # gets called multiple times.
    emitters = [GaussianEmitter(archive, [0.0, 0.0], 1, batch_size=1)] * 5

    with pytest.raises(ValueError):
        Optimizer(archive, emitters)


def test_init_fails_with_mismatched_emitters():
    archive = GridArchive([100, 100], [(-1, 1), (-1, 1)])
    emitters = [
        # Emits 2D solutions.
        GaussianEmitter(archive, [0.0, 0.0], 1),
        # Mismatch -- emits 3D solutions rather than 2D solutions.
        GaussianEmitter(archive, [0.0, 0.0, 0.0], 1),
    ]
    with pytest.raises(ValueError):
        Optimizer(archive, emitters)


def test_ask_returns_correct_solution_shape(optimizer_fixture):
    optimizer, solution_dim, num_solutions = optimizer_fixture
    solutions = optimizer.ask()
    assert solutions.shape == (num_solutions, solution_dim)


def test_ask_fails_when_called_twice(optimizer_fixture):
    optimizer, *_ = optimizer_fixture
    with pytest.raises(RuntimeError):
        optimizer.ask()
        optimizer.ask()


@pytest.mark.parametrize("tell_metadata", [True, False],
                         ids=["metadata", "no_metadata"])
def test_tell_inserts_solutions_into_archive(optimizer_fixture, tell_metadata):
    optimizer, _, num_solutions = optimizer_fixture
    _ = optimizer.ask()  # Ignore the actual values of the solutions.
    behavior_values = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
    metadata = ([f"metadata_{i}" for i in range(num_solutions)]
                if tell_metadata else None)
    expected_metadata = metadata if tell_metadata else [None] * num_solutions

    # We pass in 4 solutions with unique behavior values, so all should go into
    # the archive.
    optimizer.tell(
        objective_values=np.ones(num_solutions),
        behavior_values=behavior_values,
        metadata=metadata,
    )

    # Note: This assumes elite data is in order of insertion, which may change
    # in the future.
    assert len(optimizer.archive) == num_solutions
    df = optimizer.archive.as_pandas(include_metadata=True)
    assert (df.batch_behaviors() == behavior_values).all()
    assert (df.batch_objectives() == np.ones(num_solutions)).all()
    assert (df.batch_metadata() == expected_metadata).all()


@pytest.mark.parametrize("tell_metadata", [True, False],
                         ids=["metadata", "no_metadata"])
def test_tell_inserts_solutions_with_multiple_emitters(tell_metadata):
    archive = GridArchive([100, 100], [(-1, 1), (-1, 1)])
    emitters = [
        GaussianEmitter(archive, [0.0, 0.0], 1, batch_size=1),
        GaussianEmitter(archive, [0.5, 0.5], 1, batch_size=2),
        GaussianEmitter(archive, [-0.5, -0.5], 1, batch_size=3),
    ]
    optimizer = Optimizer(archive, emitters)

    _ = optimizer.ask()
    behavior_values = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0],
                       [0.0, 0.0], [0.0, 1.0]]
    metadata = [f"metadata_{i}" for i in range(6)] if tell_metadata else None
    expected_metadata = metadata if tell_metadata else [None] * 6

    # The sum of all the emitters' batch sizes is 6.
    optimizer.tell(
        objective_values=np.ones(6),
        behavior_values=behavior_values,
        metadata=metadata,
    )

    # Note: This assumes elite data is in order of insertion, which may change
    # in the future.
    assert len(optimizer.archive) == 6
    df = optimizer.archive.as_pandas(include_metadata=True)
    assert (df.batch_behaviors() == behavior_values).all()
    assert (df.batch_objectives() == np.ones(6)).all()
    assert (df.batch_metadata() == expected_metadata).all()


def test_tell_fails_when_ask_not_called(optimizer_fixture):
    optimizer, *_ = optimizer_fixture
    with pytest.raises(RuntimeError):
        optimizer.tell(None, None)


@pytest.mark.parametrize("array",
                         ["objective_values", "behavior_values", "metadata"])
def test_tell_fails_with_wrong_shapes(optimizer_fixture, array):
    optimizer, _, num_solutions = optimizer_fixture
    _ = optimizer.ask()  # Ignore the actual values of the solutions.

    objective_values = np.ones(num_solutions)
    behavior_values = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
    metadata = [f"metadata_{i}" for i in range(num_solutions)]

    # Each condition makes a certain array have the wrong shape by excluding the
    # last element.
    with pytest.raises(ValueError):
        if array == "objective_values":
            optimizer.tell(
                objective_values=objective_values[:-1],
                behavior_values=behavior_values,
                metadata=metadata,
            )
        elif array == "behavior_values":
            optimizer.tell(
                objective_values=objective_values,
                behavior_values=behavior_values[:-1],
                metadata=metadata,
            )
        elif array == "metadata":
            optimizer.tell(
                objective_values=objective_values,
                behavior_values=behavior_values,
                metadata=metadata[:-1],
            )
