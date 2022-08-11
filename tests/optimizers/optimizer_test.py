"""Tests for the Optimizer."""
import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.optimizers import Optimizer

from ..archives.grid_archive_test import assert_archive_elite_batch

# pylint: disable = redefined-outer-name


@pytest.fixture
def optimizer_fixture():
    """Returns an Optimizer with GridArchive and one GaussianEmitter."""
    solution_dim = 2
    num_solutions = 4
    archive = GridArchive(solution_dim, [100, 100], [(-1, 1), (-1, 1)])
    emitters = [
        GaussianEmitter(archive, [0.0, 0.0], 1, batch_size=num_solutions)
    ]
    return Optimizer(archive, emitters), solution_dim, num_solutions


def test_init_fails_with_no_emitters():
    # arbitrary sol_dim
    archive = GridArchive(10, [100, 100], [(-1, 1), (-1, 1)])
    emitters = []
    with pytest.raises(ValueError):
        Optimizer(archive, emitters)


def test_init_fails_on_non_unique_emitter_instances():
    archive = GridArchive(solution_dim=2,
                          dims=[100, 100],
                          ranges=[(-1, 1), (-1, 1)])

    # All emitters are the same instance. This is bad because the same emitter
    # gets called multiple times.
    emitters = [GaussianEmitter(archive, [0.0, 0.0], 1, batch_size=1)] * 5

    with pytest.raises(ValueError):
        Optimizer(archive, emitters)


def test_init_fails_with_mismatched_emitters():
    archive = GridArchive(2, [100, 100], [(-1, 1), (-1, 1)])
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


@pytest.mark.parametrize("add_mode", ["batch", "single"],
                         ids=["batch_add", "single_add"])
@pytest.mark.parametrize("tell_metadata", [True, False],
                         ids=["metadata", "no_metadata"])
def test_tell_inserts_solutions_into_archive(add_mode, tell_metadata):
    batch_size = 4
    archive = GridArchive(2, [100, 100], [(-1, 1), (-1, 1)])
    emitters = [GaussianEmitter(archive, [0.0, 0.0], 1, batch_size=batch_size)]
    optimizer = Optimizer(archive, emitters, add_mode=add_mode)

    measures_batch = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
    metadata = ([f"metadata_{i}" for i in range(batch_size)]
                if tell_metadata else None)
    expected_metadata = metadata if tell_metadata else [None] * batch_size

    _ = optimizer.ask()  # Ignore the actual values of the solutions.
    # We pass in 4 solutions with unique behavior values, so all should go into
    # the archive.
    optimizer.tell(np.ones(batch_size), measures_batch, metadata)

    assert_archive_elite_batch(
        archive=optimizer.archive,
        batch_size=batch_size,
        objective_batch=np.ones(batch_size),
        measures_batch=measures_batch,
        metadata_batch=expected_metadata,
    )


@pytest.mark.parametrize("add_mode", ["batch", "single"],
                         ids=["batch_add", "single_add"])
@pytest.mark.parametrize("tell_metadata", [True, False],
                         ids=["metadata", "no_metadata"])
def test_tell_inserts_solutions_with_multiple_emitters(add_mode, tell_metadata):
    archive = GridArchive(2, [100, 100], [(-1, 1), (-1, 1)])
    emitters = [
        GaussianEmitter(archive, [0.0, 0.0], 1, batch_size=1),
        GaussianEmitter(archive, [0.5, 0.5], 1, batch_size=2),
        GaussianEmitter(archive, [-0.5, -0.5], 1, batch_size=3),
    ]
    optimizer = Optimizer(archive, emitters, add_mode=add_mode)

    # The sum of all the emitters' batch sizes is 6.
    batch_size = 6
    measures_batch = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0],
                      [0.0, 0.0], [0.0, 1.0]]
    metadata = [f"metadata_{i}" for i in range(batch_size)
               ] if tell_metadata else None
    expected_metadata = metadata if tell_metadata else [None] * batch_size

    _ = optimizer.ask()
    optimizer.tell(np.ones(batch_size), measures_batch, metadata)

    assert_archive_elite_batch(
        archive=optimizer.archive,
        batch_size=batch_size,
        objective_batch=np.ones(batch_size),
        measures_batch=measures_batch,
        metadata_batch=expected_metadata,
    )


### TESTS FOR OUT-OF-ORDER ASK-TELL ###


def test_tell_fails_when_ask_not_called(optimizer_fixture):
    optimizer, *_ = optimizer_fixture
    with pytest.raises(RuntimeError):
        optimizer.tell(None, None)


def test_tell_fails_when_ask_dqd_not_called(optimizer_fixture):
    optimizer, *_ = optimizer_fixture
    with pytest.raises(RuntimeError):
        optimizer.tell_dqd(None, None, None)


def test_tell_fails_when_ask_tell_mismatch(optimizer_fixture):
    optimizer, *_ = optimizer_fixture

    _ = optimizer.ask()
    with pytest.raises(RuntimeError):
        optimizer.tell_dqd(None, None, None)


def test_tell_fails_when_ask_tell_mismatch_dqd(optimizer_fixture):
    optimizer, *_ = optimizer_fixture

    _ = optimizer.ask_dqd()
    with pytest.raises(RuntimeError):
        optimizer.tell(None, None)


### END ###


def test_emitter_returns_no_solutions(optimizer_fixture):
    optimizer, solution_dim, _ = optimizer_fixture

    # Should not return anything since there are no DQD emitters
    solution_batch = optimizer.ask_dqd()

    assert not np.any(solution_batch)
    assert solution_batch.shape == (0, solution_dim)


@pytest.mark.parametrize("array",
                         ["objective_values", "behavior_values", "metadata"])
def test_tell_fails_with_wrong_shapes(optimizer_fixture, array):
    optimizer, _, num_solutions = optimizer_fixture
    _ = optimizer.ask()  # Ignore the actual values of the solutions.

    objective_values = np.ones(num_solutions)
    measures_batch = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
    metadata = [f"metadata_{i}" for i in range(num_solutions)]

    # Each condition makes a certain array have the wrong shape by excluding the
    # last element.
    with pytest.raises(ValueError):
        if array == "objective_values":
            optimizer.tell(objective_values[:-1], measures_batch, metadata)
        elif array == "behavior_values":
            optimizer.tell(objective_values, measures_batch[:-1], metadata)
        elif array == "metadata":
            optimizer.tell(objective_values, measures_batch, metadata[:-1])
