"""Tests for the Scheduler."""
import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.schedulers import Scheduler

from ..archives.grid_archive_test import assert_archive_elite_batch

# pylint: disable = redefined-outer-name


@pytest.fixture
def scheduler_fixture():
    """Returns a Scheduler with GridArchive and one GaussianEmitter."""
    solution_dim = 2
    num_solutions = 4
    archive = GridArchive(solution_dim, [100, 100], [(-1, 1), (-1, 1)])
    emitters = [
        GaussianEmitter(archive, [0.0, 0.0], 1, batch_size=num_solutions)
    ]
    return Scheduler(archive, emitters), solution_dim, num_solutions


def test_init_fails_with_no_emitters():
    # arbitrary sol_dim
    archive = GridArchive(10, [100, 100], [(-1, 1), (-1, 1)])
    emitters = []
    with pytest.raises(ValueError):
        Scheduler(archive, emitters)


def test_init_fails_on_non_unique_emitter_instances():
    archive = GridArchive(solution_dim=2,
                          dims=[100, 100],
                          ranges=[(-1, 1), (-1, 1)])

    # All emitters are the same instance. This is bad because the same emitter
    # gets called multiple times.
    emitters = [GaussianEmitter(archive, [0.0, 0.0], 1, batch_size=1)] * 5

    with pytest.raises(ValueError):
        Scheduler(archive, emitters)


def test_init_fails_with_mismatched_emitters():
    archive = GridArchive(2, [100, 100], [(-1, 1), (-1, 1)])
    emitters = [
        # Emits 2D solutions.
        GaussianEmitter(archive, [0.0, 0.0], 1),
        # Mismatch -- emits 3D solutions rather than 2D solutions.
        GaussianEmitter(archive, [0.0, 0.0, 0.0], 1),
    ]
    with pytest.raises(ValueError):
        Scheduler(archive, emitters)


def test_ask_returns_correct_solution_shape(scheduler_fixture):
    scheduler, solution_dim, num_solutions = scheduler_fixture
    solutions = scheduler.ask()
    assert solutions.shape == (num_solutions, solution_dim)


def test_ask_fails_when_called_twice(scheduler_fixtures):
    scheduler, *_ = scheduler_fixtures
    with pytest.raises(RuntimeError):
        scheduler.ask()
        scheduler.ask()


@pytest.mark.parametrize("add_mode", ["batch", "single"],
                         ids=["batch_add", "single_add"])
@pytest.mark.parametrize("tell_metadata", [True, False],
                         ids=["metadata", "no_metadata"])
def test_tell_inserts_solutions_into_archive(add_mode, tell_metadata):
    batch_size = 4
    archive = GridArchive(2, [100, 100], [(-1, 1), (-1, 1)])
    emitters = [GaussianEmitter(archive, [0.0, 0.0], 1, batch_size=batch_size)]
    scheduler = Scheduler(archive, emitters, add_mode=add_mode)

    measures_batch = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
    metadata = ([f"metadata_{i}" for i in range(batch_size)]
                if tell_metadata else None)
    expected_metadata = metadata if tell_metadata else [None] * batch_size

    _ = scheduler.ask()  # Ignore the actual values of the solutions.
    # We pass in 4 solutions with unique measures, so all should go into
    # the archive.
    scheduler.tell(np.ones(batch_size), measures_batch, metadata)

    assert_archive_elite_batch(
        archive=scheduler.archive,
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
    scheduler = Scheduler(archive, emitters, add_mode=add_mode)

    # The sum of all the emitters' batch sizes is 6.
    batch_size = 6
    measures_batch = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0],
                      [0.0, 0.0], [0.0, 1.0]]
    metadata = [f"metadata_{i}" for i in range(batch_size)
               ] if tell_metadata else None
    expected_metadata = metadata if tell_metadata else [None] * batch_size

    _ = scheduler.ask()
    scheduler.tell(np.ones(batch_size), measures_batch, metadata)

    assert_archive_elite_batch(
        archive=scheduler.archive,
        batch_size=batch_size,
        objective_batch=np.ones(batch_size),
        measures_batch=measures_batch,
        metadata_batch=expected_metadata,
    )


### TESTS FOR OUT-OF-ORDER ASK-TELL ###


def test_tell_fails_when_ask_not_called(scheduler_fixtures):
    scheduler, *_ = scheduler_fixtures
    with pytest.raises(RuntimeError):
        scheduler.tell(None, None)


def test_tell_fails_when_ask_dqd_not_called(scheduler_fixtures):
    scheduler, *_ = scheduler_fixtures
    with pytest.raises(RuntimeError):
        scheduler.tell_dqd(None, None, None)


def test_tell_fails_when_ask_tell_mismatch(scheduler_fixtures):
    scheduler, *_ = scheduler_fixtures

    _ = scheduler.ask()
    with pytest.raises(RuntimeError):
        scheduler.tell_dqd(None, None, None)


def test_tell_fails_when_ask_tell_mismatch_dqd(scheduler_fixtures):
    scheduler, *_ = scheduler_fixtures

    _ = scheduler.ask_dqd()
    with pytest.raises(RuntimeError):
        scheduler.tell(None, None)


### END ###


def test_emitter_returns_no_solutions(scheduler_fixtures):
    scheduler, solution_dim, _ = scheduler_fixtures

    # Should not return anything since there are no DQD emitters
    solution_batch = scheduler.ask_dqd()

    assert not np.any(solution_batch)
    assert solution_batch.shape == (0, solution_dim)


@pytest.mark.parametrize("array",
                         ["objective_batch", "measures_batch", "metadata"])
def test_tell_fails_with_wrong_shapes(scheduler_fixtures, array):
    scheduler, _, num_solutions = scheduler_fixtures
    _ = scheduler.ask()  # Ignore the actual values of the solutions.

    objective_batch = np.ones(num_solutions)
    measures_batch = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
    metadata = [f"metadata_{i}" for i in range(num_solutions)]

    # Each condition makes a certain array have the wrong shape by excluding the
    # last element.
    with pytest.raises(ValueError):
        if array == "objective_batch":
            scheduler.tell(objective_batch[:-1], measures_batch, metadata)
        elif array == "measures_batch":
            scheduler.tell(objective_batch, measures_batch[:-1], metadata)
        elif array == "metadata":
            scheduler.tell(objective_batch, measures_batch, metadata[:-1])
