"""Tests for Scheduler and BanditScheduler."""
import numpy as np
import pytest

from ribs.archives import GridArchive, ProximityArchive
from ribs.emitters import GaussianEmitter
from ribs.schedulers import BanditScheduler, Scheduler

from ..archives.grid_archive_test import assert_archive_elites

# pylint: disable = redefined-outer-name


@pytest.fixture
def scheduler_fixture():
    """Returns a Scheduler with GridArchive and one GaussianEmitter."""
    solution_dim = 2
    num_solutions = 4
    archive = GridArchive(solution_dim=solution_dim,
                          dims=[100, 100],
                          ranges=[(-1, 1), (-1, 1)])
    emitters = [
        GaussianEmitter(archive,
                        sigma=1,
                        x0=[0.0, 0.0],
                        batch_size=num_solutions)
    ]
    return Scheduler(archive, emitters), solution_dim, num_solutions


@pytest.fixture(params=["single", "batch"])
def add_mode(request):
    """Single or batch add."""
    return request.param


@pytest.mark.parametrize("scheduler_type", ["Scheduler", "BanditScheduler"])
def test_attributes(scheduler_type):
    archive = GridArchive(
        solution_dim=2,
        dims=[100, 100],
        ranges=[(-1, 1), (-1, 1)],
        threshold_min=1.0,
        learning_rate=1.0,
    )
    emitters = [GaussianEmitter(archive, sigma=1, x0=[0.0, 0.0], batch_size=4)]

    if scheduler_type == "Scheduler":
        scheduler = Scheduler(archive, emitters)

        assert scheduler.archive == archive
        assert scheduler.emitters == emitters
    else:
        scheduler = BanditScheduler(archive, emitters, 1)

        assert scheduler.archive == archive
        assert scheduler.emitter_pool == emitters
        assert len(scheduler.active) == len(scheduler.emitter_pool)
        assert not np.any(scheduler.active)


def test_init_fails_with_non_list():
    archive = GridArchive(solution_dim=2,
                          dims=[100, 100],
                          ranges=[(-1, 1), (-1, 1)])

    # Just a single emitter not in a list.
    emitters = GaussianEmitter(archive, sigma=1, x0=[0.0, 0.0], batch_size=1)

    with pytest.raises(TypeError):
        Scheduler(archive, emitters)


def test_init_fails_with_no_emitters():
    # arbitrary sol_dim
    archive = GridArchive(solution_dim=10,
                          dims=[100, 100],
                          ranges=[(-1, 1), (-1, 1)])
    emitters = []
    with pytest.raises(ValueError):
        Scheduler(archive, emitters)


def test_init_fails_on_non_unique_emitter_instances():
    archive = GridArchive(solution_dim=2,
                          dims=[100, 100],
                          ranges=[(-1, 1), (-1, 1)])

    # All emitters are the same instance. This is bad because the same emitter
    # gets called multiple times.
    emitters = [GaussianEmitter(archive, sigma=1, x0=[0.0, 0.0], batch_size=1)
               ] * 5

    with pytest.raises(ValueError):
        Scheduler(archive, emitters)


def test_ask_returns_correct_solution_shape(scheduler_fixture):
    scheduler, solution_dim, num_solutions = scheduler_fixture
    solutions = scheduler.ask()
    assert solutions.shape == (num_solutions, solution_dim)


def test_ask_fails_when_called_twice(scheduler_fixture):
    scheduler, *_ = scheduler_fixture
    with pytest.raises(RuntimeError):
        scheduler.ask()
        scheduler.ask()


@pytest.mark.parametrize("scheduler_type", ["Scheduler", "BanditScheduler"])
def test_warn_nothing_added_to_archive(scheduler_type):
    archive = GridArchive(
        solution_dim=2,
        dims=[100, 100],
        ranges=[(-1, 1), (-1, 1)],
        threshold_min=1.0,
        learning_rate=1.0,
    )
    emitters = [GaussianEmitter(archive, sigma=1, x0=[0.0, 0.0], batch_size=4)]
    if scheduler_type == "Scheduler":
        scheduler = Scheduler(archive, emitters)
    else:
        scheduler = BanditScheduler(archive, emitters, 1)

    _ = scheduler.ask()
    with pytest.warns(UserWarning):
        scheduler.tell(
            # All objectives are below threshold_min of 1.0.
            objective=np.zeros(4),
            # Arbitrary measures.
            measures=np.linspace(-1, 1, 4 * 2).reshape((4, 2)),
        )


@pytest.mark.parametrize("scheduler_type", ["Scheduler", "BanditScheduler"])
def test_warn_nothing_added_to_result_archive(scheduler_type):
    archive = GridArchive(
        solution_dim=2,
        dims=[100, 100],
        ranges=[(-1, 1), (-1, 1)],
        threshold_min=-np.inf,
        learning_rate=1.0,
    )
    result_archive = GridArchive(
        solution_dim=2,
        dims=[100, 100],
        ranges=[(-1, 1), (-1, 1)],
        threshold_min=10.0,
        learning_rate=1.0,
    )
    emitters = [GaussianEmitter(archive, sigma=1, x0=[0.0, 0.0], batch_size=4)]
    if scheduler_type == "Scheduler":
        scheduler = Scheduler(
            archive,
            emitters,
            result_archive=result_archive,
        )
    else:
        scheduler = BanditScheduler(
            archive,
            emitters,
            1,
            result_archive=result_archive,
        )

    _ = scheduler.ask()
    with pytest.warns(UserWarning):
        scheduler.tell(
            # All objectives are below threshold_min of 1.0.
            objective=np.zeros(4),
            # Arbitrary measures.
            measures=np.linspace(-1, 1, 4 * 2).reshape((4, 2)),
        )


@pytest.mark.parametrize("scheduler_type", ["Scheduler", "BanditScheduler"])
def test_result_archive_mismatch_fields(scheduler_type):
    archive = GridArchive(
        solution_dim=2,
        dims=[100, 100],
        ranges=[(-1, 1), (-1, 1)],
        threshold_min=-np.inf,
        learning_rate=1.0,
        extra_fields={
            "metadata": ((), object),
            "square": ((2, 2), np.int32)
        },
    )
    result_archive = GridArchive(solution_dim=2,
                                 dims=[100, 100],
                                 ranges=[(-1, 1), (-1, 1)])
    emitters = [GaussianEmitter(archive, sigma=1, x0=[0.0, 0.0], batch_size=4)]

    if scheduler_type == "Scheduler":
        scheduler = Scheduler(
            archive,
            emitters,
            result_archive=result_archive,
        )
    else:
        scheduler = BanditScheduler(
            archive,
            emitters,
            1,
            result_archive=result_archive,
        )

    scheduler.ask()

    # The ArrayStore in the archives should throw an error when we try to add.
    with pytest.raises(ValueError):
        scheduler.tell(
            np.zeros(4),
            np.zeros((4, 2)),
            metadata=np.zeros(4, dtype=object),
            square=np.zeros((4, 2, 2)),
        )


@pytest.mark.parametrize("scheduler_type", ["Scheduler", "BanditScheduler"])
def test_result_archive_same_fields_with_threshold(scheduler_type):
    """GridArchive has a threshold field and ProximityArchive does not, but they
    should still operate together because the extra_fields are identical."""
    archive = ProximityArchive(
        solution_dim=2,
        measure_dim=2,
        k_neighbors=5,
        novelty_threshold=0.01,
        extra_fields={
            "metadata": ((), object),
            "square": ((2, 2), np.int32)
        },
    )
    result_archive = GridArchive(
        solution_dim=2,
        dims=[100, 100],
        ranges=[(-1, 1), (-1, 1)],
        extra_fields={
            "metadata": ((), object),
            "square": ((2, 2), np.int32)
        },
    )
    emitters = [GaussianEmitter(archive, sigma=1, x0=[0.0, 0.0], batch_size=4)]

    if scheduler_type == "Scheduler":
        scheduler = Scheduler(
            archive,
            emitters,
            result_archive=result_archive,
        )
    else:
        scheduler = BanditScheduler(
            archive,
            emitters,
            1,
            result_archive=result_archive,
        )

    scheduler.ask()

    # The ArrayStore in the archives should throw an error when we try to add.
    scheduler.tell(
        np.zeros(4),
        np.zeros((4, 2)),
        metadata=np.zeros(4, dtype=object),
        square=np.zeros((4, 2, 2)),
    )


def test_tell_inserts_solutions_into_archive(add_mode):
    batch_size = 4
    archive = GridArchive(solution_dim=2,
                          dims=[100, 100],
                          ranges=[(-1, 1), (-1, 1)])
    emitters = [
        GaussianEmitter(archive, sigma=1, x0=[0.0, 0.0], batch_size=batch_size)
    ]
    scheduler = Scheduler(archive, emitters, add_mode=add_mode)

    measures_batch = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]

    _ = scheduler.ask()  # Ignore the actual values of the solutions.
    # We pass in 4 solutions with unique measures, so all should go into
    # the archive.
    scheduler.tell(np.ones(batch_size), measures_batch)

    assert_archive_elites(
        archive=scheduler.archive,
        batch_size=batch_size,
        objective_batch=np.ones(batch_size),
        measures_batch=measures_batch,
    )


def test_tell_inserts_solutions_with_multiple_emitters(add_mode):
    archive = GridArchive(solution_dim=2,
                          dims=[100, 100],
                          ranges=[(-1, 1), (-1, 1)])
    emitters = [
        GaussianEmitter(archive, sigma=1, x0=[0.0, 0.0], batch_size=1),
        GaussianEmitter(archive, sigma=1, x0=[0.5, 0.5], batch_size=2),
        GaussianEmitter(archive, sigma=1, x0=[-0.5, -0.5], batch_size=3),
    ]
    scheduler = Scheduler(archive, emitters, add_mode=add_mode)

    # The sum of all the emitters' batch sizes is 6.
    batch_size = 6
    measures_batch = [
        [1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
        [1.0, -1.0],
        [0.0, 0.0],
        [0.0, 1.0],
    ]

    _ = scheduler.ask()
    scheduler.tell(np.ones(batch_size), measures_batch)

    assert_archive_elites(
        archive=scheduler.archive,
        batch_size=batch_size,
        objective_batch=np.ones(batch_size),
        measures_batch=measures_batch,
    )


def test_tell_with_fields(add_mode):
    batch_size = 4
    archive = GridArchive(
        solution_dim=2,
        dims=[100, 100],
        ranges=[(-1, 1), (-1, 1)],
        extra_fields={"metadata": ((), object)},
    )
    emitters = [
        GaussianEmitter(archive, sigma=1, x0=[0.0, 0.0], batch_size=batch_size)
    ]
    scheduler = Scheduler(archive, emitters, add_mode=add_mode)

    measures_batch = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]

    _ = scheduler.ask()  # Ignore the actual values of the solutions.
    # We pass in 4 solutions with unique measures, so all should go into
    # the archive.
    scheduler.tell(np.ones(batch_size),
                   measures_batch,
                   metadata=["a", "b", "c", "d"])

    assert_archive_elites(
        archive=scheduler.archive,
        batch_size=batch_size,
        objective_batch=np.ones(batch_size),
        measures_batch=measures_batch,
        metadata_batch=["a", "b", "c", "d"],
    )


@pytest.mark.parametrize("scheduler_type", ["Scheduler", "BanditScheduler"])
def test_tell_with_none_objective(scheduler_type, add_mode):
    archive = ProximityArchive(solution_dim=2,
                               measure_dim=2,
                               k_neighbors=1,
                               novelty_threshold=1.0)
    emitters = [GaussianEmitter(archive, sigma=1, x0=[0.0, 0.0], batch_size=4)]
    if scheduler_type == "Scheduler":
        scheduler = Scheduler(archive, emitters, add_mode=add_mode)
    else:
        scheduler = BanditScheduler(archive, emitters, 1, add_mode=add_mode)

    solutions = scheduler.ask()

    scheduler.tell(None, [[0, 0], [0, 1], [1, 1], [1, 0]])

    assert_archive_elites(
        archive,
        4,
        solution_batch=solutions,
        measures_batch=[[0, 0], [0, 1], [1, 1], [1, 0]],
    )


### TESTS FOR OUT-OF-ORDER ASK-TELL ###


def test_tell_fails_when_ask_not_called(scheduler_fixture):
    scheduler, *_ = scheduler_fixture
    with pytest.raises(RuntimeError):
        scheduler.tell(None, None)


def test_tell_fails_when_ask_dqd_not_called(scheduler_fixture):
    scheduler, *_ = scheduler_fixture
    with pytest.raises(RuntimeError):
        scheduler.tell_dqd(None, None, None)


def test_tell_fails_when_ask_tell_mismatch(scheduler_fixture):
    scheduler, *_ = scheduler_fixture

    _ = scheduler.ask()
    with pytest.raises(RuntimeError):
        scheduler.tell_dqd(None, None, None)


def test_tell_fails_when_ask_tell_mismatch_dqd(scheduler_fixture):
    scheduler, *_ = scheduler_fixture

    _ = scheduler.ask_dqd()
    with pytest.raises(RuntimeError):
        scheduler.tell(None, None)


### END ###


def test_emitter_returns_no_solutions(scheduler_fixture):
    scheduler, solution_dim, _ = scheduler_fixture

    # Should not return anything since there are no DQD emitters
    solution_batch = scheduler.ask_dqd()

    assert not np.any(solution_batch)
    assert solution_batch.shape == (0, solution_dim)


@pytest.mark.parametrize("array", ["objective_batch", "measures_batch"])
def test_tell_fails_with_wrong_shapes(scheduler_fixture, array):
    scheduler, _, num_solutions = scheduler_fixture
    _ = scheduler.ask()  # Ignore the actual values of the solutions.

    objective_batch = np.ones(num_solutions)
    measures_batch = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]

    # Each condition makes a certain array have the wrong shape by excluding the
    # last element.
    with pytest.raises(ValueError):
        if array == "objective_batch":
            scheduler.tell(objective_batch[:-1], measures_batch)
        elif array == "measures_batch":
            scheduler.tell(objective_batch, measures_batch[:-1])


def test_constant_active_emitters_bandit_scheduler():
    solution_dim = 2
    num_solutions = 4
    expected_active = 3
    archive = GridArchive(solution_dim=solution_dim,
                          dims=[100, 100],
                          ranges=[(-1, 1), (-1, 1)])
    emitters = [
        GaussianEmitter(archive,
                        sigma=1,
                        x0=[0.0, 0.0],
                        batch_size=num_solutions) for _ in range(10)
    ]
    scheduler = BanditScheduler(archive, emitters, num_active=expected_active)
    num_loops = 10

    rng = np.random.default_rng(42)

    for _ in range(num_loops):
        solutions = scheduler.ask()
        assert scheduler.active.sum() == expected_active

        # Mock objective and measures for tell
        objective = rng.random(len(solutions))
        measures = rng.random((len(solutions), 2))
        scheduler.tell(objective, measures)

        assert scheduler.active.sum() == expected_active
