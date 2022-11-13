"""Tests that should work for all emitters."""
import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.emitters import (EvolutionStrategyEmitter, GaussianEmitter,
                           GradientArborescenceEmitter, IsoLineEmitter)

# pylint: disable = redefined-outer-name


@pytest.fixture(params=[
    "GaussianEmitter", "IsoLineEmitter", "ImprovementEmitter",
    "RandomDirectionEmitter", "OptimizingEmitter"
])
def emitter_fixture(request, archive_fixture):
    """Creates an archive, emitter, and initial solution.

    Returns:
        Tuple of (archive, emitter, batch_size, x0).
    """
    emitter_type = request.param
    archive, x0 = archive_fixture
    batch_size = 3

    if emitter_type == "GaussianEmitter":
        emitter = GaussianEmitter(archive,
                                  sigma=5,
                                  x0=x0,
                                  batch_size=batch_size)
    elif emitter_type == "IsoLineEmitter":
        emitter = IsoLineEmitter(archive, x0=x0, batch_size=batch_size)
    elif emitter_type == "ImprovementEmitter":
        emitter = EvolutionStrategyEmitter(archive,
                                           x0=x0,
                                           sigma0=5,
                                           ranker="2imp",
                                           batch_size=batch_size)
    elif emitter_type == "RandomDirectionEmitter":
        emitter = EvolutionStrategyEmitter(archive,
                                           x0=x0,
                                           sigma0=5,
                                           ranker="2rd",
                                           batch_size=batch_size)
    elif emitter_type == "OptimizingEmitter":
        emitter = EvolutionStrategyEmitter(archive,
                                           x0=x0,
                                           sigma0=5,
                                           ranker="2obj",
                                           batch_size=batch_size)
    else:
        raise NotImplementedError(f"Unknown emitter type {emitter_type}")

    return archive, emitter, batch_size, x0


#
# ask()
#


def test_ask_emits_correct_num_sols(emitter_fixture):
    _, emitter, batch_size, x0 = emitter_fixture
    solutions = emitter.ask()
    assert solutions.shape == (batch_size, len(x0))


def test_ask_emits_correct_num_sols_on_nonempty_archive(emitter_fixture):
    archive, emitter, batch_size, x0 = emitter_fixture
    archive.add_single(x0, 1, np.array([0, 0]))
    solutions = emitter.ask()
    assert solutions.shape == (batch_size, len(x0))


#
# tell()
#


@pytest.fixture(
    params=["GradientArborescenceEmitter", "EvolutionStrategyEmitter"])
def shape_test_fixture(request):
    """Creates an archive, emitter, and initial solution.

    Returns:
        Tuple of (archive, emitter, batch_size, solution_dim).
    """
    emitter_name = request.param

    batch_size = 3
    archive = GridArchive(solution_dim=1, dims=[10], ranges=[(-1.0, 1.0)])
    if emitter_name == "GradientArborescenceEmitter":
        emitter = GradientArborescenceEmitter(archive,
                                             x0=np.array([0]),
                                             sigma0=1.0,
                                             step_size=0.1,
                                             batch_size=batch_size)
    elif emitter_name == "EvolutionStrategyEmitter":
        emitter = EvolutionStrategyEmitter(archive,
                                           x0=np.array([0]),
                                           sigma0=1.0,
                                           ranker="imp",
                                           batch_size=batch_size)

    return archive, emitter, batch_size


@pytest.mark.parametrize("offset", [(0, 1), (1, 0), (1, 1)])
def test_tell_wrong_shape_solution_batch(shape_test_fixture, offset):
    archive, emitter, batch_size = shape_test_fixture

    wrong_batch_size = batch_size + offset[0]
    wrong_solution_dim = archive.solution_dim + offset[1]

    solution_batch = np.ones(
        (wrong_batch_size, wrong_solution_dim))  # wrong shape
    objective_batch = np.ones(batch_size)
    measures_batch = np.ones((batch_size, archive.measure_dim))
    jacobian_batch = np.ones(
        (batch_size, archive.measure_dim + 1, archive.solution_dim))
    status_batch = np.ones(batch_size)
    value_batch = np.ones(batch_size)

    if isinstance(emitter, GradientArborescenceEmitter):
        with pytest.raises(ValueError):
            emitter.tell_dqd(solution_batch, objective_batch, measures_batch,
                             jacobian_batch, status_batch, value_batch)

    with pytest.raises(ValueError):
        # For GradientArborescenceEmitter, tell is called before tell_dqd, but
        # a shape check exception should be thrown before tell complains that
        # tell_dqd is called.
        emitter.tell(solution_batch, objective_batch, measures_batch,
                     status_batch, value_batch)


def test_tell_wrong_shape_objective_batch(shape_test_fixture):
    archive, emitter, batch_size = shape_test_fixture

    solution_batch = np.ones((batch_size, archive.solution_dim))
    objective_batch = np.ones(batch_size + 1)  # wrong shape
    measures_batch = np.ones((batch_size, archive.measure_dim))
    jacobian_batch = np.ones(
        (batch_size, archive.measure_dim + 1, archive.solution_dim))
    status_batch = np.ones(batch_size)
    value_batch = np.ones(batch_size)

    if isinstance(emitter, GradientArborescenceEmitter):
        with pytest.raises(ValueError):
            emitter.tell_dqd(solution_batch, objective_batch, measures_batch,
                             jacobian_batch, status_batch, value_batch)

    with pytest.raises(ValueError):
        # For GradientArborescenceEmitter, tell is called before tell_dqd, but
        # a shape check exception should be thrown before tell complains that
        # tell_dqd is called.
        emitter.tell(solution_batch, objective_batch, measures_batch,
                     status_batch, value_batch)


@pytest.mark.parametrize("offset", [(0, 1), (1, 0), (1, 1)])
def test_tell_wrong_shape_measures_batch(shape_test_fixture, offset):
    archive, emitter, batch_size = shape_test_fixture

    wrong_batch_size = batch_size + offset[0]
    wrong_measure_dim = archive.measure_dim + offset[1]

    solution_batch = np.ones((batch_size, archive.solution_dim))
    objective_batch = np.ones(batch_size)
    measures_batch = np.ones(
        (wrong_batch_size, wrong_measure_dim))  # wrong shape
    jacobian_batch = np.ones(
        (batch_size, archive.measure_dim + 1, archive.solution_dim))
    status_batch = np.ones(batch_size)
    value_batch = np.ones(batch_size)

    if isinstance(emitter, GradientArborescenceEmitter):
        with pytest.raises(ValueError):
            emitter.tell_dqd(solution_batch, objective_batch, measures_batch,
                             jacobian_batch, status_batch, value_batch)

    with pytest.raises(ValueError):
        # For GradientArborescenceEmitter, tell is called before tell_dqd, but
        # a shape check exception should be thrown before tell complains that
        # tell_dqd is called.
        emitter.tell(solution_batch, objective_batch, measures_batch,
                     status_batch, value_batch)


@pytest.mark.parametrize("offset", [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1),
                                    (1, 1, 0), (1, 0, 1), (1, 1, 1)])
def test_tell_wrong_shape_jacobian_batch(shape_test_fixture, offset):
    archive, emitter, batch_size = shape_test_fixture

    wrong_batch_size = batch_size + offset[0]
    wrong_measure_dim = archive.measure_dim + offset[1]
    wrong_solution_dim = archive.solution_dim + offset[2]

    solution_batch = np.ones((batch_size, archive.solution_dim))
    objective_batch = np.ones(batch_size)
    measures_batch = np.ones((batch_size, archive.measure_dim))
    jacobian_batch = np.ones((wrong_batch_size, wrong_measure_dim + 1,
                              wrong_solution_dim))  # wrong shape
    status_batch = np.ones(batch_size)
    value_batch = np.ones(batch_size)

    if isinstance(emitter, GradientArborescenceEmitter):
        with pytest.raises(ValueError):
            emitter.tell_dqd(solution_batch, objective_batch, measures_batch,
                             jacobian_batch, status_batch, value_batch)


def test_tell_wrong_shape_status_batch(shape_test_fixture):
    archive, emitter, batch_size = shape_test_fixture

    wrong_batch_size = batch_size + 1

    solution_batch = np.ones((batch_size, archive.solution_dim))
    objective_batch = np.ones(batch_size)
    measures_batch = np.ones((batch_size, archive.measure_dim))
    jacobian_batch = np.ones(
        (batch_size, archive.measure_dim + 1, archive.solution_dim))
    status_batch = np.ones(wrong_batch_size)  # wrong shape
    value_batch = np.ones(batch_size)

    if isinstance(emitter, GradientArborescenceEmitter):
        with pytest.raises(ValueError):
            emitter.tell_dqd(solution_batch, objective_batch, measures_batch,
                             jacobian_batch, status_batch, value_batch)

    with pytest.raises(ValueError):
        # For GradientArborescenceEmitter, tell is called before tell_dqd, but
        # a shape check exception should be thrown before tell complains that
        # tell_dqd is called.
        emitter.tell(solution_batch, objective_batch, measures_batch,
                     status_batch, value_batch)


def test_tell_wrong_shape_value_batch(shape_test_fixture):
    archive, emitter, batch_size = shape_test_fixture

    wrong_batch_size = batch_size + 1

    solution_batch = np.ones((batch_size, archive.solution_dim))
    objective_batch = np.ones(batch_size)
    measures_batch = np.ones((batch_size, archive.measure_dim))
    jacobian_batch = np.ones(
        (batch_size, archive.measure_dim + 1, archive.solution_dim))
    status_batch = np.ones(batch_size)
    value_batch = np.ones(wrong_batch_size)  # wrong shape

    if isinstance(emitter, GradientArborescenceEmitter):
        with pytest.raises(ValueError):
            emitter.tell_dqd(solution_batch, objective_batch, measures_batch,
                             jacobian_batch, status_batch, value_batch)

    with pytest.raises(ValueError):
        # For GradientArborescenceEmitter, tell is called before tell_dqd, but
        # a shape check exception should be thrown before tell complains that
        # tell_dqd is called.
        emitter.tell(solution_batch, objective_batch, measures_batch,
                     status_batch, value_batch)


def test_tell_wrong_shape_batch_size(shape_test_fixture):
    archive, emitter, batch_size = shape_test_fixture

    wrong_batch_size = batch_size + 1

    # Everything has wrong shape.
    solution_batch = np.ones((wrong_batch_size, archive.solution_dim))
    objective_batch = np.ones(wrong_batch_size)
    measures_batch = np.ones((wrong_batch_size, archive.measure_dim))
    jacobian_batch = np.ones(
        (wrong_batch_size, archive.solution_dim + 1, archive.solution_dim))
    status_batch = np.ones(wrong_batch_size)
    value_batch = np.ones(wrong_batch_size)

    if isinstance(emitter, GradientArborescenceEmitter):
        with pytest.raises(ValueError):
            emitter.tell_dqd(solution_batch, objective_batch, measures_batch,
                             jacobian_batch, status_batch, value_batch)

    with pytest.raises(ValueError):
        # For GradientArborescenceEmitter, tell is called before tell_dqd, but
        # a shape check exception should be thrown before tell complains that
        # tell_dqd is called.
        emitter.tell(solution_batch, objective_batch, measures_batch,
                     status_batch, value_batch)


#
# Bounds handling (only uses GaussianEmitter).
#


def test_array_bound_correct(archive_fixture):
    archive, x0 = archive_fixture
    bounds = []
    for i in range(len(x0) - 1):
        bounds.append((-i, i))
    bounds.append(None)
    emitter = GaussianEmitter(archive, sigma=1, x0=x0, bounds=bounds)

    lower_bounds = np.concatenate((-np.arange(len(x0) - 1), [-np.inf]))
    upper_bounds = np.concatenate((np.arange(len(x0) - 1), [np.inf]))

    assert (emitter.lower_bounds == lower_bounds).all()
    assert (emitter.upper_bounds == upper_bounds).all()


def test_long_array_bound_fails(archive_fixture):
    archive, x0 = archive_fixture
    bounds = [(-1, 1)] * (len(x0) + 1)  # More bounds than solution dims.
    with pytest.raises(ValueError):
        GaussianEmitter(archive, sigma=1, x0=x0, bounds=bounds)


def test_array_bound_bad_entry_fails(archive_fixture):
    archive, x0 = archive_fixture
    bounds = [(-1, 1)] * len(x0)
    bounds[0] = (-1, 0, 1)  # Invalid entry.
    with pytest.raises(ValueError):
        GaussianEmitter(archive, sigma=1, x0=x0, bounds=bounds)


@pytest.mark.parametrize(
    "emitter_type", ["GaussianEmitter", "IsoLineEmitter", "ImprovementEmitter"])
def test_emitters_fail_when_x0_not_1d(emitter_type, archive_fixture):
    archive, _ = archive_fixture
    x0 = [[1], [1]]

    with pytest.raises(ValueError):
        if emitter_type == "GaussianEmitter":
            _ = GaussianEmitter(archive, sigma=5, x0=x0)
        elif emitter_type == "IsoLineEmitter":
            _ = IsoLineEmitter(archive, x0=x0)
        elif emitter_type == "ImprovementEmitter":
            _ = EvolutionStrategyEmitter(archive,
                                         x0=x0,
                                         sigma0=5,
                                         ranker="2imp")
