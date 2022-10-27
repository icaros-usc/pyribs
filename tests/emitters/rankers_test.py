"""Tests for the rankers."""
import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.emitters.rankers import (ObjectiveRanker, RandomDirectionRanker,
                                   TwoStageImprovementRanker,
                                   TwoStageObjectiveRanker,
                                   TwoStageRandomDirectionRanker)


@pytest.fixture
def emitter():
    """The rankers currently do not need the actual emitter, so we just return
    None."""
    return None


@pytest.fixture
def rng():
    """An rng for the rankers."""
    return np.random.default_rng(seed=0)


# emitter and rng would be marked by pytest as redefined.
# pylint: disable = redefined-outer-name


def test_two_stage_improvement_ranker(archive_fixture, emitter, rng):
    archive, x0 = archive_fixture
    solution_batch = [x0, x0, x0, x0]
    objective_batch = [0, 1, 3, 6]
    measures_batch = [[0, 0], [0, 0], [0, 0], [0, 0]]
    metadata_batch = [None, None, None, None]

    # Artificially force the first solution to have a new status and the other
    # solutions to have improve status.
    first_status, first_value = archive.add_single(solution_batch[0],
                                                   objective_batch[0],
                                                   measures_batch[0],
                                                   metadata_batch[0])
    status_batch, value_batch = archive.add(solution_batch[1:],
                                            objective_batch[1:],
                                            measures_batch[1:],
                                            metadata_batch[1:])
    status_batch = np.concatenate(([first_status], status_batch))
    value_batch = np.concatenate(([first_value], value_batch))

    ranker = TwoStageImprovementRanker()
    indices, ranking_values = ranker.rank(emitter, archive, rng, solution_batch,
                                          objective_batch, measures_batch,
                                          status_batch, value_batch,
                                          metadata_batch)

    assert (indices == [0, 3, 2, 1]).all()
    assert (ranking_values == [
        [status_batch[0], value_batch[0]],
        [status_batch[1], value_batch[1]],
        [status_batch[2], value_batch[2]],
        [status_batch[3], value_batch[3]],
    ]).all()


def test_random_direction_ranker(emitter, rng):
    x0 = np.array([1, 2, 3, 4])
    archive = GridArchive(solution_dim=len(x0),
                          dims=[10, 10, 10],
                          ranges=[(-1, 1), (-1, 1), (-1, 1)])
    solution_batch = [x0, x0, x0, x0]
    objective_batch = [0, 1, 2, 3]
    measures_batch = [
        [0, 0.9, 0],
        [0, 1, 0],
        [0, 0.5, 0.5],
        [0, 0.8, 0.8],
    ]
    metadata_batch = [None, None, None, None]
    status_batch, value_batch = archive.add(solution_batch, objective_batch,
                                            measures_batch, metadata_batch)

    ranker = RandomDirectionRanker()
    ranker.target_measure_dir = [0, 1, 0]  # Set the random direction.
    indices, ranking_values = ranker.rank(emitter, archive, rng, solution_batch,
                                          objective_batch, measures_batch,
                                          status_batch, value_batch,
                                          metadata_batch)

    assert (indices == [1, 0, 3, 2]).all()
    assert (ranking_values == np.dot(measures_batch, [0, 1, 0])).all()


def test_two_stage_random_direction(emitter, rng):
    x0 = np.array([1, 2, 3, 4])
    archive = GridArchive(solution_dim=len(x0),
                          dims=[10, 10, 10],
                          ranges=[(-1, 1), (-1, 1), (-1, 1)])
    solution_batch = [x0, x0, x0, x0]
    objective_batch = [0, 1, 2, 3]
    measures_batch = [
        [0, 0.9, 0],
        [0, 1, 0],
        [0, 0.5, 0.5],
        [0, 0.8, 0.8],
    ]
    metadata_batch = [None, None, None, None]

    # Artificially force the first solution to have a new status and the other
    # solutions to have improve status.
    first_status, first_value = archive.add_single(solution_batch[0],
                                                   objective_batch[0],
                                                   measures_batch[0],
                                                   metadata_batch[0])
    status_batch, value_batch = archive.add(solution_batch[1:],
                                            objective_batch[1:],
                                            measures_batch[1:],
                                            metadata_batch[1:])
    status_batch = np.concatenate(([first_status], status_batch))
    value_batch = np.concatenate(([first_value], value_batch))

    ranker = TwoStageRandomDirectionRanker()
    ranker.target_measure_dir = [0, 1, 0]  # Set the random direction.
    indices, ranking_values = ranker.rank(emitter, archive, rng, solution_batch,
                                          objective_batch, measures_batch,
                                          status_batch, value_batch,
                                          metadata_batch)

    assert (indices == [0, 3, 2, 1]).all()
    projections = np.dot(measures_batch, [0, 1, 0])
    assert (ranking_values == [
        [status_batch[0], projections[0]],
        [status_batch[1], projections[1]],
        [status_batch[2], projections[2]],
        [status_batch[3], projections[3]],
    ]).all()


def test_objective_ranker(archive_fixture, emitter, rng):
    archive, x0 = archive_fixture
    solution_batch = [x0, x0, x0, x0]
    objective_batch = [0, 3, 2, 1]
    measures_batch = [[0, 0], [0, 0], [1, 1], [1, 1]]
    metadata_batch = [None, None, None, None]
    status_batch, value_batch = archive.add(solution_batch, objective_batch,
                                            measures_batch, metadata_batch)

    ranker = ObjectiveRanker()
    indices, ranking_values = ranker.rank(emitter, archive, rng, solution_batch,
                                          objective_batch, measures_batch,
                                          status_batch, value_batch,
                                          metadata_batch)

    assert (indices == [1, 2, 3, 0]).all()
    assert ranking_values == objective_batch


def test_two_stage_objective_ranker(archive_fixture, emitter, rng):
    archive, x0 = archive_fixture
    solution_batch = [x0, x0, x0, x0]
    objective_batch = [0, 1, 3, 2]
    measures_batch = [[0, 0], [1, 1], [0, 0], [1, 1]]
    metadata_batch = [None, None, None, None]

    # Artificially force the first two solutions to have a new status and the
    # other two solutions to have improve status.
    status_batch_1, value_batch_1 = archive.add(solution_batch[:2],
                                                objective_batch[:2],
                                                measures_batch[:2],
                                                metadata_batch[:2])
    status_batch_2, value_batch_2 = archive.add(solution_batch[2:],
                                                objective_batch[2:],
                                                measures_batch[2:],
                                                metadata_batch[2:])
    status_batch = np.concatenate((status_batch_1, status_batch_2))
    value_batch = np.concatenate((value_batch_1, value_batch_2))

    ranker = TwoStageObjectiveRanker()
    indices, ranking_values = ranker.rank(emitter, archive, rng, solution_batch,
                                          objective_batch, measures_batch,
                                          status_batch, value_batch,
                                          metadata_batch)

    assert (indices == [1, 0, 2, 3]).all()
    assert (ranking_values == [
        [status_batch[0], objective_batch[0]],
        [status_batch[1], objective_batch[1]],
        [status_batch[2], objective_batch[2]],
        [status_batch[3], objective_batch[3]],
    ]).all()
