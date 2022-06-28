"""Tests for the rankers."""

import numpy as np

from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.emitters.rankers import (ObjectiveRanker, RandomDirectionRanker,
                                   TwoStageImprovementRanker,
                                   TwoStageObjectiveRanker,
                                   TwoStageRandomDirectionRanker)


def test_two_stage_improvement_ranker(archive_fixture):
    archive, x0 = archive_fixture
    emitter = GaussianEmitter(archive, x0, 1, batch_size=4)

    # solutions doesn't actually matter since we are manually assigning
    # objective_values and behavior_values.
    solutions = emitter.ask()
    objective_values = [0, 1, 3, 6]
    behavior_values = [[0, 0], [0, 0], [0, 0], [0, 0]]
    metadata = []
    statuses = []
    values = []
    for (sol, obj, beh) in zip(solutions, objective_values, behavior_values):
        status, value = archive.add(sol, obj, beh)
        statuses.append(status)
        values.append(value)

    ranker = TwoStageImprovementRanker()
    indices, ranking_values = ranker.rank(emitter, archive, None, solutions,
                                          objective_values, behavior_values,
                                          metadata, statuses, values)

    assert (indices == [0, 3, 2, 1]).all()
    assert (ranking_values == [
        [values[0], statuses[0]],
        [values[1], statuses[1]],
        [values[2], statuses[2]],
        [values[3], statuses[3]],
    ]).all()


def test_random_direction_ranker():
    x0 = np.array([1, 2, 3, 4])
    archive = GridArchive(len(x0), [10, 10, 10], [(-1, 1), (-1, 1), (-1, 1)])
    emitter = GaussianEmitter(archive, x0, 1, batch_size=4)

    # solutions doesn't actually matter since we are manually assigning
    # objective_values and behavior_values.
    solutions = emitter.ask()
    objective_values = [0, 1, 2, 3]
    behavior_values = [
        [0, 0.9, 0],
        [0, 1, 0],
        [0, 0.5, 0.5],
        [0, 0.8, 0.8],
    ]
    metadata = []
    statuses = []
    values = []
    for (sol, obj, beh) in zip(solutions, objective_values, behavior_values):
        status, value = archive.add(sol, obj, beh)
        statuses.append(status)
        values.append(value)

    ranker = RandomDirectionRanker()

    # set the random direction
    ranker.target_measure_dir = [0, 1, 0]

    indices, ranking_values = ranker.rank(emitter, archive, None, solutions,
                                          objective_values, behavior_values,
                                          metadata, statuses, values)

    assert (indices == [1, 0, 3, 2]).all()
    assert (ranking_values == np.dot(behavior_values, [0, 1, 0])).all()


def test_two_stage_random_direction():
    x0 = np.array([1, 2, 3, 4])
    archive = GridArchive(len(x0), [10, 10, 10], [(-1, 1), (-1, 1), (-1, 1)])
    emitter = GaussianEmitter(archive, x0, 1, batch_size=4)

    # solutions doesn't actually matter since we are manually assigning
    # objective_values and behavior_values.
    solutions = emitter.ask()
    objective_values = [0, 1, 2, 3]
    behavior_values = [
        [0, 0.9, 0],
        [0, 1, 0],
        [0, 0.5, 0.5],
        [0, 0.8, 0.8],
    ]
    metadata = []
    statuses = []
    values = []
    for (sol, obj, beh) in zip(solutions, objective_values, behavior_values):
        status, value = archive.add(sol, obj, beh)
        statuses.append(status)
        values.append(value)

    ranker = TwoStageRandomDirectionRanker()

    # set the random direction
    ranker.target_measure_dir = [0, 1, 0]

    indices, ranking_values = ranker.rank(emitter, archive, None, solutions,
                                          objective_values, behavior_values,
                                          metadata, statuses, values)

    assert (indices == [0, 3, 2, 1]).all()

    projections = np.dot(behavior_values, [0, 1, 0])
    assert (ranking_values == [
        [projections[0], statuses[0]],
        [projections[1], statuses[1]],
        [projections[2], statuses[2]],
        [projections[3], statuses[3]],
    ]).all()


def test_objective_ranker(archive_fixture):
    archive, x0 = archive_fixture
    emitter = GaussianEmitter(archive, x0, 1, batch_size=2)

    solutions = emitter.ask()
    objective_values = [0, 3, 2, 1]
    behavior_values = [[0, 0], [0, 0], [1, 1], [1, 1]]
    metadata = []
    statuses = []
    values = []
    for (sol, obj, beh) in zip(solutions, objective_values, behavior_values):
        status, value = archive.add(sol, obj, beh)
        statuses.append(status)
        values.append(value)

    ranker = ObjectiveRanker()

    indices, ranking_values = ranker.rank(emitter, archive, None, solutions,
                                          objective_values, behavior_values,
                                          metadata, statuses, values)

    assert (indices == [1, 2, 3, 0]).all()
    assert ranking_values == objective_values


def test_two_stage_objective_ranker(archive_fixture):
    archive, x0 = archive_fixture
    emitter = GaussianEmitter(archive, x0, 1, batch_size=4)

    solutions = emitter.ask()
    objective_values = [0, 3, 1, 2]
    behavior_values = [[0, 0], [0, 0], [1, 1], [1, 1]]
    metadata = []
    statuses = []
    values = []
    for (sol, obj, beh) in zip(solutions, objective_values, behavior_values):
        status, value = archive.add(sol, obj, beh)
        statuses.append(status)
        values.append(value)

    ranker = TwoStageObjectiveRanker()

    indices, ranking_values = ranker.rank(emitter, archive, None, solutions,
                                          objective_values, behavior_values,
                                          metadata, statuses, values)

    assert (indices == [2, 0, 1, 3]).all()
    assert (ranking_values == [
        [objective_values[0], statuses[0]],
        [objective_values[1], statuses[1]],
        [objective_values[2], statuses[2]],
        [objective_values[3], statuses[3]],
    ]).all()
