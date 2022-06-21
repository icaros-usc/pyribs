"""Tests for the rankers."""

import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.emitters.rankers import (TwoStageImprovementRanker,
                                   RandomDirectionRanker,
                                   TwoStageRandomDirectionRanker,
                                   ObjectiveRanker, TwoStageObjectiveRanker)


def test_two_stage_improvement_ranker(archive_fixture):
    archive, x0 = archive_fixture
    emitter = GaussianEmitter(archive, x0, 1, batch_size=4)

    # solutions doesn't actually matter since we are manually assigning
    # objective_values and behavior_values.
    solutions = emitter.ask()
    objective_values = [0, 1, 3, 6]
    behavior_values = [0, 0, 0, 0]
    metadata = []
    statuses = []
    values = []
    for (sol, obj, beh) in zip(solutions, objective_values, behavior_values):
        status, value = archive.add(sol, obj, beh)
        statuses.append(status)
        values.append(value)

    ranker = TwoStageImprovementRanker()
    indicies = ranker.rank(emitter, archive, solutions, objective_values,
                           behavior_values, metadata, statuses, values)

    assert (indicies == [0, 3, 2, 1]).all()


def test_random_direction_ranker():
    x0 = np.array([1, 2, 3, 4])
    archive = GridArchive(len(x0), [10, 10, 10], [(-1, 1), (-1, 1), (-1, 1)])
    emitter = GaussianEmitter(archive, x0, 1, batch_size=4)

    # solutions doesn't actually matter since we are manually assigning
    # objective_values and behavior_values.
    solutions = emitter.ask()
    objective_values = [0, 1, 2, 3]
    behavior_values = [
        [1, 0, 0],
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
    ranker._target_behavior_dir = [0, 1, 0]  # pylint: disable=protected-access

    indicies = ranker.rank(emitter, archive, solutions, objective_values,
                           behavior_values, metadata, statuses, values)

    assert (indicies == [1, 3, 2, 0]).all()


def test_two_stage_random_direction():
    x0 = np.array([1, 2, 3, 4])
    archive = GridArchive(len(x0), [10, 10, 10], [(-1, 1), (-1, 1), (-1, 1)])
    emitter = GaussianEmitter(archive, x0, 1, batch_size=4)

    # solutions doesn't actually matter since we are manually assigning
    # objective_values and behavior_values.
    solutions = emitter.ask()
    objective_values = [0, 1, 2, 3]
    behavior_values = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0.5, 0.5],
        [0, 0.8, 0.8],
    ]
    metadata = []
    statuses = []
    values = []
    for (sol, obj, beh) in zip(solutions, objective_values, behavior_values):
        print(sol, obj, beh)
        status, value = archive.add(sol, obj, beh)
        statuses.append(status)
        values.append(value)

    ranker = TwoStageRandomDirectionRanker()

    # set the random direction
    ranker._target_behavior_dir = [0, 1, 0]  # pylint: disable=protected-access

    indicies = ranker.rank(emitter, archive, solutions, objective_values,
                           behavior_values, metadata, statuses, values)

    assert (indicies == [1, 3, 2, 0]).all()


def test_objective_ranker(archive_fixture):
    archive, x0 = archive_fixture
    emitter = GaussianEmitter(archive, x0, 1, batch_size=2)

    solutions = emitter.ask()
    objective_values = [0, 3, 2, 1]
    behavior_values = [0, 0, 1, 1]
    metadata = []
    statuses = []
    values = []
    for (sol, obj, beh) in zip(solutions, objective_values, behavior_values):
        status, value = archive.add(sol, obj, beh)
        statuses.append(status)
        values.append(value)

    ranker = ObjectiveRanker()

    indicies = ranker.rank(emitter, archive, solutions, objective_values,
                           behavior_values, metadata, statuses, values)

    assert (indicies == [1, 2, 3, 0]).all()


def test_two_stage_objective_ranker(archive_fixture):
    archive, x0 = archive_fixture
    emitter = GaussianEmitter(archive, x0, 1, batch_size=4)

    solutions = emitter.ask()
    objective_values = [0, 3, 1, 2]
    behavior_values = [0, 0, 1, 1]
    metadata = []
    statuses = []
    values = []
    for (sol, obj, beh) in zip(solutions, objective_values, behavior_values):
        status, value = archive.add(sol, obj, beh)
        statuses.append(status)
        values.append(value)

    ranker = TwoStageObjectiveRanker()

    indicies = ranker.rank(emitter, archive, solutions, objective_values,
                           behavior_values, metadata, statuses, values)

    assert (indicies == [2, 0, 1, 3]).all()
