"""Tests for the rankers."""

import numpy as np
import pytest

from ribs.emitters import GaussianEmitter
from ribs.emitters.rankers import (TwoStageImprovementRanker,
                                   RandomDirectionRanker)


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


def test_random_direction_ranker(archive_fixture):
    archive, x0 = archive_fixture
    emitter = GaussianEmitter(archive, x0, 1, batch_size=2)

    # solutions doesn't actually matter since we are manually assigning
    # objective_values and behavior_values.
    solutions = emitter.ask()
    objective_values = [0, 1]
    behavior_values = [
        [1, 0],
        [0, 1],
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
    ranker._target_behavior_dir = [0, 1]

    indicies = ranker.rank(emitter, archive, solutions, objective_values,
                           behavior_values, metadata, statuses, values)

    assert (indicies == [1, 0]).all()
