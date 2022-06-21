"""Tests for the rankers."""

import numpy as np
import pytest

from ribs.emitters import GaussianEmitter
from ribs.emitters.rankers import TwoStageImprovementRanker


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

    assert indicies == [0, 3, 2, 1]
