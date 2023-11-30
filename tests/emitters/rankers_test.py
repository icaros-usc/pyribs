"""Tests for the rankers."""
import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.emitters.rankers import (ObjectiveRanker, RandomDirectionRanker,
                                   TwoStageImprovementRanker,
                                   TwoStageObjectiveRanker,
                                   TwoStageRandomDirectionRanker)

# pylint: disable = redefined-outer-name


@pytest.fixture
def emitter():
    """The rankers currently do not need the actual emitter, so we just return
    None."""
    return None


def test_two_stage_improvement_ranker(archive_fixture, emitter):
    archive, x0 = archive_fixture
    solution_batch = [x0, x0, x0, x0]
    objective_batch = [0, 1, 3, 6]
    measures_batch = [[0, 0], [0, 0], [0, 0], [0, 0]]

    # Artificially force the first solution to have a new status and the other
    # solutions to have improve status.
    first_info = archive.add_single(solution_batch[0], objective_batch[0],
                                    measures_batch[0])
    second_info = archive.add(solution_batch[1:], objective_batch[1:],
                              measures_batch[1:])
    add_info = {
        key: np.concatenate(([first_info[key]], second_info[key]))
        for key in first_info
    }

    ranker = TwoStageImprovementRanker()
    indices, ranking_values = ranker.rank(
        emitter,
        archive,
        {
            "solution": solution_batch,
            "objective": objective_batch,
            "measures": measures_batch,
        },
        add_info,
    )

    assert (indices == [0, 3, 2, 1]).all()
    assert (ranking_values == [
        [add_info["status"][0], add_info["value"][0]],
        [add_info["status"][1], add_info["value"][1]],
        [add_info["status"][2], add_info["value"][2]],
        [add_info["status"][3], add_info["value"][3]],
    ]).all()


def test_random_direction_ranker(emitter):
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
    add_info = archive.add(solution_batch, objective_batch, measures_batch)

    ranker = RandomDirectionRanker()
    ranker.target_measure_dir = [0, 1, 0]  # Set the random direction.
    indices, ranking_values = ranker.rank(
        emitter,
        archive,
        {
            "solution": solution_batch,
            "objective": objective_batch,
            "measures": measures_batch,
        },
        add_info,
    )

    assert (indices == [1, 0, 3, 2]).all()
    assert (ranking_values == np.dot(measures_batch, [0, 1, 0])).all()


def test_two_stage_random_direction(emitter):
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

    # Artificially force the first solution to have a new status and the other
    # solutions to have improve status.
    first_info = archive.add_single(solution_batch[0], objective_batch[0],
                                    measures_batch[0])
    second_info = archive.add(solution_batch[1:], objective_batch[1:],
                              measures_batch[1:])
    add_info = {
        key: np.concatenate(([first_info[key]], second_info[key]))
        for key in first_info
    }

    ranker = TwoStageRandomDirectionRanker()
    ranker.target_measure_dir = [0, 1, 0]  # Set the random direction.
    indices, ranking_values = ranker.rank(
        emitter,
        archive,
        {
            "solution": solution_batch,
            "objective": objective_batch,
            "measures": measures_batch,
        },
        add_info,
    )

    assert (indices == [0, 3, 2, 1]).all()
    projections = np.dot(measures_batch, [0, 1, 0])
    assert (ranking_values == [
        [add_info["status"][0], projections[0]],
        [add_info["status"][1], projections[1]],
        [add_info["status"][2], projections[2]],
        [add_info["status"][3], projections[3]],
    ]).all()


def test_objective_ranker(archive_fixture, emitter):
    archive, x0 = archive_fixture
    solution_batch = [x0, x0, x0, x0]
    objective_batch = [0, 3, 2, 1]
    measures_batch = [[0, 0], [0, 0], [1, 1], [1, 1]]
    add_info = archive.add(solution_batch, objective_batch, measures_batch)

    ranker = ObjectiveRanker()
    indices, ranking_values = ranker.rank(
        emitter,
        archive,
        {
            "solution": solution_batch,
            "objective": objective_batch,
            "measures": measures_batch,
        },
        add_info,
    )

    assert (indices == [1, 2, 3, 0]).all()
    assert ranking_values == objective_batch


def test_two_stage_objective_ranker(archive_fixture, emitter):
    archive, x0 = archive_fixture
    solution_batch = [x0, x0, x0, x0]
    objective_batch = [0, 1, 3, 2]
    measures_batch = [[0, 0], [1, 1], [0, 0], [1, 1]]

    # Artificially force the first two solutions to have a new status and the
    # other two solutions to have improve status.
    first_info = archive.add(solution_batch[:2], objective_batch[:2],
                             measures_batch[:2])
    second_info = archive.add(solution_batch[2:], objective_batch[2:],
                              measures_batch[2:])
    add_info = {
        key: np.concatenate((first_info[key], second_info[key]))
        for key in first_info
    }

    ranker = TwoStageObjectiveRanker()
    indices, ranking_values = ranker.rank(
        emitter,
        archive,
        {
            "solution": solution_batch,
            "objective": objective_batch,
            "measures": measures_batch,
        },
        add_info,
    )

    assert (indices == [1, 0, 2, 3]).all()
    assert (ranking_values == [
        [add_info["status"][0], objective_batch[0]],
        [add_info["status"][1], objective_batch[1]],
        [add_info["status"][2], objective_batch[2]],
        [add_info["status"][3], objective_batch[3]],
    ]).all()
