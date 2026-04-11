"""Tests for the rankers."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ribs.archives import DensityArchive, GridArchive, ProximityArchive
from ribs.emitters.rankers import (
    DensityRanker,
    NoveltyRanker,
    NSLCRanker,
    ObjectiveRanker,
    RandomDirectionRanker,
    TwoStageImprovementRanker,
    TwoStageObjectiveRanker,
    TwoStageRandomDirectionRanker,
    _get_ranker,
)

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
    first_info = archive.add_single(
        solution_batch[0], objective_batch[0], measures_batch[0]
    )
    second_info = archive.add(
        solution_batch[1:], objective_batch[1:], measures_batch[1:]
    )
    add_info = {
        key: np.concatenate(([first_info[key]], second_info[key])) for key in first_info
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
    assert (
        ranking_values
        == [
            [add_info["status"][0], add_info["value"][0]],
            [add_info["status"][1], add_info["value"][1]],
            [add_info["status"][2], add_info["value"][2]],
            [add_info["status"][3], add_info["value"][3]],
        ]
    ).all()


def test_random_direction_ranker(emitter):
    x0 = np.array([1, 2, 3, 4])
    archive = GridArchive(
        solution_dim=len(x0), dims=[10, 10, 10], ranges=[(-1, 1), (-1, 1), (-1, 1)]
    )
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
    ranker.target_measure_dir = np.asarray([0, 1, 0], dtype=np.float64)
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
    archive = GridArchive(
        solution_dim=len(x0), dims=[10, 10, 10], ranges=[(-1, 1), (-1, 1), (-1, 1)]
    )
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
    first_info = archive.add_single(
        solution_batch[0], objective_batch[0], measures_batch[0]
    )
    second_info = archive.add(
        solution_batch[1:], objective_batch[1:], measures_batch[1:]
    )
    add_info = {
        key: np.concatenate(([first_info[key]], second_info[key]))
        for key in first_info  # pylint: disable = consider-using-dict-items
    }

    ranker = TwoStageRandomDirectionRanker()
    ranker.target_measure_dir = np.asarray([0, 1, 0], dtype=np.float64)
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
    assert (
        ranking_values
        == [
            [add_info["status"][0], projections[0]],
            [add_info["status"][1], projections[1]],
            [add_info["status"][2], projections[2]],
            [add_info["status"][3], projections[3]],
        ]
    ).all()


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
    first_info = archive.add(
        solution_batch[:2], objective_batch[:2], measures_batch[:2]
    )
    second_info = archive.add(
        solution_batch[2:], objective_batch[2:], measures_batch[2:]
    )
    add_info = {
        key: np.concatenate((first_info[key], second_info[key])) for key in first_info
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
    assert (
        ranking_values
        == [
            [add_info["status"][0], objective_batch[0]],
            [add_info["status"][1], objective_batch[1]],
            [add_info["status"][2], objective_batch[2]],
            [add_info["status"][3], objective_batch[3]],
        ]
    ).all()


def test_novelty_ranker(emitter):
    ranker = NoveltyRanker()
    archive = ProximityArchive(
        solution_dim=3, measure_dim=2, k_neighbors=5, novelty_threshold=0.01
    )

    indices, ranking_values = ranker.rank(
        emitter=emitter,
        archive=archive,
        data={
            "solution": [[1, 2, 3]] * 4,
            "measures": [[0, 0], [1.2, 1.2], [0.1, 0.1], [1.5, 1.5]],
        },
        add_info={
            "novelty": [1.0, 0.5, 0.9, 0.4],
        },
    )

    assert (indices == [0, 2, 1, 3]).all()
    assert_allclose(ranking_values, [1.0, 0.5, 0.9, 0.4])


def test_density_ranker(emitter):
    ranker = DensityRanker()
    archive = DensityArchive(measure_dim=2)

    indices, ranking_values = ranker.rank(
        emitter=emitter,
        archive=archive,
        data={
            "solution": [[1, 2, 3]] * 4,
            "measures": [[0, 0], [1.2, 1.2], [0.1, 0.1], [1.5, 1.5]],
        },
        add_info={
            "density": [0.5, 0.3, 0.7, 0.1],
        },
    )
    assert (indices == [3, 1, 0, 2]).all()
    assert_allclose(ranking_values, [0.5, 0.3, 0.7, 0.1])


def _nslc_archive():
    return ProximityArchive(
        solution_dim=3,
        measure_dim=2,
        k_neighbors=3,
        novelty_threshold=0.01,
        local_competition=True,
    )


def test_nslc_ranker_single_front(emitter):
    # All four solutions are mutually non-dominated: each is strictly better than
    # every other on exactly one of the two objectives (novelty or local
    # competition). They share one front, and crowding distance decides the order.
    # Solutions 0 and 3 are the boundary points (extreme on at least one
    # objective) and should rank ahead of the interior points 1 and 2.
    ranker = NSLCRanker()
    archive = _nslc_archive()

    novelty = np.array([1.0, 0.6, 0.4, 0.2])
    local_competition = np.array([0.0, 0.3, 0.6, 1.0])

    indices, ranking_values = ranker.rank(
        emitter=emitter,
        archive=archive,
        data={
            "solution": [[1, 2, 3]] * 4,
            "measures": [[0, 0], [0.3, 0.3], [0.6, 0.6], [1.0, 1.0]],
        },
        add_info={
            "novelty": novelty,
            "local_competition": local_competition,
        },
    )

    # All members are in the first front.
    assert set(indices.tolist()) == {0, 1, 2, 3}
    # Boundary points (0 and 3) have infinite crowding distance and rank first.
    assert set(indices[:2].tolist()) == {0, 3}

    expected_ranking = np.stack((novelty, local_competition), axis=-1)
    assert_allclose(ranking_values, expected_ranking)


def test_nslc_ranker_dominated(emitter):
    # Solution 0 dominates everything: strictly highest novelty AND strictly
    # highest local competition. Solutions 1 and 2 form the second front (mutually
    # non-dominated). Solution 3 is dominated by all others and forms the third
    # front.
    ranker = NSLCRanker()
    archive = _nslc_archive()

    indices, _ = ranker.rank(
        emitter=emitter,
        archive=archive,
        data={
            "solution": [[1, 2, 3]] * 4,
            "measures": [[0, 0], [0, 0], [0, 0], [0, 0]],
        },
        add_info={
            "novelty": np.array([1.0, 0.7, 0.5, 0.1]),
            "local_competition": np.array([5.0, 2.0, 3.0, 0.0]),
        },
    )

    # Front 0: {0}. Front 1: {1, 2}. Front 2: {3}.
    assert indices[0] == 0
    assert set(indices[1:3].tolist()) == {1, 2}
    assert indices[3] == 3


def test_nslc_ranker_ties_stable(emitter):
    # Two identical points (index 0 and 1) plus one dominated point (index 2).
    # Identical points do not dominate each other, so they share the first front.
    # The dominated point forms its own front.
    ranker = NSLCRanker()
    archive = _nslc_archive()

    indices, _ = ranker.rank(
        emitter=emitter,
        archive=archive,
        data={
            "solution": [[1, 2, 3]] * 3,
            "measures": [[0, 0], [0, 0], [0, 0]],
        },
        add_info={
            "novelty": np.array([1.0, 1.0, 0.2]),
            "local_competition": np.array([2.0, 2.0, 0.0]),
        },
    )

    assert set(indices[:2].tolist()) == {0, 1}
    assert indices[2] == 2


def test_nslc_ranker_with_proximity_archive(emitter):
    # End-to-end smoke test: fill a ProximityArchive(local_competition=True),
    # then add a batch, take its add_info, and feed it to NSLCRanker.
    archive = _nslc_archive()
    seed_solution = np.array([1, 2, 3], dtype=np.float64)

    # Seed with a few solutions so novelty queries have neighbors to compare to.
    archive.add(
        solution=np.tile(seed_solution, (4, 1)),
        objective=np.array([0.0, 1.0, 2.0, 3.0]),
        measures=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
    )

    batch_solution = np.tile(seed_solution, (3, 1))
    batch_objective = np.array([5.0, 0.0, 2.5])
    batch_measures = np.array([[0.5, 0.5], [5.0, 5.0], [2.5, 2.5]])
    add_info = archive.add(batch_solution, batch_objective, batch_measures)

    # Sanity-check that ProximityArchive produced the fields NSLC needs.
    assert "novelty" in add_info
    assert "local_competition" in add_info

    ranker = NSLCRanker()
    indices, ranking_values = ranker.rank(
        emitter=emitter,
        archive=archive,
        data={
            "solution": batch_solution,
            "objective": batch_objective,
            "measures": batch_measures,
        },
        add_info=add_info,
    )

    # The ranker should return a valid permutation with the expected shapes.
    assert sorted(indices.tolist()) == [0, 1, 2]
    assert ranking_values.shape == (3, 2)


def test_nslc_ranker_name_lookup():
    # Verify that the ranker is registered under both its full name and the
    # abbreviation, and that _get_ranker returns an NSLCRanker instance.
    full = _get_ranker("NSLCRanker", seed=0)
    short = _get_ranker("nslc", seed=0)
    assert isinstance(full, NSLCRanker)
    assert isinstance(short, NSLCRanker)
