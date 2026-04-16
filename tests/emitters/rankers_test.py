"""Tests for the rankers."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ribs.archives import DensityArchive, GridArchive, ProximityArchive
from ribs.emitters.rankers import (
    DensityRanker,
    NoveltyRanker,
    NSLCClassicRanker,
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
    """Build a ProximityArchive configured for use with NSLCRanker tests."""
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


def test_nslc_ranker_interior_crowding_ordering(emitter):
    # Five points on the non-dominated front. Each strictly better than every
    # other on one of the two objectives => single front. The interior points
    # (1, 2, 3) all have finite crowding distance and should be ordered by the
    # *density* of their neighbors: an interior point whose two neighbors on the
    # front are far apart gets a larger crowding distance than one with close
    # neighbors.
    #
    # Front shape in (novelty, LC) space (both higher-is-better):
    #
    #    idx: 0     1     2     3     4
    #    nov: 1.00  0.80  0.55  0.30  0.00
    #    LC : 0.00  0.20  0.55  0.80  1.00
    #
    # Boundaries 0 and 4 get +inf crowding. Among interior 1, 2, 3 we compare the
    # sum of normalized gaps on both objectives. Point 2 has the widest neighbor
    # gap on both objectives, so it should outrank 1 and 3.
    ranker = NSLCRanker()
    archive = _nslc_archive()

    indices, _ = ranker.rank(
        emitter=emitter,
        archive=archive,
        data={
            "solution": [[1, 2, 3]] * 5,
            "measures": [[0, 0]] * 5,
        },
        add_info={
            "novelty": np.array([1.0, 0.8, 0.55, 0.3, 0.0]),
            "local_competition": np.array([0.0, 0.2, 0.55, 0.8, 1.0]),
        },
    )

    # All five are in the first front.
    assert sorted(indices.tolist()) == [0, 1, 2, 3, 4]
    # The two boundary points come first (they have infinite crowding distance).
    assert set(indices[:2].tolist()) == {0, 4}
    # Among interior points, point 2 has the widest neighbor gaps so it should
    # come before points 1 and 3.
    interior = indices[2:].tolist()
    assert interior[0] == 2


def test_nslc_ranker_single_solution(emitter):
    # Batch size 1: single solution forms its own front by itself. The ranker
    # should return a length-1 index array pointing at index 0 and a length-1
    # ranking_values array with the right shape.
    ranker = NSLCRanker()
    archive = _nslc_archive()

    indices, ranking_values = ranker.rank(
        emitter=emitter,
        archive=archive,
        data={
            "solution": [[1, 2, 3]],
            "measures": [[0, 0]],
        },
        add_info={
            "novelty": np.array([0.42]),
            "local_competition": np.array([3.0]),
        },
    )

    assert indices.tolist() == [0]
    assert ranking_values.shape == (1, 2)
    assert_allclose(ranking_values, [[0.42, 3.0]])


def test_nslc_ranker_missing_novelty(emitter):
    # add_info without 'novelty' should raise a KeyError with a clear message
    # that tells the user how to fix it.
    ranker = NSLCRanker()
    archive = _nslc_archive()

    with pytest.raises(KeyError, match="novelty"):
        ranker.rank(
            emitter=emitter,
            archive=archive,
            data={
                "solution": [[1, 2, 3]] * 2,
                "measures": [[0, 0]] * 2,
            },
            add_info={
                "local_competition": np.array([1, 2]),
            },
        )


def test_nslc_ranker_missing_local_competition(emitter):
    # add_info without 'local_competition' should raise a KeyError hinting at
    # local_competition=True on ProximityArchive.
    ranker = NSLCRanker()
    archive = _nslc_archive()

    with pytest.raises(KeyError, match="local_competition"):
        ranker.rank(
            emitter=emitter,
            archive=archive,
            data={
                "solution": [[1, 2, 3]] * 2,
                "measures": [[0, 0]] * 2,
            },
            add_info={
                "novelty": np.array([1.0, 0.5]),
            },
        )


def test_nslc_ranker_three_objective_paper_faithful(emitter):
    # Paper-faithful 3-objective NSLC: pass diversity_field to pull a third
    # higher-is-better objective from add_info. With identical (novelty, LC)
    # scores, the diversity objective is what decides the ordering.
    ranker = NSLCRanker(diversity_field="diversity")
    assert ranker.diversity_field == "diversity"
    archive = _nslc_archive()

    indices, ranking_values = ranker.rank(
        emitter=emitter,
        archive=archive,
        data={
            "solution": [[1, 2, 3]] * 3,
            "measures": [[0, 0]] * 3,
        },
        add_info={
            # All three solutions tie on (novelty, local_competition). Without a
            # third objective they'd all be in the same front and crowding
            # distance would rank them arbitrarily. With the diversity objective,
            # solution 2 strictly dominates 1, which strictly dominates 0, so
            # the resulting order is [2, 1, 0].
            "novelty": np.array([0.5, 0.5, 0.5]),
            "local_competition": np.array([1.0, 1.0, 1.0]),
            "diversity": np.array([0.1, 0.5, 0.9]),
        },
    )

    assert indices.tolist() == [2, 1, 0]
    assert ranking_values.shape == (3, 3)
    # ranking_values should echo the raw objectives in (novelty, LC, diversity)
    # order for each solution.
    assert_allclose(
        ranking_values,
        np.stack(
            (
                np.array([0.5, 0.5, 0.5]),
                np.array([1.0, 1.0, 1.0]),
                np.array([0.1, 0.5, 0.9]),
            ),
            axis=-1,
        ),
    )


def test_nslc_ranker_three_objective_from_data(emitter):
    # When the diversity objective is produced by the emitter rather than the
    # archive, the ranker should fall back to looking it up in `data`.
    ranker = NSLCRanker(diversity_field="geno_diversity")
    archive = _nslc_archive()

    indices, _ = ranker.rank(
        emitter=emitter,
        archive=archive,
        data={
            "solution": [[1, 2, 3]] * 3,
            "measures": [[0, 0]] * 3,
            "geno_diversity": np.array([0.2, 0.9, 0.5]),
        },
        add_info={
            "novelty": np.array([0.5, 0.5, 0.5]),
            "local_competition": np.array([1.0, 1.0, 1.0]),
        },
    )

    # Tied on (novelty, LC); geno_diversity strictly orders them, so index 1
    # (highest diversity) should be first.
    assert indices[0] == 1


def test_nslc_ranker_three_objective_missing_field(emitter):
    # Requesting a diversity_field that isn't in add_info *or* data should raise
    # a helpful KeyError rather than a silent NumPy error.
    ranker = NSLCRanker(diversity_field="mystery")
    archive = _nslc_archive()

    with pytest.raises(KeyError, match="mystery"):
        ranker.rank(
            emitter=emitter,
            archive=archive,
            data={
                "solution": [[1, 2, 3]] * 2,
                "measures": [[0, 0]] * 2,
            },
            add_info={
                "novelty": np.array([1.0, 0.5]),
                "local_competition": np.array([2.0, 1.0]),
            },
        )


def test_nslc_ranker_shape_mismatch(emitter):
    # If novelty and local_competition arrive with mismatched shapes, the ranker
    # should raise a ValueError before attempting to sort garbage.
    ranker = NSLCRanker()
    archive = _nslc_archive()

    with pytest.raises(ValueError, match="shape"):
        ranker.rank(
            emitter=emitter,
            archive=archive,
            data={
                "solution": [[1, 2, 3]] * 2,
                "measures": [[0, 0]] * 2,
            },
            add_info={
                "novelty": np.array([1.0, 0.5]),
                "local_competition": np.array([2.0, 1.0, 0.3]),  # wrong size
            },
        )


def test_nslc_ranker_defaults():
    # The default constructor should leave diversity_field as None, which means
    # we're operating in the 2-objective + crowding-distance mode.
    ranker = NSLCRanker()
    assert ranker.diversity_field is None


def test_nslc_classic_ranker_diversity_field():
    # NSLCClassicRanker should always use "genotypic_diversity" as the diversity field.
    ranker = NSLCClassicRanker()
    assert ranker.diversity_field == "genotypic_diversity"


def test_nslc_classic_ranker_name_lookup():
    # Verify that NSLCClassicRanker is registered under both its full name and the
    # abbreviation, and that _get_ranker returns an NSLCClassicRanker instance.
    full = _get_ranker("NSLCClassicRanker", seed=0)
    short = _get_ranker("nslc_classic", seed=0)
    assert isinstance(full, NSLCClassicRanker)
    assert isinstance(short, NSLCClassicRanker)
    assert full.diversity_field == "genotypic_diversity"
    assert short.diversity_field == "genotypic_diversity"


def test_nslc_classic_ranker_three_objective(emitter):
    # Test that NSLCClassicRanker properly uses the genotypic_diversity field
    # when it's available.
    archive = _nslc_archive()
    seed_solution = np.array([1, 2, 3], dtype=np.float64)

    # Seed with a few solutions.
    archive.add(
        solution=np.tile(seed_solution, (4, 1)),
        objective=np.array([0.0, 1.0, 2.0, 3.0]),
        measures=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
    )

    batch_solution = np.tile(seed_solution, (3, 1))
    batch_objective = np.array([5.0, 0.0, 2.5])
    batch_measures = np.array([[0.5, 0.5], [5.0, 5.0], [2.5, 2.5]])
    add_info = archive.add(batch_solution, batch_objective, batch_measures)

    # Add the genotypic_diversity field that NSLCClassicRanker expects.
    add_info["genotypic_diversity"] = np.array([10.0, 5.0, 15.0])

    ranker = NSLCClassicRanker()
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

    # The ranker should return a valid permutation with 3 ranking values.
    assert sorted(indices.tolist()) == [0, 1, 2]
    assert ranking_values.shape == (
        3,
        3,
    )  # 3 objectives: novelty, local_competition, diversity


def test_nslc_classic_ranker_missing_field(emitter):
    # NSLCClassicRanker should raise an error if genotypic_diversity is not provided.
    archive = _nslc_archive()
    seed_solution = np.array([1, 2, 3], dtype=np.float64)

    archive.add(
        solution=np.tile(seed_solution, (4, 1)),
        objective=np.array([0.0, 1.0, 2.0, 3.0]),
        measures=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
    )

    batch_solution = np.tile(seed_solution, (3, 1))
    batch_objective = np.array([5.0, 0.0, 2.5])
    batch_measures = np.array([[0.5, 0.5], [5.0, 5.0], [2.5, 2.5]])
    add_info = archive.add(batch_solution, batch_objective, batch_measures)

    # Note: not adding genotypic_diversity field

    ranker = NSLCClassicRanker()
    with pytest.raises(KeyError, match="genotypic_diversity"):
        ranker.rank(
            emitter=emitter,
            archive=archive,
            data={
                "solution": batch_solution,
                "objective": batch_objective,
                "measures": batch_measures,
            },
            add_info=add_info,
        )
