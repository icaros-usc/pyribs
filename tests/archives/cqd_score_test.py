"""Tests for cqd_score."""

import numpy as np
import pytest

from ribs.archives import GridArchive, ProximityArchive, cqd_score

from .conftest import get_archive_data

# pylint: disable = redefined-outer-name


@pytest.fixture
def data():
    """Data for grid archive tests."""
    return get_archive_data("GridArchive")


def test_cqd_score_detects_wrong_shapes(data):
    with pytest.raises(ValueError):
        cqd_score(
            data.archive,
            iterations=1,
            target_points=np.array([1.0]),  # Should be 3D.
            penalties=2,
            obj_min=0.0,
            obj_max=1.0,
            dist_max=np.sqrt(2),
        )

    with pytest.raises(ValueError):
        rng = np.random.default_rng()

        # (iterations, n, measure_dim)
        target_points = rng.uniform(size=(1, 5, 2))

        cqd_score(
            data.archive,
            iterations=1,
            target_points=target_points,
            penalties=[[1.0, 1.0]],  # Should be 1D.
            obj_min=0.0,
            obj_max=1.0,
            dist_max=np.sqrt(2),
        )


def test_cqd_score_with_one_elite():
    archive = GridArchive(solution_dim=2, dims=[10, 10], ranges=[(-1, 1), (-1, 1)])
    archive.add_single([4.0, 4.0], 1.0, [0.0, 0.0])

    score = cqd_score(
        archive,
        iterations=1,
        # With this target point, the solution above at [0, 0] has a normalized
        # distance of 0.5, since it is halfway between the archive bounds of
        # (-1, -1) and (1, 1).
        target_points=np.array([[[1.0, 1.0]]]),
        penalties=2,
        obj_min=0.0,
        obj_max=1.0,
        dist_max=np.linalg.norm(archive.upper_bounds - archive.lower_bounds),
    ).mean

    # For theta=0, the score should be 1.0 - 0 * 0.5 = 1.0
    # For theta=1, the score should be 1.0 - 1 * 0.5 = 0.5
    assert np.isclose(score, 1.0 + 0.5)


def test_cqd_score_with_max_dist():
    archive = GridArchive(solution_dim=2, dims=[10, 10], ranges=[(-1, 1), (-1, 1)])
    archive.add_single([4.0, 4.0], 0.5, [0.0, 1.0])

    score = cqd_score(
        archive,
        iterations=1,
        # With this target point and dist_max, the solution above at [0, 1]
        # has a normalized distance of 0.5, since it is one unit away.
        target_points=np.array([[[1.0, 1.0]]]),
        penalties=2,
        obj_min=0.0,
        obj_max=1.0,
        dist_max=2.0,
    ).mean

    # For theta=0, the score should be 0.5 - 0 * 0.5 = 0.5
    # For theta=1, the score should be 0.5 - 1 * 0.5 = 0.0
    assert np.isclose(score, 0.5 + 0.0)


def test_cqd_score_l1_norm():
    archive = GridArchive(solution_dim=2, dims=[10, 10], ranges=[(-1, 1), (-1, 1)])
    archive.add_single([4.0, 4.0], 0.5, [0.0, 0.0])

    score = cqd_score(
        archive,
        iterations=1,
        # With this target point and dist_max, the solution above at [0, 0]
        # has a normalized distance of 1.0, since it is two units away.
        target_points=np.array([[[1.0, 1.0]]]),
        penalties=2,
        obj_min=0.0,
        obj_max=1.0,
        dist_max=2.0,
        # L1 norm.
        dist_ord=1,
    ).mean

    # For theta=0, the score should be 0.5 - 0 * 1.0 = 0.5
    # For theta=1, the score should be 0.5 - 1 * 1.0 = -0.5
    assert np.isclose(score, 0.5 + -0.5)


def test_cqd_score_full_output():
    archive = GridArchive(solution_dim=2, dims=[10, 10], ranges=[(-1, 1), (-1, 1)])
    archive.add_single([4.0, 4.0], 1.0, [0.0, 0.0])

    result = cqd_score(
        archive,
        iterations=5,
        # With this target point, the solution above at [0, 0] has a normalized
        # distance of 0.5, since it is halfway between the archive bounds of
        # (-1, -1) and (1, 1).
        target_points=np.array(
            [
                [[1.0, 1.0]],
                [[1.0, 1.0]],
                [[1.0, 1.0]],
                [[-1.0, -1.0]],
                [[-1.0, -1.0]],
            ]
        ),
        penalties=2,
        obj_min=0.0,
        obj_max=1.0,
        dist_max=np.linalg.norm(archive.upper_bounds - archive.lower_bounds),
    )

    # For theta=0, the score should be 1.0 - 0 * 0.5 = 1.0
    # For theta=1, the score should be 1.0 - 1 * 0.5 = 0.5
    assert result.iterations == 5
    assert np.isclose(result.mean, 1.0 + 0.5)
    assert np.all(np.isclose(result.scores, 1.0 + 0.5))
    assert np.all(
        np.isclose(
            result.target_points,
            np.array(
                [
                    [[1.0, 1.0]],
                    [[1.0, 1.0]],
                    [[1.0, 1.0]],
                    [[-1.0, -1.0]],
                    [[-1.0, -1.0]],
                ]
            ),
        )
    )
    assert np.all(np.isclose(result.penalties, [0.0, 1.0]))
    assert np.isclose(result.obj_min, 0.0)
    assert np.isclose(result.obj_max, 1.0)
    # Distance from (-1,-1) to (1,1).
    assert np.isclose(result.dist_max, 2 * np.sqrt(2))
    assert result.dist_ord is None


def test_cqd_score_with_two_elites():
    archive = GridArchive(solution_dim=2, dims=[10, 10], ranges=[(-1, 1), (-1, 1)])
    archive.add_single([4.0, 4.0], 0.25, [0.0, 0.0])  # Elite 1.
    archive.add_single([4.0, 4.0], 0.0, [1.0, 1.0])  # Elite 2.

    score = cqd_score(
        archive,
        iterations=1,
        # With this target point, Elite 1 at [0, 0] has a normalized distance of
        # 0.5, since it is halfway between the archive bounds of (-1, -1) and
        # (1, 1).
        #
        # Elite 2 has a normalized distance of 0, since it is exactly at [1, 1].
        target_points=np.array([[[1.0, 1.0]]]),
        penalties=2,  # Penalties of 0 and 1.
        obj_min=0.0,
        obj_max=1.0,
        dist_max=np.linalg.norm(archive.upper_bounds - archive.lower_bounds),
    ).mean

    # For theta=0, the score should be max(0.25 - 0 * 0.5, 0 - 0 * 0) = 0.25
    # For theta=1, the score should be max(0.25 - 1 * 0.5, 0 - 1 * 0) = 0
    assert np.isclose(score, 0.25 + 0)


def test_proximity_archive_cqd_score():
    archive = ProximityArchive(
        solution_dim=2,
        measure_dim=2,
        k_neighbors=5,
        novelty_threshold=1.0,
    )
    archive.add_single([4.0, 4.0], 0.5, [0.0, 1.0])

    score = cqd_score(
        archive,
        iterations=1,
        # With this target point and dist_max, the solution above at [0, 1]
        # has a normalized distance of 0.5, since it is one unit away.
        target_points=np.array([[[1.0, 1.0]]]),
        penalties=2,
        obj_min=0.0,
        obj_max=1.0,
        dist_max=2.0,
    ).mean

    # For theta=0, the score should be 0.5 - 0 * 0.5 = 0.5
    # For theta=1, the score should be 0.5 - 1 * 0.5 = 0.0
    assert np.isclose(score, 0.5 + 0.0)
