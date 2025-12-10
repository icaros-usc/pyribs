"""Tests for the CVTArhive."""

import unittest

import numpy as np
import pytest

from ribs.archives import AddStatus, CVTArchive

from .conftest import get_archive_data


@pytest.fixture
def data(nearest_neighbors):
    """Data for CVT Archive tests."""
    return get_archive_data(f"CVTArchive-{nearest_neighbors}")


def assert_archive_elite(archive, solution, objective, measures, centroid):
    """Asserts that the archive has one specific elite."""
    assert len(archive) == 1
    elite = next(iter(archive))
    assert np.isclose(elite["solution"], solution).all()
    assert np.isclose(elite["objective"], objective).all()
    assert np.isclose(elite["measures"], measures).all()
    assert np.isclose(archive.centroids[elite["index"]], centroid).all()


def test_properties_are_correct(data):
    assert np.all(data.archive.lower_bounds == [-1, -1])
    assert np.all(data.archive.upper_bounds == [1, 1])
    assert np.all(data.archive.interval_size == [2, 2])

    points = [[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]
    unittest.TestCase().assertCountEqual(data.archive.centroids.tolist(), points)  # noqa: PT009


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
def test_regular_centroids(dtype):
    """Check that generating centroids the regular way is alright."""
    archive = CVTArchive(
        solution_dim=3,
        centroids=100,
        ranges=[(-1, 1), (-1, 1)],
        dtype=dtype,
    )
    assert archive.centroids.shape == (100, 2)
    assert archive.centroids.dtype == dtype
    assert np.all(archive.centroids >= [-1, -1])
    assert np.all(archive.centroids <= [1, 1])


def test_centroids_are_same_with_same_seed():
    archive = CVTArchive(
        solution_dim=3,
        centroids=100,
        ranges=[(-1, 1), (-1, 1)],
        seed=42,
    )
    archive2 = CVTArchive(
        solution_dim=3,
        centroids=100,
        ranges=[(-1, 1), (-1, 1)],
        seed=42,
    )
    assert np.all(np.isclose(archive.centroids, archive2.centroids))


def test_custom_centroids(nearest_neighbors):
    centroids = np.asarray([[-0.25, -0.25], [0.25, 0.25]])
    archive = CVTArchive(
        solution_dim=3,
        centroids=centroids,
        ranges=[(-1, 1), (-1, 1)],
        nearest_neighbors=nearest_neighbors,
    )
    assert np.allclose(archive.centroids, centroids)


def test_custom_centroids_bad_shape(nearest_neighbors):
    with pytest.raises(
        ValueError, match=r"Expected centroids to be an array with shape .*"
    ):
        CVTArchive(
            solution_dim=3,
            # The centroids are 3D measures, but the ranges specify 2D measures.
            centroids=np.asarray([[-0.25, -0.25, -0.25], [0.25, 0.25, 0.25]]),
            ranges=[(-1, 1), (-1, 1)],
            nearest_neighbors=nearest_neighbors,
        )


def test_custom_centroids_from_file(tmp_path):
    centroids = np.asarray([[-0.25, -0.25], [0.25, 0.25]])
    file = tmp_path / "centroids.npy"
    np.save(file, centroids)

    archive = CVTArchive(
        solution_dim=3,
        centroids=file,
        ranges=[(-1, 1), (-1, 1)],
    )

    assert np.allclose(archive.centroids, centroids)


@pytest.mark.parametrize("use_list", [True, False], ids=["list", "ndarray"])
def test_add_single_to_archive(data, use_list, add_mode):
    solution = data.solution
    objective = data.objective
    measures = data.measures

    if use_list:
        solution = list(data.solution)
        measures = list(data.measures)

    if add_mode == "single":
        add_info = data.archive.add_single(solution, objective, measures)
    else:
        add_info = data.archive.add([solution], [objective], [measures])

    assert add_info["status"] == AddStatus.NEW
    assert np.isclose(add_info["value"], data.objective)
    assert_archive_elite(
        data.archive_with_elite,
        data.solution,
        data.objective,
        data.measures,
        data.centroid,
    )


def test_add_single_and_overwrite(data, add_mode):
    """Test adding a new solution with a higher objective value."""
    arbitrary_sol = data.solution + 1
    high_objective = data.objective + 1.0

    if add_mode == "single":
        add_info = data.archive_with_elite.add_single(
            arbitrary_sol, high_objective, data.measures
        )
    else:
        add_info = data.archive_with_elite.add(
            [arbitrary_sol], [high_objective], [data.measures]
        )

    assert add_info["status"] == AddStatus.IMPROVE_EXISTING
    assert np.isclose(add_info["value"], high_objective - data.objective)
    assert_archive_elite(
        data.archive_with_elite,
        arbitrary_sol,
        high_objective,
        data.measures,
        data.centroid,
    )


def test_add_single_without_overwrite(data, add_mode):
    """Test adding a new solution with a lower objective value."""
    arbitrary_sol = data.solution + 1
    low_objective = data.objective - 1.0

    if add_mode == "single":
        add_info = data.archive_with_elite.add_single(
            arbitrary_sol, low_objective, data.measures
        )
    else:
        add_info = data.archive_with_elite.add(
            [arbitrary_sol], [low_objective], [data.measures]
        )

    assert add_info["status"] == AddStatus.NOT_ADDED
    assert np.isclose(add_info["value"], low_objective - data.objective)
    assert_archive_elite(
        data.archive_with_elite,
        data.solution,
        data.objective,
        data.measures,
        data.centroid,
    )


def test_index_of_no_solutions(data):
    """When no solutions are in the archive, index_of should return empty array.

    Relevant because there's a special case for when nearest_neighbors="sklearn_nn".
    """
    indices = data.archive.index_of(np.empty((0, data.archive.measure_dim)))
    assert indices.shape == (0,)


def test_kdtree_query_kwargs():
    archive = CVTArchive(
        solution_dim=2,
        centroids=[[0, 1], [1, 0]],
        ranges=[(-1, 1), (-1, 1)],
        nearest_neighbors="scipy_kd_tree",
        kdtree_query_kwargs={
            "p": 1,
            "workers": 2,
        },
    )

    indices = archive.index_of([[0, 0.5], [0.5, 0]])

    assert np.all(indices == [0, 1])


def test_chunked_calculation():
    """Testing accuracy of chunked computation for nearest neighbors."""
    centroids = [
        [-1, 1],
        [0, 1],
        [1, 1],
        [-1, 0],
        [0, 0],
        [1, 0],
        [-1, -1],
        [0, -1],
        [1, -1],
    ]

    archive = CVTArchive(
        solution_dim=0,
        centroids=centroids,
        ranges=[(-1, 1), (-1, 1)],
        chunk_size=2,
        nearest_neighbors="brute_force",
    )
    measure_batch = [
        [-1, 1],
        [-1, 0.9],
        [-0.1, 1],
        [0.9, 0.9],
        [-0.9, 0],
        [0.1, 0],
        [1, 0],
        [-1, -0.9],
        [0.1, -0.9],
        [0.9, -0.9],
    ]
    closest_centroids = archive.index_of(measure_batch)
    correct_centroids = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    assert np.all(closest_centroids == correct_centroids)


def test_cosine_distance():
    archive = CVTArchive(
        solution_dim=2,
        centroids=[[0, 10], [1, 0]],
        ranges=[(-1, 1), (-1, 1)],
        nearest_neighbors="sklearn_nn",
        sklearn_nn_kwargs={
            "metric": "cosine",
        },
    )

    indices = archive.index_of([[0, 1], [1, 0.1]])

    # The first solution is closer to [0, 10] since we are using cosine rather than
    # Euclidean distance. Similarly, the second solution is closer to [1, 0].
    assert np.all(indices == [0, 1])
