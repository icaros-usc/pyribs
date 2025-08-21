"""Tests for the CVTArhive."""

import unittest

import numpy as np
import pytest

from ribs.archives import AddStatus, CVTArchive

from .conftest import get_archive_data


@pytest.fixture
def data(use_kd_tree):
    """Data for CVT Archive tests."""
    return (
        get_archive_data("CVTArchive-kd_tree")
        if use_kd_tree
        else get_archive_data("CVTArchive-brute_force")
    )


def assert_archive_elite(archive, solution, objective, measures, centroid):
    """Asserts that the archive has one specific elite."""
    assert len(archive) == 1
    elite = list(archive)[0]
    assert np.isclose(elite["solution"], solution).all()
    assert np.isclose(elite["objective"], objective).all()
    assert np.isclose(elite["measures"], measures).all()
    assert np.isclose(archive.centroids[elite["index"]], centroid).all()


def test_samples_bad_shape(use_kd_tree):
    # The measure space is 2D but samples are 3D.
    with pytest.raises(ValueError):
        CVTArchive(
            solution_dim=0,
            cells=10,
            ranges=[(-1, 1), (-1, 1)],
            samples=[[-1, -1, -1], [1, 1, 1]],
            use_kd_tree=use_kd_tree,
        )


def test_properties_are_correct(data):
    assert np.all(data.archive.lower_bounds == [-1, -1])
    assert np.all(data.archive.upper_bounds == [1, 1])
    assert np.all(data.archive.interval_size == [2, 2])

    points = [[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]
    unittest.TestCase().assertCountEqual(data.archive.samples.tolist(), points)
    unittest.TestCase().assertCountEqual(data.archive.centroids.tolist(), points)


def test_custom_centroids(use_kd_tree):
    centroids = np.array([[-0.25, -0.25], [0.25, 0.25]])
    archive = CVTArchive(
        solution_dim=3,
        cells=centroids.shape[0],
        ranges=[(-1, 1), (-1, 1)],
        custom_centroids=centroids,
        use_kd_tree=use_kd_tree,
    )
    assert archive.samples is None
    assert (archive.centroids == centroids).all()


def test_custom_centroids_bad_shape(use_kd_tree):
    with pytest.raises(ValueError):
        # The centroids array should be of shape (10, 2) instead of just (1, 2),
        # hence a ValueError will be raised.
        CVTArchive(
            solution_dim=0,
            cells=10,
            ranges=[(-1, 1), (-1, 1)],
            custom_centroids=[[0.0, 0.0]],
            use_kd_tree=use_kd_tree,
        )


@pytest.mark.parametrize("method", ["random", "sobol", "scrambled_sobol", "halton"])
def test_alternative_centroids(method):
    archive = CVTArchive(
        solution_dim=10,
        cells=100,
        ranges=[(0.1, 0.5), (-0.6, -0.2)],
        centroid_method=method,
    )

    centroid_x = archive.centroids[:, 0]
    centroid_y = archive.centroids[:, 1]

    # Centroids should have correct shape and be within bounds.
    assert archive.centroids.shape == (100, 2)
    assert np.all(centroid_x >= 0.1) and np.all(centroid_x <= 0.5)
    assert np.all(centroid_y >= -0.6) and np.all(centroid_y <= -0.2)


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
        cells=9,
        ranges=[(-1, 1), (-1, 1)],
        samples=10,
        chunk_size=2,
        custom_centroids=centroids,
        use_kd_tree=False,
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
