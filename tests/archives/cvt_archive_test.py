"""Tests for the CVTArhive."""
import unittest

import numpy as np
import pytest

from ribs.archives import AddStatus, CVTArchive

from .conftest import get_archive_data

# pylint: disable = redefined-outer-name


@pytest.fixture
def data(use_kd_tree):
    """Data for CVT Archive tests."""
    return (get_archive_data("CVTArchive-kd_tree")
            if use_kd_tree else get_archive_data("CVTArchive-brute_force"))


def assert_archive_elite(archive, solution, objective, measures, centroid,
                         metadata):
    """Asserts that the archive has one specific elite."""
    assert len(archive) == 1
    elite = list(archive)[0]
    assert np.isclose(elite.solution, solution).all()
    assert np.isclose(elite.objective, objective).all()
    assert np.isclose(elite.measures, measures).all()
    assert np.isclose(archive.centroids[elite.index], centroid).all()
    assert elite.metadata == metadata


def test_samples_bad_shape(use_kd_tree):
    # The measure space is 2D but samples are 3D.
    with pytest.raises(ValueError):
        CVTArchive(solution_dim=0,
                   cells=10,
                   ranges=[(-1, 1), (-1, 1)],
                   samples=[[-1, -1, -1], [1, 1, 1]],
                   use_kd_tree=use_kd_tree)


def test_properties_are_correct(data):
    assert np.all(data.archive.lower_bounds == [-1, -1])
    assert np.all(data.archive.upper_bounds == [1, 1])

    points = [[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]
    unittest.TestCase().assertCountEqual(data.archive.samples.tolist(), points)
    unittest.TestCase().assertCountEqual(data.archive.centroids.tolist(),
                                         points)


def test_custom_centroids(use_kd_tree):
    centroids = np.array([[-0.25, -0.25], [0.25, 0.25]])
    archive = CVTArchive(solution_dim=3,
                         cells=centroids.shape[0],
                         ranges=[(-1, 1), (-1, 1)],
                         custom_centroids=centroids,
                         use_kd_tree=use_kd_tree)
    assert archive.samples is None
    assert (archive.centroids == centroids).all()


def test_custom_centroids_bad_shape(use_kd_tree):
    with pytest.raises(ValueError):
        # The centroids array should be of shape (10, 2) instead of just (1, 2),
        # hence a ValueError will be raised.
        CVTArchive(solution_dim=0,
                   cells=10,
                   ranges=[(-1, 1), (-1, 1)],
                   custom_centroids=[[0.0, 0.0]],
                   use_kd_tree=use_kd_tree)


@pytest.mark.parametrize("use_list", [True, False], ids=["list", "ndarray"])
def test_add_single_to_archive(data, use_list, add_mode):
    solution = data.solution
    objective = data.objective
    measures = data.measures
    metadata = data.metadata

    if use_list:
        solution = list(data.solution)
        measures = list(data.measures)

    if add_mode == "single":
        status, value = data.archive.add_single(solution, objective, measures,
                                                metadata)
    else:
        status, value = data.archive.add([solution], [objective], [measures],
                                         [metadata])

    assert status == AddStatus.NEW
    assert np.isclose(value, data.objective)
    assert_archive_elite(data.archive_with_elite, data.solution, data.objective,
                         data.measures, data.centroid, data.metadata)


def test_add_single_and_overwrite(data, add_mode):
    """Test adding a new solution with a higher objective value."""
    arbitrary_sol = data.solution + 1
    arbitrary_metadata = {"foobar": 12}
    high_objective = data.objective + 1.0

    if add_mode == "single":
        status, value = data.archive_with_elite.add_single(
            arbitrary_sol, high_objective, data.measures, arbitrary_metadata)
    else:
        status, value = data.archive_with_elite.add([arbitrary_sol],
                                                    [high_objective],
                                                    [data.measures],
                                                    [arbitrary_metadata])

    assert status == AddStatus.IMPROVE_EXISTING
    assert np.isclose(value, high_objective - data.objective)
    assert_archive_elite(data.archive_with_elite, arbitrary_sol, high_objective,
                         data.measures, data.centroid, arbitrary_metadata)


def test_add_single_without_overwrite(data, add_mode):
    """Test adding a new solution with a lower objective value."""
    arbitrary_sol = data.solution + 1
    arbitrary_metadata = {"foobar": 12}
    low_objective = data.objective - 1.0

    if add_mode == "single":
        status, value = data.archive_with_elite.add_single(
            arbitrary_sol, low_objective, data.measures, arbitrary_metadata)
    else:
        status, value = data.archive_with_elite.add([arbitrary_sol],
                                                    [low_objective],
                                                    [data.measures],
                                                    [arbitrary_metadata])

    assert status == AddStatus.NOT_ADDED
    assert np.isclose(value, low_objective - data.objective)
    assert_archive_elite(data.archive_with_elite, data.solution, data.objective,
                         data.measures, data.centroid, data.metadata)
