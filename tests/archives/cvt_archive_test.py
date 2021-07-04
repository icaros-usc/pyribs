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


def assert_archive_elite(archive, solution, objective_value, behavior_values,
                         centroid, metadata):
    """Asserts that the archive has one specific elite."""
    elite = archive.table().item()
    assert np.isclose(elite.sol, solution).all()
    assert np.isclose(elite.obj, objective_value).all()
    assert np.isclose(elite.beh, behavior_values).all()
    assert np.isclose(archive.centroids[elite.idx], centroid).all()
    assert elite.meta == metadata


def test_samples_bad_shape(use_kd_tree):
    # The behavior space is 2D but samples are 3D.
    with pytest.raises(ValueError):
        CVTArchive(10, [(-1, 1), (-1, 1)],
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
    archive = CVTArchive(centroids.shape[0], [(-1, 1), (-1, 1)],
                         custom_centroids=centroids,
                         use_kd_tree=use_kd_tree)
    archive.initialize(solution_dim=3)
    assert archive.samples is None
    assert (archive.centroids == centroids).all()


def test_custom_centroids_bad_shape(use_kd_tree):
    with pytest.raises(ValueError):
        # The centroids array should be of shape (10, 2) instead of just (1, 2),
        # hence a ValueError will be raised.
        CVTArchive(10, [(-1, 1), (-1, 1)],
                   custom_centroids=[[0.0, 0.0]],
                   use_kd_tree=use_kd_tree)


@pytest.mark.parametrize("use_list", [True, False], ids=["list", "ndarray"])
def test_add_to_archive(data, use_list):
    if use_list:
        status, value = data.archive.add(list(data.solution),
                                         data.objective_value,
                                         list(data.behavior_values),
                                         data.metadata)
    else:
        status, value = data.archive.add(data.solution, data.objective_value,
                                         data.behavior_values, data.metadata)

    assert status == AddStatus.NEW
    assert np.isclose(value, data.objective_value)
    assert_archive_elite(data.archive_with_elite, data.solution,
                         data.objective_value, data.behavior_values,
                         data.centroid, data.metadata)


def test_add_and_overwrite(data):
    """Test adding a new solution with a higher objective value."""
    arbitrary_sol = data.solution + 1
    arbitrary_metadata = {"foobar": 12}
    high_objective_value = data.objective_value + 1.0

    status, value = data.archive_with_elite.add(arbitrary_sol,
                                                high_objective_value,
                                                data.behavior_values,
                                                arbitrary_metadata)
    assert status == AddStatus.IMPROVE_EXISTING
    assert np.isclose(value, high_objective_value - data.objective_value)
    assert_archive_elite(data.archive_with_elite, arbitrary_sol,
                         high_objective_value, data.behavior_values,
                         data.centroid, arbitrary_metadata)


def test_add_without_overwrite(data):
    """Test adding a new solution with a lower objective value."""
    arbitrary_sol = data.solution + 1
    arbitrary_metadata = {"foobar": 12}
    low_objective_value = data.objective_value - 1.0

    status, value = data.archive_with_elite.add(arbitrary_sol,
                                                low_objective_value,
                                                data.behavior_values,
                                                arbitrary_metadata)
    assert status == AddStatus.NOT_ADDED
    assert np.isclose(value, low_objective_value - data.objective_value)
    assert_archive_elite(data.archive_with_elite, data.solution,
                         data.objective_value, data.behavior_values,
                         data.centroid, data.metadata)
