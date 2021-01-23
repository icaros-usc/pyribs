"""Tests for the CVTArhive."""
import unittest

import numpy as np
import pytest

from ribs.archives import AddStatus, CVTArchive

from .conftest import get_archive_data


@pytest.fixture
def _data(use_kd_tree):
    """Data for CVT Archive tests."""
    return (get_archive_data("CVTArchive-kd_tree")
            if use_kd_tree else get_archive_data("CVTArchive-brute_force"))


def _assert_archive_has_entry(archive, centroid, behavior_values,
                              objective_value, solution):
    """Assert that the archive has one specific entry."""
    archive_data = archive.as_pandas()
    assert len(archive_data) == 1

    # Check that the centroid is correct.
    index = archive_data.loc[0, "index"]
    assert (archive.centroids[index] == centroid).all()

    assert (archive_data.loc[0, "behavior_0":] == (list(behavior_values) +
                                                   [objective_value] +
                                                   list(solution))).all()


def test_samples_bad_shape(use_kd_tree):
    # The behavior space is 2D but samples are 3D.
    with pytest.raises(ValueError):
        CVTArchive([(-1, 1), (-1, 1)],
                   10,
                   samples=[[-1, -1, -1], [1, 1, 1]],
                   use_kd_tree=use_kd_tree)


def test_properties_are_correct(_data):
    assert np.all(_data.archive.lower_bounds == [-1, -1])
    assert np.all(_data.archive.upper_bounds == [1, 1])

    points = [[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]
    unittest.TestCase().assertCountEqual(_data.archive.samples.tolist(), points)
    unittest.TestCase().assertCountEqual(_data.archive.centroids.tolist(),
                                         points)


def test_custom_centroids(use_kd_tree):
    centroids = np.array([[-0.25, -0.25], [0.25, 0.25]])
    archive = CVTArchive([(-1, 1), (-1, 1)],
                         bins=centroids.shape[0],
                         custom_centroids=centroids,
                         use_kd_tree=use_kd_tree)
    archive.initialize(solution_dim=3)
    assert archive.samples is None
    assert (archive.centroids == centroids).all()


def test_custom_centroids_bad_shape(use_kd_tree):
    with pytest.raises(ValueError):
        # The centroids array should be of shape (10, 2) instead of just (1, 2),
        # hence a ValueError will be raised.
        CVTArchive([(-1, 1), (-1, 1)],
                   bins=10,
                   custom_centroids=[[0.0, 0.0]],
                   use_kd_tree=use_kd_tree)


def test_add_to_archive(_data):
    status, value = _data.archive.add(_data.solution, _data.objective_value,
                                      _data.behavior_values)
    assert status == AddStatus.NEW
    assert np.isclose(value, _data.objective_value)
    _assert_archive_has_entry(_data.archive_with_entry, _data.centroid,
                              _data.behavior_values, _data.objective_value,
                              _data.solution)


def test_add_and_overwrite(_data):
    """Test adding a new entry with a higher objective value."""
    arbitrary_sol = _data.solution + 1
    high_objective_value = _data.objective_value + 1.0

    status, value = _data.archive_with_entry.add(arbitrary_sol,
                                                 high_objective_value,
                                                 _data.behavior_values)
    assert status == AddStatus.IMPROVE_EXISTING
    assert np.isclose(value, high_objective_value - _data.objective_value)
    _assert_archive_has_entry(_data.archive_with_entry, _data.centroid,
                              _data.behavior_values, high_objective_value,
                              arbitrary_sol)


def test_add_without_overwrite(_data):
    """Test adding a new entry with a lower objective value."""
    arbitrary_sol = _data.solution + 1
    low_objective_value = _data.objective_value - 1.0

    status, value = _data.archive_with_entry.add(arbitrary_sol,
                                                 low_objective_value,
                                                 _data.behavior_values)
    assert status == AddStatus.NOT_ADDED
    assert np.isclose(value, low_objective_value - _data.objective_value)
    _assert_archive_has_entry(_data.archive_with_entry, _data.centroid,
                              _data.behavior_values, _data.objective_value,
                              _data.solution)


@pytest.mark.parametrize("with_entry", [True, False], ids=["nonempty", "empty"])
@pytest.mark.parametrize("include_solutions", [True, False],
                         ids=["solutions", "no_solutions"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32],
                         ids=["float64", "float32"])
def test_as_pandas(use_kd_tree, with_entry, include_solutions, dtype):
    data = (get_archive_data("CVTArchive-kd_tree", dtype) if use_kd_tree else
            get_archive_data("CVTArchive-brute_force", dtype))
    if with_entry:
        df = data.archive_with_entry.as_pandas(include_solutions)
    else:
        df = data.archive.as_pandas(include_solutions)

    expected_columns = ['index', 'behavior_0', 'behavior_1', 'objective']
    expected_dtypes = [int, dtype, dtype, dtype]
    if include_solutions:
        expected_columns += ['solution_0', 'solution_1', 'solution_2']
        expected_dtypes += [dtype, dtype, dtype]
    assert (df.columns == expected_columns).all()
    assert (df.dtypes == expected_dtypes).all()

    if with_entry:
        index = df.loc[0, "index"]
        assert (data.archive_with_entry.centroids[index] == data.centroid).all()

        expected_data = [*data.behavior_values, data.objective_value]
        if include_solutions:
            expected_data += list(data.solution)
        assert (df.loc[0, "behavior_0":] == expected_data).all()
