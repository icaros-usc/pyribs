"""Tests for the CVTArhive."""
import unittest

import numpy as np
import pytest

from .conftest import get_archive_data

# pylint: disable = invalid-name


@pytest.fixture
def _cvt_data(use_kd_tree):
    """Data for CVT Archive tests."""
    return (get_archive_data("CVTArchive-kd_tree")
            if use_kd_tree else get_archive_data("CVTArchive-brute_force"))


def _assert_archive_has_entry(archive, centroid, behavior_values,
                              objective_value, solution):
    """Assert that the archive has one specific entry."""
    archive_data = archive.as_pandas()
    assert len(archive_data) == 1

    # Start at 1 to ignore the "index" column.
    assert (archive_data.iloc[0][1:] == (list(centroid) +
                                         list(behavior_values) +
                                         [objective_value] +
                                         list(solution))).all()


def test_properties_are_correct(_cvt_data):
    assert np.all(_cvt_data.archive.lower_bounds == [-1, -1])
    assert np.all(_cvt_data.archive.upper_bounds == [1, 1])

    points = [[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]
    unittest.TestCase().assertCountEqual(_cvt_data.archive.samples.tolist(),
                                         points)
    unittest.TestCase().assertCountEqual(_cvt_data.archive.centroids.tolist(),
                                         points)


def test_add_to_archive(_cvt_data):
    _assert_archive_has_entry(_cvt_data.archive_with_entry, _cvt_data.centroid,
                              _cvt_data.behavior_values,
                              _cvt_data.objective_value, _cvt_data.solution)


def test_add_and_overwrite(_cvt_data):
    """Test adding a new entry with a higher objective value."""
    arbitrary_sol = _cvt_data.solution + 1
    high_objective_value = _cvt_data.objective_value + 1.0

    assert _cvt_data.archive_with_entry.add(arbitrary_sol, high_objective_value,
                                            _cvt_data.behavior_values)

    _assert_archive_has_entry(_cvt_data.archive_with_entry, _cvt_data.centroid,
                              _cvt_data.behavior_values, high_objective_value,
                              arbitrary_sol)


def test_add_without_overwrite(_cvt_data):
    """Test adding a new entry with a lower objective value."""
    arbitrary_sol = _cvt_data.solution + 1
    low_objective_value = _cvt_data.objective_value - 1.0

    assert not _cvt_data.archive_with_entry.add(
        arbitrary_sol, low_objective_value, _cvt_data.behavior_values)

    _assert_archive_has_entry(_cvt_data.archive_with_entry, _cvt_data.centroid,
                              _cvt_data.behavior_values,
                              _cvt_data.objective_value, _cvt_data.solution)


def test_as_pandas(_cvt_data):
    df = _cvt_data.archive_with_entry.as_pandas()
    assert np.all(df.columns == [
        'index',
        'centroid-0',
        'centroid-1',
        'behavior-0',
        'behavior-1',
        'objective',
        'solution-0',
        'solution-1',
        'solution-2',
    ])
    assert (df.loc[0][1:] == np.array([
        *_cvt_data.centroid,
        *_cvt_data.behavior_values,
        _cvt_data.objective_value,
        *_cvt_data.solution,
    ])).all()
