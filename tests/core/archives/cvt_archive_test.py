"""Tests for the CVTArhive."""
import unittest

import numpy as np
import pytest

from ribs.archives import CVTArchive

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

    # Check that the centroid is correct.
    index = archive_data.loc[0, "index"]
    assert (archive.centroids[index] == centroid).all()

    assert (archive_data.loc[0, "behavior-0":] == (list(behavior_values) +
                                                   [objective_value] +
                                                   list(solution))).all()


def test_samples_bad_shape(use_kd_tree):
    # The behavior space is 2D but samples are 3D.
    with pytest.raises(ValueError):
        CVTArchive([(-1, 1), (-1, 1)],
                   10,
                   samples=[[-1, -1, -1], [1, 1, 1]],
                   use_kd_tree=use_kd_tree)


def test_properties_are_correct(_cvt_data):
    assert np.all(_cvt_data.archive.lower_bounds == [-1, -1])
    assert np.all(_cvt_data.archive.upper_bounds == [1, 1])

    points = [[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]
    unittest.TestCase().assertCountEqual(_cvt_data.archive.samples.tolist(),
                                         points)
    unittest.TestCase().assertCountEqual(_cvt_data.archive.centroids.tolist(),
                                         points)


def test_construct_with_too_many_bins():
    with pytest.raises(RuntimeError):
        # Having a very low samples to bins ratio can result in centroids being
        # dropped, but even then, the odds of this happening are somewhat low,
        # so we do a few retries.
        for _ in range(10):
            archive = CVTArchive([(-1, 1), (-1, 1)], bins=5000, samples=7500)
            archive.initialize(2)


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
        'behavior-0',
        'behavior-1',
        'objective',
        'solution-0',
        'solution-1',
        'solution-2',
    ])
    assert (df.dtypes == [
        int,
        float,
        float,
        float,
        float,
        float,
        float,
    ]).all()

    index = df.loc[0, "index"]
    assert (_cvt_data.archive_with_entry.centroids[index] == _cvt_data.centroid
           ).all()

    assert (df.loc[0, "behavior-0":] == np.array([
        *_cvt_data.behavior_values,
        _cvt_data.objective_value,
        *_cvt_data.solution,
    ])).all()
