"""Tests for the GridArchive."""
import numpy as np
import pytest

from .conftest import get_archive_data

# pylint: disable = invalid-name


@pytest.fixture
def _grid_data():
    """Data for grid archive tests."""
    return get_archive_data("GridArchive")


def _assert_archive_has_entry(archive, indices, behavior_values,
                              objective_value, solution):
    """Assert that the archive has one specific entry."""
    archive_df = archive.as_pandas()
    assert len(archive_df) == 1
    assert (archive_df.iloc[0] == (list(indices) + list(behavior_values) +
                                   [objective_value] + list(solution))).all()


def test_properties_are_correct(_grid_data):
    assert np.all(_grid_data.archive.dims == [10, 20])
    assert np.all(_grid_data.archive.lower_bounds == [-1, -2])
    assert np.all(_grid_data.archive.upper_bounds == [1, 2])
    assert np.all(_grid_data.archive.interval_size == [2, 4])


def test_add_to_archive(_grid_data):
    _assert_archive_has_entry(_grid_data.archive_with_entry,
                              _grid_data.grid_indices,
                              _grid_data.behavior_values,
                              _grid_data.objective_value, _grid_data.solution)


def test_add_with_low_behavior_val(_grid_data):
    behavior_values = np.array([-2, -3])
    indices = (0, 0)
    _grid_data.archive.add(_grid_data.solution, _grid_data.objective_value,
                           behavior_values)

    _assert_archive_has_entry(_grid_data.archive, indices, behavior_values,
                              _grid_data.objective_value, _grid_data.solution)


def test_add_with_high_behavior_val(_grid_data):
    behavior_values = np.array([2, 3])
    indices = (9, 19)
    _grid_data.archive.add(_grid_data.solution, _grid_data.objective_value,
                           behavior_values)

    _assert_archive_has_entry(_grid_data.archive, indices, behavior_values,
                              _grid_data.objective_value, _grid_data.solution)


def test_add_and_overwrite(_grid_data):
    """Test adding a new entry with a higher objective value."""
    arbitrary_sol = _grid_data.solution + 1
    high_objective_value = _grid_data.objective_value + 1.0

    assert _grid_data.archive_with_entry.add(arbitrary_sol,
                                             high_objective_value,
                                             _grid_data.behavior_values)

    _assert_archive_has_entry(_grid_data.archive_with_entry,
                              _grid_data.grid_indices,
                              _grid_data.behavior_values, high_objective_value,
                              arbitrary_sol)


def test_add_without_overwrite(_grid_data):
    """Test adding a new entry with a lower objective value."""
    arbitrary_sol = _grid_data.solution + 1
    low_objective_value = _grid_data.objective_value - 1.0

    assert not _grid_data.archive_with_entry.add(
        arbitrary_sol, low_objective_value, _grid_data.behavior_values)

    _assert_archive_has_entry(_grid_data.archive_with_entry,
                              _grid_data.grid_indices,
                              _grid_data.behavior_values,
                              _grid_data.objective_value, _grid_data.solution)


def test_as_pandas(_grid_data):
    df = _grid_data.archive_with_entry.as_pandas()
    assert np.all(df.columns == [
        'index-0',
        'index-1',
        'behavior-0',
        'behavior-1',
        'objective',
        'solution-0',
        'solution-1',
        'solution-2',
    ])
    assert (df.dtypes == [int, int, float, float, float, float, float,
                          float]).all()
    assert (df.loc[0] == np.array([
        *_grid_data.grid_indices,
        *_grid_data.behavior_values,
        _grid_data.objective_value,
        *_grid_data.solution,
    ])).all()
