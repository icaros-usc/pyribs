"""Tests for the GridArchive."""
import numpy as np
import pytest

from .conftest import get_archive_data

# pylint: disable = invalid-name


@pytest.fixture
def _data():
    """Data for grid archive tests."""
    return get_archive_data("GridArchive")


def _assert_archive_has_entry(archive, indices, behavior_values,
                              objective_value, solution):
    """Assert that the archive has one specific entry."""
    archive_df = archive.as_pandas()
    assert len(archive_df) == 1
    assert (archive_df.iloc[0] == (list(indices) + list(behavior_values) +
                                   [objective_value] + list(solution))).all()


def test_properties_are_correct(_data):
    assert np.all(_data.archive.dims == [10, 20])
    assert np.all(_data.archive.lower_bounds == [-1, -2])
    assert np.all(_data.archive.upper_bounds == [1, 2])
    assert np.all(_data.archive.interval_size == [2, 4])


def test_add_to_archive(_data):
    _assert_archive_has_entry(_data.archive_with_entry, _data.grid_indices,
                              _data.behavior_values, _data.objective_value,
                              _data.solution)


def test_add_with_low_behavior_val(_data):
    behavior_values = np.array([-2, -3])
    indices = (0, 0)
    _data.archive.add(_data.solution, _data.objective_value, behavior_values)

    _assert_archive_has_entry(_data.archive, indices, behavior_values,
                              _data.objective_value, _data.solution)


def test_add_with_high_behavior_val(_data):
    behavior_values = np.array([2, 3])
    indices = (9, 19)
    _data.archive.add(_data.solution, _data.objective_value, behavior_values)

    _assert_archive_has_entry(_data.archive, indices, behavior_values,
                              _data.objective_value, _data.solution)


def test_add_and_overwrite(_data):
    """Test adding a new entry with a higher objective value."""
    arbitrary_sol = _data.solution + 1
    high_objective_value = _data.objective_value + 1.0

    assert _data.archive_with_entry.add(arbitrary_sol, high_objective_value,
                                        _data.behavior_values)

    _assert_archive_has_entry(_data.archive_with_entry, _data.grid_indices,
                              _data.behavior_values, high_objective_value,
                              arbitrary_sol)


def test_add_without_overwrite(_data):
    """Test adding a new entry with a lower objective value."""
    arbitrary_sol = _data.solution + 1
    low_objective_value = _data.objective_value - 1.0

    assert not _data.archive_with_entry.add(arbitrary_sol, low_objective_value,
                                            _data.behavior_values)

    _assert_archive_has_entry(_data.archive_with_entry, _data.grid_indices,
                              _data.behavior_values, _data.objective_value,
                              _data.solution)


@pytest.mark.parametrize("with_entry", [True, False], ids=["nonempty", "empty"])
@pytest.mark.parametrize("include_solutions", [True, False],
                         ids=["solutions", "no_solutions"])
def test_as_pandas(_data, with_entry, include_solutions):
    if with_entry:
        df = _data.archive_with_entry.as_pandas(include_solutions)
    else:
        df = _data.archive.as_pandas(include_solutions)

    expected_columns = [
        'index-0', 'index-1', 'behavior-0', 'behavior-1', 'objective'
    ]
    expected_dtypes = [int, int, float, float, float]
    if include_solutions:
        expected_columns += ['solution-0', 'solution-1', 'solution-2']
        expected_dtypes += [float, float, float]
    assert (df.columns == expected_columns).all()
    assert (df.dtypes == expected_dtypes).all()

    if with_entry:
        expected_data = [
            *_data.grid_indices, *_data.behavior_values, _data.objective_value
        ]
        if include_solutions:
            expected_data += list(_data.solution)
        assert (df.loc[0] == np.array(expected_data)).all()
