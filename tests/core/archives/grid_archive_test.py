"""Tests for the GridArchive."""
import numpy as np
import pytest

from ribs.archives import AddStatus, GridArchive

from .conftest import get_archive_data


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


def test_fails_on_dim_mismatch():
    with pytest.raises(ValueError):
        GridArchive(
            dims=[10] * 2,  # 2D space here.
            ranges=[(-1, 1)] * 3,  # But 3D space here.
        )


def test_properties_are_correct(_data):
    assert np.all(_data.archive.dims == [10, 20])
    assert np.all(_data.archive.lower_bounds == [-1, -2])
    assert np.all(_data.archive.upper_bounds == [1, 2])
    assert np.all(_data.archive.interval_size == [2, 4])

    boundaries = _data.archive.boundaries
    assert len(boundaries) == 2
    assert np.isclose(boundaries[0], np.linspace(-1, 1, 10 + 1)).all()
    assert np.isclose(boundaries[1], np.linspace(-2, 2, 20 + 1)).all()


@pytest.mark.parametrize("use_list", [True, False], ids=["list", "ndarray"])
def test_add_to_archive(_data, use_list):
    if use_list:
        status, value = _data.archive.add(list(_data.solution),
                                          _data.objective_value,
                                          list(_data.behavior_values))
    else:
        status, value = _data.archive.add(_data.solution, _data.objective_value,
                                          _data.behavior_values)

    assert status == AddStatus.NEW
    assert np.isclose(value, _data.objective_value)
    _assert_archive_has_entry(_data.archive_with_entry, _data.grid_indices,
                              _data.behavior_values, _data.objective_value,
                              _data.solution)


def test_add_with_low_behavior_val(_data):
    behavior_values = np.array([-2, -3])
    indices = (0, 0)
    status, _ = _data.archive.add(_data.solution, _data.objective_value,
                                  behavior_values)
    assert status
    _assert_archive_has_entry(_data.archive, indices, behavior_values,
                              _data.objective_value, _data.solution)


def test_add_with_high_behavior_val(_data):
    behavior_values = np.array([2, 3])
    indices = (9, 19)
    status, _ = _data.archive.add(_data.solution, _data.objective_value,
                                  behavior_values)
    assert status
    _assert_archive_has_entry(_data.archive, indices, behavior_values,
                              _data.objective_value, _data.solution)


def test_add_and_overwrite(_data):
    """Test adding a new entry with a higher objective value."""
    arbitrary_sol = _data.solution + 1
    high_objective_value = _data.objective_value + 1.0

    status, value = _data.archive_with_entry.add(arbitrary_sol,
                                                 high_objective_value,
                                                 _data.behavior_values)
    assert status == AddStatus.IMPROVE_EXISTING
    assert np.isclose(value, high_objective_value - _data.objective_value)
    _assert_archive_has_entry(_data.archive_with_entry, _data.grid_indices,
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
    _assert_archive_has_entry(_data.archive_with_entry, _data.grid_indices,
                              _data.behavior_values, _data.objective_value,
                              _data.solution)


@pytest.mark.parametrize("with_entry", [True, False], ids=["nonempty", "empty"])
@pytest.mark.parametrize("include_solutions", [True, False],
                         ids=["solutions", "no_solutions"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32],
                         ids=["float64", "float32"])
def test_as_pandas(with_entry, include_solutions, dtype):
    data = get_archive_data("GridArchive", dtype)
    if with_entry:
        df = data.archive_with_entry.as_pandas(include_solutions)
    else:
        df = data.archive.as_pandas(include_solutions)

    expected_columns = [
        'index_0', 'index_1', 'behavior_0', 'behavior_1', 'objective'
    ]
    expected_dtypes = [int, int, dtype, dtype, dtype]
    if include_solutions:
        expected_columns += ['solution_0', 'solution_1', 'solution_2']
        expected_dtypes += [dtype, dtype, dtype]
    assert (df.columns == expected_columns).all()
    assert (df.dtypes == expected_dtypes).all()

    if with_entry:
        expected_data = [
            *data.grid_indices, *data.behavior_values, data.objective_value
        ]
        if include_solutions:
            expected_data += list(data.solution)
        assert (df.loc[0] == np.array(expected_data)).all()
