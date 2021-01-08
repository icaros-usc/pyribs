"""Test for SlidingBoundaryArchive."""
import numpy as np
import pytest

from .conftest import get_archive_data

# pylint: disable = invalid-name


@pytest.fixture
def _data():
    """Data for sliding boundary archive tests."""
    return get_archive_data("SlidingBoundaryArchive")


def _assert_archive_has_entry(archive, indices, behavior_values,
                              objective_value, solution):
    """Assert that the archive has one specific entry."""
    archive_data = archive.as_pandas()
    assert len(archive_data) == 1
    assert (archive_data.iloc[0] == (list(indices) + list(behavior_values) +
                                     [objective_value] + list(solution))).all()


def test_attributes_correctly_constructed(_data):
    assert np.all(_data.archive.dims == [10, 20])
    assert np.all(_data.archive.lower_bounds == [-1, -2])
    assert np.all(_data.archive.upper_bounds == [1, 2])
    assert np.all(_data.archive.interval_size == [2, 4])

    # Check the shape of boundaries.
    assert len(_data.archive.boundaries[0]) == 10
    assert len(_data.archive.boundaries[1]) == 20
    assert _data.archive.remap_frequency == 100
    assert _data.archive.buffer_capacity == 1000


def test_add_to_archive_with_remap(_data):
    # The first remap has been done while adding the first solution.
    _assert_archive_has_entry(_data.archive_with_entry, _data.grid_indices,
                              _data.behavior_values, _data.objective_value,
                              _data.solution)


def test_add_to_archive_with_full_buffer(_data):
    for _ in range(_data.archive.buffer_capacity + 1):
        _data.archive.add(_data.solution, _data.objective_value,
                          _data.behavior_values)

    _assert_archive_has_entry(_data.archive, _data.grid_indices,
                              _data.behavior_values, _data.objective_value,
                              _data.solution)


def test_add_to_archive_without_remap(_data):
    behavior_values = np.array([0.25, 0.25])
    solution = np.array([3, 4, 5])
    objective_value = 2
    indices = (9, 19)
    _data.archive_with_entry.add(solution, objective_value, behavior_values)

    _assert_archive_has_entry(_data.archive_with_entry, indices,
                              behavior_values, objective_value, solution)


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
