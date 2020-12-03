"""Test for SlidingBoundaryArchive."""
import numpy as np
import pytest

from .conftest import get_archive_data

# pylint: disable = invalid-name


@pytest.fixture
def _sliding_boundary_data():
    """Data for sliding boundary archive tests."""
    return get_archive_data("SlidingBoundaryArchive")


def _assert_archive_has_entry(archive, indices, behavior_values,
                              objective_value, solution, num_sol):
    """Assert that the archive has one specific entry."""
    archive_data = archive.as_pandas()
    assert len(archive_data) == 1
    assert (archive_data.iloc[0] == (list(indices) + list(behavior_values) +
                                     [objective_value] + list(solution))).all()

    # Check size of buffer.
    assert archive.buffer.size == num_sol


def test_attributes_correctly_constructed(_sliding_boundary_data):
    archive, *_ = _sliding_boundary_data

    assert np.all(archive.dims == [10, 20])
    assert np.all(archive.lower_bounds == [-1, -2])
    assert np.all(archive.upper_bounds == [1, 2])
    assert np.all(archive.interval_size == [2, 4])

    # Check the shape of boundaries.
    assert len(archive.boundaries) == 2
    assert len(archive.boundaries[0]) == 10
    assert len(archive.boundaries[1]) == 20


def test_add_to_archive_with_remap(_sliding_boundary_data):
    (_, archive_with_entry, solution, objective_value, behavior_values, indices,
     _) = _sliding_boundary_data

    # The first remap has been done while adding the first solution.
    _assert_archive_has_entry(archive_with_entry, indices, behavior_values,
                              objective_value, solution, 1)

def test_add_to_archive_with_full_buffer(_sliding_boundary_data):
    (archive, _, solution, objective_value, behavior_values, indices,
     _) = _sliding_boundary_data

    for _ in range(archive.buffer.capacity + 1):
        archive.add(solution, objective_value, behavior_values)

    _assert_archive_has_entry(archive, indices, behavior_values,
                              objective_value, solution, archive.buffer.size)


def test_add_to_archive_without_remap(_sliding_boundary_data):
    _, archive_with_entry, *_, = _sliding_boundary_data
    behavior_values = np.array([0.25, 0.25])
    solution = np.array([3, 4, 5])
    objective_value = 2
    indices = (9, 19)
    archive_with_entry.add(solution, objective_value, behavior_values)

    _assert_archive_has_entry(archive_with_entry, indices, behavior_values,
                              objective_value, solution, 2)


def test_as_pandas(_sliding_boundary_data):
    (_, archive_with_entry, solution, objective_value, behavior_values, indices,
     _) = _sliding_boundary_data

    df = archive_with_entry.as_pandas()
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
    assert (df.loc[0] == np.array(
        [*indices, *behavior_values, objective_value, *solution],
        dtype=object,
    )).all()
