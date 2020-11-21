"""Tests for the GridArchive."""
import numpy as np
import pytest

from ribs.archives import GridArchive

# pylint: disable = invalid-name


@pytest.fixture
def _archive_fixture():
    """Returns a simple 2D archive."""
    solution = np.array([1, 2, 3])

    archive = GridArchive([10, 20], [(-1, 1), (-2, 2)])
    archive.initialize(len(solution))

    archive_with_entry = GridArchive([10, 20], [(-1, 1), (-2, 2)])
    archive_with_entry.initialize(len(solution))
    behavior_values = np.array([0, 0])
    indices = (5, 10)
    objective_value = 1.0
    archive_with_entry.add(solution, objective_value, behavior_values)

    return (
        archive,
        archive_with_entry,
        behavior_values,
        indices,
        solution,
        objective_value,
    )


def _assert_archive_has_entry(archive, indices, behavior_values,
                              objective_value, solution):
    """Assert that the archive has one specific entry."""
    archive_data = archive.as_pandas()
    assert len(archive_data) == 1
    assert (archive_data.iloc[0] == (list(indices) + list(behavior_values) +
                                     [objective_value] + list(solution))).all()


def test_attributes_correctly_constructed(_archive_fixture):
    archive, *_ = _archive_fixture

    assert np.all(archive.dims == [10, 20])
    assert np.all(archive.lower_bounds == [-1, -2])
    assert np.all(archive.upper_bounds == [1, 2])
    assert np.all(archive.interval_size == [2, 4])


def test_add_to_archive(_archive_fixture):
    (_, archive_with_entry, behavior_values, indices, solution,
     objective_value) = _archive_fixture

    _assert_archive_has_entry(archive_with_entry, indices, behavior_values,
                              objective_value, solution)


def test_add_with_low_behavior_val(_archive_fixture):
    archive, *_, solution, objective_value = _archive_fixture
    behavior_values = np.array([-2, -3])
    indices = (0, 0)
    archive.add(solution, objective_value, behavior_values)

    _assert_archive_has_entry(archive, indices, behavior_values,
                              objective_value, solution)


def test_add_with_high_behavior_val(_archive_fixture):
    archive, *_, solution, objective_value = _archive_fixture
    behavior_values = np.array([2, 3])
    indices = (9, 19)
    archive.add(solution, objective_value, behavior_values)

    _assert_archive_has_entry(archive, indices, behavior_values,
                              objective_value, solution)


def test_add_and_overwrite(_archive_fixture):
    """Test adding a new entry with a higher objective value."""
    (_, archive_with_entry, behavior_values, indices, solution,
     objective_value) = _archive_fixture

    new_solution = solution - 1
    new_objective_value = objective_value + 1.0

    assert archive_with_entry.add(new_solution, new_objective_value,
                                  behavior_values)

    _assert_archive_has_entry(archive_with_entry, indices, behavior_values,
                              new_objective_value, new_solution)


def test_add_without_overwrite(_archive_fixture):
    """Test adding a new entry with a lower objective value."""
    (_, archive_with_entry, behavior_values, indices, solution,
     objective_value) = _archive_fixture

    new_solution = solution + 1
    new_objective_value = objective_value - 1.0

    assert not archive_with_entry.add(new_solution, new_objective_value,
                                      behavior_values)

    _assert_archive_has_entry(archive_with_entry, indices, behavior_values,
                              objective_value, solution)


def test_archive_is_2d(_archive_fixture):
    archive, *_ = _archive_fixture
    assert archive.is_2d()


def test_new_archive_is_empty(_archive_fixture):
    (archive, *_) = _archive_fixture
    assert archive.is_empty()


def test_archive_with_entry_is_not_empty(_archive_fixture):
    (_, archive_with_entry, *_) = _archive_fixture
    assert not archive_with_entry.is_empty()


def test_random_elite_gets_single_elite(_archive_fixture):
    (_, archive_with_entry, behavior_values, _, solution,
     objective_value) = _archive_fixture
    retrieved = archive_with_entry.get_random_elite()
    assert np.all(retrieved[0] == solution)
    assert retrieved[1] == objective_value
    assert np.all(retrieved[2] == behavior_values)


def test_random_elite_fails_when_empty(_archive_fixture):
    (archive, *_) = _archive_fixture
    with pytest.raises(IndexError):
        archive.get_random_elite()


def test_as_pandas(_archive_fixture):
    (_, archive_with_entry, behavior_values, indices, solution,
     objective_value) = _archive_fixture

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
    assert (df.loc[0] == np.array([
        *indices,
        *behavior_values,
        objective_value,
        *solution,
    ])).all()
