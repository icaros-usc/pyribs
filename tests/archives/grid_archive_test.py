"""Tests for the GridArchive."""
import numpy as np
import pytest

from ribs.archives import GridArchive

# pylint: disable=invalid-name


@pytest.fixture
def _archive_fixture():
    """Returns a simple 2D archive."""
    archive = GridArchive([10, 20], [(-1, 1), (-2, 2)])

    archive_with_entry = GridArchive([10, 20], [(-1, 1), (-2, 2)])
    solution = np.array([1, 2, 3])
    behavior_values = np.array([0, 0])
    indices = (5, 10)
    objective_value = 1.0
    archive_with_entry.add(solution, objective_value, behavior_values)

    return (archive, archive_with_entry, behavior_values, indices, solution,
            objective_value)


def test_attributes_correctly_constructed(_archive_fixture):
    archive, *_ = _archive_fixture

    assert np.all(archive.dims == [10, 20])
    assert np.all(archive.lower_bounds == [-1, -2])
    assert np.all(archive.upper_bounds == [1, 2])
    assert np.all(archive.interval_size == [2, 4])


def test_add_to_archive(_archive_fixture):
    (_, archive_with_entry, behavior_values, indices, solution,
     objective_value) = _archive_fixture

    retrieved = archive_with_entry.get(indices)
    assert retrieved is not None
    assert retrieved[0] == objective_value
    assert np.all(retrieved[1] == behavior_values)
    assert np.all(retrieved[2] == solution)


def test_add_and_overwrite(_archive_fixture):
    """Test adding a new entry with a higher objective value."""
    (_, archive_with_entry, behavior_values, indices, solution,
     objective_value) = _archive_fixture

    new_solution = solution - 1
    new_objective_value = objective_value + 1.0

    assert archive_with_entry.add(new_solution, new_objective_value,
                                  behavior_values)
    retrieved = archive_with_entry.get(indices)
    assert retrieved is not None
    assert retrieved[0] == new_objective_value
    assert np.all(retrieved[1] == behavior_values)
    assert np.all(retrieved[2] == new_solution)


def test_add_without_overwrite(_archive_fixture):
    """Test adding a new entry with a lower objective value."""
    (_, archive_with_entry, behavior_values, indices, solution,
     objective_value) = _archive_fixture

    new_solution = solution + 1
    new_objective_value = objective_value - 1.0

    assert not archive_with_entry.add(new_solution, new_objective_value,
                                      behavior_values)
    retrieved = archive_with_entry.get(indices)
    assert retrieved is not None
    assert retrieved[0] == objective_value
    assert np.all(retrieved[1] == behavior_values)
    assert np.all(retrieved[2] == solution)


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
    assert retrieved[0] == objective_value
    assert np.all(retrieved[1] == behavior_values)
    assert np.all(retrieved[2] == solution)


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
        'solution',
    ])
    assert (df.loc[0] == np.array(
        [*indices, *behavior_values, objective_value, solution],
        dtype=object,
    )).all()
