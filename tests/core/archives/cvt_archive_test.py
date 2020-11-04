"""Tests for the CVTArhive."""
import numpy as np
import pytest

from ribs.archives import CVTArchive

# pylint: disable = invalid-name


@pytest.fixture
def _archive_fixture(use_kd_tree):
    """Returns a simple 2D archive.

    The archive has bounds (-1,1) and (-1,1), and should have 4 centroids at
    (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), and (0.5, -0.5).

    archive_with_entry has an entry with coordinates (1, 1), so it should be
    matched to centroid (0.5, 0.5).
    """
    samples = [[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]
    archive = CVTArchive([(-1, 1), (-1, 1)],
                         4,
                         samples=samples,
                         config={"use_kd_tree": use_kd_tree})

    archive_with_entry = CVTArchive([(-1, 1), (-1, 1)],
                                    4,
                                    samples=samples,
                                    config={"use_kd_tree": use_kd_tree})
    solution = np.array([1, 2, 3])
    behavior_values = np.array([1, 1])
    centroid = [0.5, 0.5]
    objective_value = 1.0
    archive_with_entry.add(solution, objective_value, behavior_values)

    return (
        archive,
        archive_with_entry,
        behavior_values,
        solution,
        objective_value,
        centroid,
    )


def _assert_archive_has_entry(archive, centroid, behavior_values,
                              objective_value, solution):
    """Assert that the archive has one specific entry."""
    archive_data = archive.as_pandas()
    assert len(archive_data) == 1

    # Start at 1 to ignore the "index" column.
    assert (archive_data.iloc[0][1:-1] == (list(centroid) +
                                           list(behavior_values) +
                                           [objective_value])).all()

    # Solution comparison separate since the solution is itself an array.
    archive_sol = archive_data.iloc[0][-1]
    assert archive_sol.shape == solution.shape
    assert np.all(archive_sol == solution)


def test_attributes_correctly_constructed(_archive_fixture):
    archive, *_ = _archive_fixture

    assert np.all(archive.lower_bounds == [-1, -1])
    assert np.all(archive.upper_bounds == [1, 1])


def test_add_to_archive(_archive_fixture):
    (_, archive_with_entry, behavior_values, solution, objective_value,
     centroid) = _archive_fixture

    _assert_archive_has_entry(archive_with_entry, centroid, behavior_values,
                              objective_value, solution)


def test_add_and_overwrite(_archive_fixture):
    """Test adding a new entry with a higher objective value."""
    (_, archive_with_entry, behavior_values, solution, objective_value,
     centroid) = _archive_fixture

    new_solution = solution - 1
    new_objective_value = objective_value + 1.0

    assert archive_with_entry.add(new_solution, new_objective_value,
                                  behavior_values)

    _assert_archive_has_entry(archive_with_entry, centroid, behavior_values,
                              new_objective_value, new_solution)


def test_add_without_overwrite(_archive_fixture):
    """Test adding a new entry with a lower objective value."""
    (_, archive_with_entry, behavior_values, solution, objective_value,
     centroid) = _archive_fixture

    new_solution = solution + 1
    new_objective_value = objective_value - 1.0

    assert not archive_with_entry.add(new_solution, new_objective_value,
                                      behavior_values)

    _assert_archive_has_entry(archive_with_entry, centroid, behavior_values,
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
    (_, archive_with_entry, behavior_values, solution, objective_value,
     _) = _archive_fixture
    retrieved = archive_with_entry.get_random_elite()
    assert np.all(retrieved[0] == solution)
    assert retrieved[1] == objective_value
    assert np.all(retrieved[2] == behavior_values)


def test_random_elite_fails_when_empty(_archive_fixture):
    (archive, *_) = _archive_fixture
    with pytest.raises(IndexError):
        archive.get_random_elite()


def test_as_pandas(_archive_fixture):
    (_, archive_with_entry, behavior_values, solution, objective_value,
     centroid) = _archive_fixture

    df = archive_with_entry.as_pandas()
    assert np.all(df.columns == [
        'index',
        'centroid-0',
        'centroid-1',
        'behavior-0',
        'behavior-1',
        'objective',
        'solution',
    ])
    assert (df.loc[0][1:] == np.array(
        [*centroid, *behavior_values, objective_value, solution],
        dtype=object,
    )).all()
