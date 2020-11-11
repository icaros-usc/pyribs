"""Test for SlidingBoundaryArchive."""
import numpy as np
import pytest

from ribs.archives import SlidingBoundaryArchive

# pylint: disable = invalid-name


@pytest.fixture
def _archive_fixture():
    """Returns a simple 2D archive with sliding boundaries."""
    archive = SlidingBoundaryArchive([10, 20], [(-1, 1,), (-2, 2)])

    archive_with_entry = SlidingBoundaryArchive([10, 20], [(-1, 1), (-2, 2)])
    solution = np.array([1, 2, 3])
    behavior_values = np.array([0, 0])
    indices = (9, 19)
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
                              objective_value, solution, num_sol):
    """Assert that the archive has one specific entry."""
    archive_data = archive.as_pandas()
    print(archive_data)
    assert len(archive_data) == 1
    assert (archive_data.iloc[0][:-1] == (list(indices) +
                                          list(behavior_values) +
                                          [objective_value])).all()

    # Solution comparison separate since the solution is itself an array.
    archive_sol = archive_data.iloc[0][-1]
    assert archive_sol.shape == solution.shape
    assert np.all(archive_sol == solution)

    # should record all solutions
    assert len(archive.all_solutions) == num_sol
    assert len(archive.all_behavior_values) == num_sol
    assert len(archive.all_objective_values) == num_sol


def test_attributes_correctly_constructed(_archive_fixture):
    archive, *_ = _archive_fixture

    assert np.all(archive.dims == [10, 20])
    assert np.all(archive.lower_bounds == [-1, -2])
    assert np.all(archive.upper_bounds == [1, 2])
    assert np.all(archive.interval_size == [2, 4])

    # check the shape of boundaries
    assert len(archive.boundaries) == 2
    assert len(archive.boundaries[0]) == 10
    assert len(archive.boundaries[1]) == 20


def test_add_to_archive_with_remap(_archive_fixture):
    (_, archive_with_entry, behavior_values, indices, solution,
     objective_value) = _archive_fixture

    _assert_archive_has_entry(archive_with_entry, indices, behavior_values,
                              objective_value, solution, 1)


def test_add_to_archive_without_remap(_archive_fixture):
    _, archive_with_entry, *_, = _archive_fixture
    behavior_values = np.array([0, 0])
    solution = np.array([3, 4, 5])
    objective_value = 2
    indices = (9, 19)
    archive_with_entry.add(solution, objective_value, behavior_values)

    _assert_archive_has_entry(archive_with_entry, indices, behavior_values,
                              objective_value, solution, 2)

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
        'solution',
    ])
    assert (df.loc[0] == np.array(
        [*indices, *behavior_values, objective_value, solution],
        dtype=object,
    )).all()
