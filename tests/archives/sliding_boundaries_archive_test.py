"""Test for SlidingBoundariesArchive."""
import numpy as np
import pytest

from ribs.archives import AddStatus, SlidingBoundariesArchive

from .conftest import get_archive_data


@pytest.fixture
def _data():
    """Data for sliding boundary archive tests."""
    return get_archive_data("SlidingBoundariesArchive")


def _assert_archive_has_entry(archive, indices, behavior_values,
                              objective_value, solution):
    """Assert that the archive has one specific entry."""
    archive_data = archive.as_pandas()
    assert len(archive_data) == 1
    assert (archive_data.iloc[0] == (list(indices) + list(behavior_values) +
                                     [objective_value] + list(solution))).all()


def test_fails_on_dim_mismatch():
    with pytest.raises(ValueError):
        SlidingBoundariesArchive(
            dims=[10] * 2,  # 2D space here.
            ranges=[(-1, 1)] * 3,  # But 3D space here.
        )


def test_attributes_correctly_constructed(_data):
    assert np.all(_data.archive.dims == [10, 20])
    assert np.all(_data.archive.lower_bounds == [-1, -2])
    assert np.all(_data.archive.upper_bounds == [1, 2])
    assert np.all(_data.archive.interval_size == [2, 4])

    # Check the shape of boundaries.
    assert len(_data.archive.boundaries[0]) == 11
    assert len(_data.archive.boundaries[1]) == 21
    assert _data.archive.remap_frequency == 100
    assert _data.archive.buffer_capacity == 1000


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


def test_initial_remap(_data):
    """Checks that boundaries and entries are correct after initial remap."""
    # remap_frequency is (10 + 1) * (20 + 1)
    archive = SlidingBoundariesArchive([10, 20], [(-1, 1), (-2, 2)],
                                       remap_frequency=231,
                                       buffer_capacity=1000)
    archive.initialize(2)

    # Buffer should have 230 entries after this (since the first entry is
    # skipped).
    first = True
    expected_bcs = []
    for ix, x in enumerate(np.linspace(-1, 1, 11)):
        for iy, y in enumerate(np.linspace(-2, 2, 21)):
            if first:
                first = False
                continue

            # The second to last row and column get overridden by other entries
            # because we set their objective lower.
            if ix == 9 or iy == 19:
                obj = 1
            else:
                expected_bcs.append((x, y))
                obj = 2

            # Solutions are same as BCs.
            archive.add([x, y], obj, [x, y])

    # There are 199 entries because the last entry has not been inserted.
    assert archive.entries == 199

    # Buffer should now have 231 entries; hence it remaps.
    archive.add([-1, -2], 1, [-1, -2])
    expected_bcs.append((-1, -2))

    assert archive.entries == 200

    # Since we passed in unique entries generated with linspace, the boundaries
    # should come from linspace.
    assert np.isclose(archive.boundaries[0], np.linspace(-1, 1, 11)).all()
    assert np.isclose(archive.boundaries[1], np.linspace(-2, 2, 21)).all()

    # Check that all the BCs are as expected.
    pandas_bcs = archive.as_pandas(include_solutions=False)[[
        "behavior_0", "behavior_1"
    ]]
    bcs = list(pandas_bcs.itertuples(name=None, index=False))
    assert np.isclose(sorted(bcs), sorted(expected_bcs)).all()


def test_add_to_archive_with_full_buffer(_data):
    for _ in range(_data.archive.buffer_capacity + 1):
        _data.archive.add(_data.solution, _data.objective_value,
                          _data.behavior_values)

    # After adding the same entry multiple times, there should only be one
    # entry, and it should be at (0, 0).
    _assert_archive_has_entry(_data.archive, (0, 0), _data.behavior_values,
                              _data.objective_value, _data.solution)

    # Even if another entry is added, it should still go to the same cell
    # because the behavior values are clipped to the boundaries before being
    # inserted.
    _data.archive.add(2 * _data.solution, 2 * _data.objective_value,
                      2 * _data.behavior_values)
    _assert_archive_has_entry(_data.archive, (0, 0), 2 * _data.behavior_values,
                              2 * _data.objective_value, 2 * _data.solution)


def test_adds_solutions_from_old_archive(_data):
    """Solutions from previous archive should be inserted during remap."""
    archive = SlidingBoundariesArchive([10, 20], [(-1, 1), (-2, 2)],
                                       remap_frequency=231,
                                       buffer_capacity=231)
    archive.initialize(2)

    for x in np.linspace(-1, 1, 11):
        for y in np.linspace(-2, 2, 21):
            archive.add([x, y], 2, [x, y])

    assert archive.entries == 200

    # Archive gets remapped again, but it should maintain the same BCs since
    # solutions are the same. All the high-performing solutions should be
    # cleared from the buffer since the buffer only has capacity 200.
    for x in np.linspace(-1, 1, 11):
        for y in np.linspace(-2, 2, 21):
            archive.add([x, y], 1, [x, y])

    assert archive.entries == 200

    # The objective values from the previous archive should remain because they
    # are higher.
    assert (archive.as_pandas(include_solutions=False)["objective"] == 2).all()
