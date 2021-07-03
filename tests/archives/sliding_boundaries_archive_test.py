"""Test for SlidingBoundariesArchive."""
import numpy as np
import pytest

from ribs.archives import AddStatus, SlidingBoundariesArchive

from .conftest import get_archive_data
from .grid_archive_test import assert_archive_entry

# pylint: disable = redefined-outer-name


@pytest.fixture
def data():
    """Data for sliding boundary archive tests."""
    return get_archive_data("SlidingBoundariesArchive")


def test_fails_on_dim_mismatch():
    with pytest.raises(ValueError):
        SlidingBoundariesArchive(
            dims=[10] * 2,  # 2D space here.
            ranges=[(-1, 1)] * 3,  # But 3D space here.
        )


def test_attributes_correctly_constructed(data):
    assert np.all(data.archive.dims == [10, 20])
    assert np.all(data.archive.lower_bounds == [-1, -2])
    assert np.all(data.archive.upper_bounds == [1, 2])
    assert np.all(data.archive.interval_size == [2, 4])

    # Check the shape of boundaries.
    assert len(data.archive.boundaries[0]) == 11
    assert len(data.archive.boundaries[1]) == 21
    assert data.archive.remap_frequency == 100
    assert data.archive.buffer_capacity == 1000


@pytest.mark.parametrize("use_list", [True, False], ids=["list", "ndarray"])
def test_add_to_archive(data, use_list):
    if use_list:
        status, value = data.archive.add(list(data.solution),
                                         data.objective_value,
                                         list(data.behavior_values),
                                         data.metadata)
    else:
        status, value = data.archive.add(data.solution, data.objective_value,
                                         data.behavior_values, data.metadata)

    assert status == AddStatus.NEW
    assert np.isclose(value, data.objective_value)
    assert_archive_entry(data.archive_with_elite, data.solution,
                         data.objective_value, data.behavior_values,
                         data.grid_indices, data.metadata)


def test_add_and_overwrite(data):
    """Test adding a new entry with a higher objective value."""
    arbitrary_sol = data.solution + 1
    arbitrary_metadata = {"foobar": 12}
    high_objective_value = data.objective_value + 1.0

    status, value = data.archive_with_elite.add(arbitrary_sol,
                                                high_objective_value,
                                                data.behavior_values,
                                                arbitrary_metadata)
    assert status == AddStatus.IMPROVE_EXISTING
    assert np.isclose(value, high_objective_value - data.objective_value)
    assert_archive_entry(data.archive_with_elite, arbitrary_sol,
                         high_objective_value, data.behavior_values,
                         data.grid_indices, arbitrary_metadata)


def test_add_without_overwrite(data):
    """Test adding a new entry with a lower objective value."""
    arbitrary_sol = data.solution + 1
    arbitrary_metadata = {"foobar": 12}
    low_objective_value = data.objective_value - 1.0

    status, value = data.archive_with_elite.add(arbitrary_sol,
                                                low_objective_value,
                                                data.behavior_values,
                                                arbitrary_metadata)
    assert status == AddStatus.NOT_ADDED
    assert np.isclose(value, low_objective_value - data.objective_value)
    assert_archive_entry(data.archive_with_elite, data.solution,
                         data.objective_value, data.behavior_values,
                         data.grid_indices, data.metadata)


def test_initial_remap():
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
    # TODO(btjanaka): Use the archive.occupied property when it is available.
    assert len(archive.as_pandas(include_solutions=False)) == 199

    # Buffer should now have 231 entries; hence it remaps.
    archive.add([-1, -2], 1, [-1, -2])
    expected_bcs.append((-1, -2))

    # TODO(btjanaka): Use the archive.occupied property when it is available.
    assert len(archive.as_pandas(include_solutions=False)) == 200

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


def test_add_to_archive_with_full_buffer(data):
    for _ in range(data.archive.buffer_capacity + 1):
        data.archive.add(data.solution, data.objective_value,
                         data.behavior_values, data.metadata)

    # After adding the same entry multiple times, there should only be one
    # entry, and it should be at (0, 0).
    assert_archive_entry(data.archive, data.solution, data.objective_value,
                         data.behavior_values, (0, 0), data.metadata)

    # Even if another entry is added, it should still go to the same cell
    # because the behavior values are clipped to the boundaries before being
    # inserted.
    arbitrary_metadata = {"foobar": 12}
    data.archive.add(2 * data.solution, 2 * data.objective_value,
                     2 * data.behavior_values, arbitrary_metadata)
    assert_archive_entry(data.archive, 2 * data.solution,
                         2 * data.objective_value, 2 * data.behavior_values,
                         (0, 0), arbitrary_metadata)


def test_adds_solutions_from_old_archive():
    """Solutions from previous archive should be inserted during remap."""
    archive = SlidingBoundariesArchive([10, 20], [(-1, 1), (-2, 2)],
                                       remap_frequency=231,
                                       buffer_capacity=231)
    archive.initialize(2)

    for x in np.linspace(-1, 1, 11):
        for y in np.linspace(-2, 2, 21):
            archive.add([x, y], 2, [x, y])

    # TODO(btjanaka): Use the archive.occupied property when it is available.
    assert len(archive.as_pandas(include_solutions=False)) == 200

    # Archive gets remapped again, but it should maintain the same BCs since
    # solutions are the same. All the high-performing solutions should be
    # cleared from the buffer since the buffer only has capacity 200.
    for x in np.linspace(-1, 1, 11):
        for y in np.linspace(-2, 2, 21):
            archive.add([x, y], 1, [x, y])

    # TODO(btjanaka): Use the archive.occupied property when it is available.
    assert len(archive.as_pandas(include_solutions=False)) == 200

    # The objective values from the previous archive should remain because they
    # are higher.
    assert (archive.as_pandas(include_solutions=False)["objective"] == 2).all()
