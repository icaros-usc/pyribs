"""Tests for the GridArchive."""
import numpy as np
import pytest

from ribs.archives import AddStatus, GridArchive

from .conftest import get_archive_data

# pylint: disable = redefined-outer-name


@pytest.fixture
def data():
    """Data for grid archive tests."""
    return get_archive_data("GridArchive")


def assert_archive_elite(archive, solution, objective_value, behavior_values,
                         indices, metadata):
    """Asserts that the archive has one specific elite."""
    assert len(archive) == 1
    elite = list(archive)[0]
    assert np.isclose(elite.sol, solution).all()
    assert np.isclose(elite.obj, objective_value).all()
    assert np.isclose(elite.beh, behavior_values).all()
    assert elite.idx == indices
    assert elite.meta == metadata


def test_fails_on_dim_mismatch():
    with pytest.raises(ValueError):
        GridArchive(
            dims=[10] * 2,  # 2D space here.
            ranges=[(-1, 1)] * 3,  # But 3D space here.
        )


def test_properties_are_correct(data):
    assert np.all(data.archive.dims == [10, 20])
    assert np.all(data.archive.lower_bounds == [-1, -2])
    assert np.all(data.archive.upper_bounds == [1, 2])
    assert np.all(data.archive.interval_size == [2, 4])

    boundaries = data.archive.boundaries
    assert len(boundaries) == 2
    assert np.isclose(boundaries[0], np.linspace(-1, 1, 10 + 1)).all()
    assert np.isclose(boundaries[1], np.linspace(-2, 2, 20 + 1)).all()


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
    assert_archive_elite(data.archive_with_elite, data.solution,
                         data.objective_value, data.behavior_values,
                         data.grid_indices, data.metadata)


def test_add_with_low_behavior_val(data):
    behavior_values = np.array([-2, -3])
    indices = (0, 0)
    status, _ = data.archive.add(data.solution, data.objective_value,
                                 behavior_values, data.metadata)
    assert status
    assert_archive_elite(data.archive, data.solution, data.objective_value,
                         behavior_values, indices, data.metadata)


def test_add_with_high_behavior_val(data):
    behavior_values = np.array([2, 3])
    indices = (9, 19)
    status, _ = data.archive.add(data.solution, data.objective_value,
                                 behavior_values, data.metadata)
    assert status
    assert_archive_elite(data.archive, data.solution, data.objective_value,
                         behavior_values, indices, data.metadata)


def test_add_and_overwrite(data):
    """Test adding a new solution with a higher objective value."""
    arbitrary_sol = data.solution + 1
    arbitrary_metadata = {"foobar": 12}
    high_objective_value = data.objective_value + 1.0

    status, value = data.archive_with_elite.add(arbitrary_sol,
                                                high_objective_value,
                                                data.behavior_values,
                                                arbitrary_metadata)
    assert status == AddStatus.IMPROVE_EXISTING
    assert np.isclose(value, high_objective_value - data.objective_value)
    assert_archive_elite(data.archive_with_elite, arbitrary_sol,
                         high_objective_value, data.behavior_values,
                         data.grid_indices, arbitrary_metadata)


def test_add_without_overwrite(data):
    """Test adding a new solution with a lower objective value."""
    arbitrary_sol = data.solution + 1
    arbitrary_metadata = {"foobar": 12}
    low_objective_value = data.objective_value - 1.0

    status, value = data.archive_with_elite.add(arbitrary_sol,
                                                low_objective_value,
                                                data.behavior_values,
                                                arbitrary_metadata)
    assert status == AddStatus.NOT_ADDED
    assert np.isclose(value, low_objective_value - data.objective_value)
    assert_archive_elite(data.archive_with_elite, data.solution,
                         data.objective_value, data.behavior_values,
                         data.grid_indices, data.metadata)
