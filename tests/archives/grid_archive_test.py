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


def assert_archive_elite(archive, solution, objective, measures, grid_indices,
                         metadata):
    """Asserts that the archive has one specific elite."""
    assert len(archive) == 1
    elite = list(archive)[0]
    assert np.isclose(elite.solution, solution).all()
    assert np.isclose(elite.objective, objective).all()
    assert np.isclose(elite.measures, measures).all()
    assert elite.index == archive.grid_to_int_index([grid_indices])
    assert elite.metadata == metadata


def test_fails_on_dim_mismatch():
    with pytest.raises(ValueError):
        GridArchive(
            solution_dim=10,  #arbitrary
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


def test_grid_to_int_index(data):
    assert (data.archive.grid_to_int_index([data.grid_indices
                                           ])[0] == data.int_index)


def test_grid_to_int_index_wrong_shape(data):
    with pytest.raises(ValueError):
        data.archive.grid_to_int_index([data.grid_indices[:-1]])


def test_int_to_grid_index(data):
    assert np.all(
        data.archive.int_to_grid_index([data.int_index])[0] ==
        data.grid_indices)


def test_int_to_grid_index_wrong_shape(data):
    with pytest.raises(ValueError):
        data.archive.int_to_grid_index(data.int_index)


@pytest.mark.parametrize("dtype", [np.float64, np.float32],
                         ids=["float64", "float32"])
def test_values_go_to_correct_bin(dtype):
    """Bins tend to be a bit fuzzy at the edges due to floating point precision
    errors, so this test checks if we can get everything to land in the correct
    bin."""
    archive = GridArchive(
        solution_dim=0,
        dims=[10],
        ranges=[(0, 0.1)],
        epsilon=1e-6,
        dtype=dtype,
    )

    # Values below the lower bound land in the first bin.
    assert archive.index_of_single([-0.01]) == 0

    assert archive.index_of_single([0.0]) == 0
    assert archive.index_of_single([0.01]) == 1
    assert archive.index_of_single([0.02]) == 2
    assert archive.index_of_single([0.03]) == 3
    assert archive.index_of_single([0.04]) == 4
    assert archive.index_of_single([0.05]) == 5
    assert archive.index_of_single([0.06]) == 6
    assert archive.index_of_single([0.07]) == 7
    assert archive.index_of_single([0.08]) == 8
    assert archive.index_of_single([0.09]) == 9

    # Upper bound and above belong in last bin.
    assert archive.index_of_single([0.1]) == 9
    assert archive.index_of_single([0.11]) == 9
