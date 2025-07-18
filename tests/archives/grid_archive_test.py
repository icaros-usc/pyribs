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


def assert_archive_elite(archive, solution, objective, measures, grid_indices):
    """Asserts that the archive has one specific elite."""
    assert len(archive) == 1
    elite = list(archive)[0]
    assert np.isclose(elite["solution"], solution).all()
    assert np.isclose(elite["objective"], objective).all()
    assert np.isclose(elite["measures"], measures).all()
    assert elite["index"] == archive.grid_to_int_index([grid_indices])


def assert_archive_elites(
    archive,
    batch_size,
    solution_batch=None,
    objective_batch=None,
    measures_batch=None,
    grid_indices_batch=None,
    metadata_batch=None,
):
    """Asserts that the archive contains a batch of elites.

    Any of the batch items may be excluded by setting to None.
    """
    data = archive.data()

    # Check the number of solutions.
    assert len(data["index"]) == batch_size

    if grid_indices_batch is not None:
        index_batch = archive.grid_to_int_index(grid_indices_batch)

    # Enforce a one-to-one correspondence between entries in the archive and in
    # the provided input -- see
    # https://www.geeksforgeeks.org/check-two-unsorted-array-duplicates-allowed-elements/
    archive_covered = [False for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(len(data["index"])):
            if archive_covered[j]:
                continue

            if solution_batch is not None:
                if data["solution"].dtype.kind == "f":
                    solution_match = np.allclose(data["solution"][j], solution_batch[i])
                else:
                    solution_match = np.all(data["solution"][j] == solution_batch[i])
            else:
                solution_match = True

            objective_match = objective_batch is None or np.isclose(
                data["objective"][j], objective_batch[i]
            )

            if measures_batch is not None:
                if data["measures"].dtype.kind == "f":
                    measures_match = np.allclose(data["measures"][j], measures_batch[i])
                else:
                    measures_match = np.all(data["measures"][j] == measures_batch[i])
            else:
                measures_match = True

            index_match = (
                grid_indices_batch is None
                or
                # pylint: disable-next = possibly-used-before-assignment
                data["index"][j] == index_batch[i]
            )

            # Used for testing custom fields.
            metadata_match = (
                metadata_batch is None or data["metadata"][j] == metadata_batch[i]
            )

            if (
                solution_match
                and objective_match
                and measures_match
                and index_match
                and metadata_match
            ):
                archive_covered[j] = True

    assert np.all(archive_covered)


def test_fails_on_dim_mismatch():
    with pytest.raises(ValueError):
        GridArchive(
            solution_dim=10,  # arbitrary
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
def test_add_single_to_archive(data, use_list, add_mode):
    solution = data.solution
    objective = data.objective
    measures = data.measures

    if use_list:
        solution = list(data.solution)
        measures = list(data.measures)

    if add_mode == "single":
        add_info = data.archive.add_single(solution, objective, measures)
    else:
        add_info = data.archive.add([solution], [objective], [measures])

    assert add_info["status"] == AddStatus.NEW
    assert np.isclose(add_info["value"], data.objective)
    assert_archive_elite(
        data.archive_with_elite,
        data.solution,
        data.objective,
        data.measures,
        data.grid_indices,
    )


@pytest.mark.parametrize("use_list", [True, False], ids=["list", "ndarray"])
def test_add_single_to_archive_negative_objective(data, use_list, add_mode):
    """Same as test_add_single_to_archive, but negative objective since there
    are some weird cases when handling value calculations."""
    solution = data.solution
    objective = -data.objective
    measures = data.measures

    if use_list:
        solution = list(data.solution)
        measures = list(data.measures)

    if add_mode == "single":
        add_info = data.archive.add_single(solution, objective, measures)
    else:
        add_info = data.archive.add([solution], [objective], [measures])

    assert add_info["status"] == AddStatus.NEW
    assert np.isclose(add_info["value"], -data.objective)
    assert_archive_elite(
        data.archive_with_elite,
        data.solution,
        data.objective,
        data.measures,
        data.grid_indices,
    )


def test_add_single_with_low_measures(data, add_mode):
    measures = np.array([-2, -3])
    indices = (0, 0)
    if add_mode == "single":
        add_info = data.archive.add_single(data.solution, data.objective, measures)
    else:
        add_info = data.archive.add([data.solution], [data.objective], [measures])

    assert add_info["status"]
    assert_archive_elite(data.archive, data.solution, data.objective, measures, indices)


def test_add_single_with_high_measures(data, add_mode):
    measures = np.array([2, 3])
    indices = (9, 19)
    if add_mode == "single":
        add_info = data.archive.add_single(data.solution, data.objective, measures)
    else:
        add_info = data.archive.add([data.solution], [data.objective], [measures])
    assert add_info["status"]
    assert_archive_elite(data.archive, data.solution, data.objective, measures, indices)


def test_add_single_and_overwrite(data, add_mode):
    """Test adding a new solution with a higher objective value."""
    arbitrary_sol = data.solution + 1
    high_objective = data.objective + 1.0

    if add_mode == "single":
        add_info = data.archive_with_elite.add_single(
            arbitrary_sol, high_objective, data.measures
        )
    else:
        add_info = data.archive_with_elite.add(
            [arbitrary_sol], [high_objective], [data.measures]
        )

    assert add_info["status"] == AddStatus.IMPROVE_EXISTING
    assert np.isclose(add_info["value"], high_objective - data.objective)
    assert_archive_elite(
        data.archive_with_elite,
        arbitrary_sol,
        high_objective,
        data.measures,
        data.grid_indices,
    )


def test_add_single_without_overwrite(data, add_mode):
    """Test adding a new solution with a lower objective value."""
    arbitrary_sol = data.solution + 1
    low_objective = data.objective - 1.0

    if add_mode == "single":
        add_info = data.archive_with_elite.add_single(
            arbitrary_sol, low_objective, data.measures
        )
    else:
        add_info = data.archive_with_elite.add(
            [arbitrary_sol], [low_objective], [data.measures]
        )

    assert add_info["status"] == AddStatus.NOT_ADDED
    assert np.isclose(add_info["value"], low_objective - data.objective)
    assert_archive_elite(
        data.archive_with_elite,
        data.solution,
        data.objective,
        data.measures,
        data.grid_indices,
    )


def test_add_single_threshold_update(add_mode):
    archive = GridArchive(
        solution_dim=3,
        dims=[10, 10],
        ranges=[(-1, 1), (-1, 1)],
        threshold_min=-1.0,
        learning_rate=0.1,
    )
    solution = [1, 2, 3]
    measures = [0.1, 0.1]

    # Add a new solution to the archive.
    if add_mode == "single":
        add_info = archive.add_single(solution, 0.0, measures)
    else:
        add_info = archive.add([solution], [0.0], [measures])
        add_info = {name: arr[0] for name, arr in add_info.items()}

    assert add_info["status"] == 2  # NEW
    assert np.isclose(add_info["value"], 1.0)  # 0.0 - (-1.0)

    # Threshold should now be (1 - 0.1) * -1.0 + 0.1 * 0.0 = -0.9

    # These solutions are below the threshold and should not be inserted.
    if add_mode == "single":
        add_info = archive.add_single(solution, -0.95, measures)
    else:
        add_info = archive.add([solution], [-0.95], [measures])
        add_info = {name: arr[0] for name, arr in add_info.items()}

    assert add_info["status"] == 0  # NOT_ADDED
    assert np.isclose(add_info["value"], -0.05)  # -0.95 - (-0.9)

    # These solutions are above the threshold and should be inserted.
    if add_mode == "single":
        add_info = archive.add_single(solution, -0.8, measures)
    else:
        add_info = archive.add([solution], [-0.8], [measures])
        add_info = {name: arr[0] for name, arr in add_info.items()}

    assert add_info["status"] == 1  # IMPROVE_EXISTING
    assert np.isclose(add_info["value"], 0.1)  # -0.8 - (-0.9)


def test_add_single_after_clear(data):
    """After clearing, we should still get the same status and value when adding
    to the archive.

    https://github.com/icaros-usc/pyribs/pull/260
    """
    add_info = data.archive.add_single(data.solution, data.objective, data.measures)

    assert add_info["status"] == 2
    assert add_info["value"] == data.objective

    data.archive.clear()

    add_info = data.archive.add_single(data.solution, data.objective, data.measures)

    assert add_info["status"] == 2
    assert add_info["value"] == data.objective


def test_add_single_wrong_shapes(data):
    with pytest.raises(ValueError):
        data.archive.add_single(
            solution=[1, 1],  # 2D instead of 3D solution.
            objective=0,
            measures=[0, 0],
        )
    with pytest.raises(ValueError):
        data.archive.add_single(
            solution=[0, 0, 0],
            objective=0,
            measures=[1, 1, 1],  # 3D instead of 2D measures.
        )


def test_add_batch_all_new(data):
    add_info = data.archive.add(
        # 4 solutions of arbitrary value.
        solution=[[1, 2, 3]] * 4,
        # The first two solutions end up in separate cells, and the next two end
        # up in the same cell.
        objective=[0, 0, 0, 1],
        measures=[[0, 0], [0.25, 0.25], [0.5, 0.5], [0.5, 0.5]],
    )
    assert (add_info["status"] == 2).all()
    assert np.isclose(add_info["value"], [0, 0, 0, 1]).all()

    assert_archive_elites(
        archive=data.archive,
        batch_size=3,
        solution_batch=[[1, 2, 3]] * 3,
        objective_batch=[0, 0, 1],
        measures_batch=[[0, 0], [0.25, 0.25], [0.5, 0.5]],
        grid_indices_batch=[[5, 10], [6, 11], [7, 12]],
    )


def test_add_batch_none_inserted(data):
    add_info = data.archive_with_elite.add(
        solution=[[1, 2, 3]] * 4,
        objective=[data.objective - 1 for _ in range(4)],
        measures=[data.measures for _ in range(4)],
    )

    # All solutions were inserted into the same cell as the elite already in the
    # archive and had objective value 1 less.
    assert (add_info["status"] == 0).all()
    assert np.isclose(add_info["value"], -1.0).all()

    assert_archive_elites(
        archive=data.archive_with_elite,
        batch_size=1,
        solution_batch=[data.solution],
        objective_batch=[data.objective],
        measures_batch=[data.measures],
        grid_indices_batch=[data.grid_indices],
    )


def test_add_batch_with_improvement(data):
    add_info = data.archive_with_elite.add(
        solution=[[1, 2, 3]] * 4,
        objective=[data.objective + 1 for _ in range(4)],
        measures=[data.measures for _ in range(4)],
    )

    # All solutions were inserted into the same cell as the elite already in the
    # archive and had objective value 1 greater.
    assert (add_info["status"] == 1).all()
    assert np.isclose(add_info["value"], 1.0).all()

    assert_archive_elites(
        archive=data.archive_with_elite,
        batch_size=1,
        solution_batch=[[1, 2, 3]],
        objective_batch=[data.objective + 1],
        measures_batch=[data.measures],
        grid_indices_batch=[data.grid_indices],
    )


def test_add_batch_mixed_statuses(data):
    add_info = data.archive_with_elite.add(
        solution=[[1, 2, 3]] * 6,
        objective=[
            # Not added.
            data.objective - 1.0,
            # Not added.
            data.objective - 2.0,
            # Improve but not added.
            data.objective + 1.0,
            # Improve and added since it has higher objective.
            data.objective + 2.0,
            # New but not added.
            1.0,
            # New and added.
            2.0,
        ],
        measures=[
            data.measures,
            data.measures,
            data.measures,
            data.measures,
            [0, 0],
            [0, 0],
        ],
    )
    assert (add_info["status"] == [0, 0, 1, 1, 2, 2]).all()
    assert np.isclose(add_info["value"], [-1, -2, 1, 2, 1, 2]).all()

    assert_archive_elites(
        archive=data.archive_with_elite,
        batch_size=2,
        solution_batch=[[1, 2, 3]] * 2,
        objective_batch=[data.objective + 2.0, 2.0],
        measures_batch=[data.measures, [0, 0]],
        grid_indices_batch=[data.grid_indices, [5, 10]],
    )


def test_add_batch_first_solution_wins_in_ties(data):
    add_info = data.archive_with_elite.add(
        solution=[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ],
        objective=[
            # Ties for improvement.
            data.objective + 1.0,
            data.objective + 1.0,
            # Ties for new solution.
            3.0,
            3.0,
        ],
        measures=[
            data.measures,
            data.measures,
            [0, 0],
            [0, 0],
        ],
    )
    assert (add_info["status"] == [1, 1, 2, 2]).all()
    assert np.isclose(add_info["value"], [1, 1, 3, 3]).all()

    assert_archive_elites(
        archive=data.archive_with_elite,
        batch_size=2,
        # The first and third solution should be inserted since they come first.
        solution_batch=[[1, 2, 3], [7, 8, 9]],
        objective_batch=[data.objective + 1.0, 3.0],
        measures_batch=[data.measures, [0, 0]],
        grid_indices_batch=[data.grid_indices, [5, 10]],
    )


def test_add_batch_not_inserted_if_below_threshold_min():
    archive = GridArchive(
        solution_dim=3,
        dims=[10, 10],
        ranges=[(-1, 1), (-1, 1)],
        threshold_min=-10.0,
        learning_rate=0.1,
    )

    add_info = archive.add(
        solution=[[1, 2, 3]] * 4,
        objective=[-20.0, -20.0, 10.0, 10.0],
        measures=[[0.0, 0.0]] * 4,
    )

    # The first two solutions should not have been inserted since they did not
    # cross the threshold_min of -10.0.
    assert (add_info["status"] == [0, 0, 2, 2]).all()
    assert np.isclose(add_info["value"], [-10.0, -10.0, 20.0, 20.0]).all()

    assert_archive_elites(
        archive=archive,
        batch_size=1,
        solution_batch=[[1, 2, 3]],
        objective_batch=[10.0],
        measures_batch=[[0.0, 0.0]],
        grid_indices_batch=[[5, 5]],
    )


def test_add_batch_threshold_update():
    archive = GridArchive(
        solution_dim=3,
        dims=[10, 10],
        ranges=[(-1, 1), (-1, 1)],
        threshold_min=-1.0,
        learning_rate=0.1,
    )
    solution = [1, 2, 3]
    measures = [0.1, 0.1]
    measures2 = [-0.1, -0.1]

    # Add new solutions to the archive in two cells determined by measures and
    # measures2.
    add_info = archive.add(
        [solution, solution, solution, solution, solution, solution],
        # The first three solutions are inserted since they cross
        # threshold_min, but the last solution is not inserted since it does not
        # cross threshold_min.
        [0.0, 1.0, 2.0, 10.0, 100.0, -10.0],
        [measures, measures, measures, measures2, measures2, measures2],
    )

    assert (add_info["status"] == [2, 2, 2, 2, 2, 0]).all()
    assert np.isclose(
        add_info["value"], [1.0, 2.0, 3.0, 11.0, 101.0, -9.0]
    ).all()  # [...] - (-1.0)

    # Thresholds based on batch update rule should now be
    # (1 - 0.1)**3 * -1.0 + (0.0 + 1.0 + 2.0) / 3 * (1 - (1 - 0.1)**3) = -0.458
    # and
    # (1 - 0.1)**2 * -1.0 + (10.0 + 100.0) / 2 * (1 - (1 - 0.1)**2) = 9.64

    # Mix between solutions which are inserted and not inserted.
    add_info = archive.add(
        [solution, solution, solution, solution],
        [-0.95, -0.457, 9.63, 9.65],
        [measures, measures, measures2, measures2],
    )

    assert (add_info["status"] == [0, 1, 0, 1]).all()
    # [-0.95 - (-0.458), -0.458 - (-0.457), 9.63 - 9.64, 9.65 - 9.64]
    assert np.isclose(add_info["value"], [-0.492, 0.001, -0.01, 0.01]).all()

    # Thresholds should now be
    # (1 - 0.1)**1 * -0.458 + (-0.457) / 1 * (1 - (1 - 0.1)**1) = -0.4579
    # and
    # (1 - 0.1)**1 * 9.64 + (9.65) / 1 * (1 - (1 - 0.1)**1) = 9.641

    # Again mix between solutions which are inserted and not inserted.
    add_info = archive.add(
        [solution, solution, solution, solution],
        [-0.4580, -0.4578, 9.640, 9.642],
        [measures, measures, measures2, measures2],
    )

    assert (add_info["status"] == [0, 1, 0, 1]).all()
    assert np.isclose(add_info["value"], [-0.0001, 0.0001, -0.001, 0.001]).all()


def test_add_batch_threshold_update_inf_threshold_min():
    # These default values of threshold_min and learning_rate induce special
    # CMA-ME behavior for threshold updates.
    archive = GridArchive(
        solution_dim=3,
        dims=[10, 10],
        ranges=[(-1, 1), (-1, 1)],
        threshold_min=-np.inf,
        learning_rate=1.0,
    )
    solution = [1, 2, 3]
    measures = [0.1, 0.1]
    measures2 = [-0.1, -0.1]

    # Add new solutions to the archive.
    add_info = archive.add(
        [solution, solution, solution, solution, solution, solution],
        [0.0, 1.0, 2.0, -10.0, 10.0, 100.0],
        [measures, measures, measures, measures2, measures2, measures2],
    )

    # Value is same as objective since these are new cells.
    assert (add_info["status"] == [2, 2, 2, 2, 2, 2]).all()
    assert np.isclose(add_info["value"], [0.0, 1.0, 2.0, -10.0, 10.0, 100.0]).all()

    # Thresholds are updated based on maximum values in each cell, i.e. 2.0 and
    # 100.0.

    # Mix between solutions which are inserted and not inserted.
    add_info = archive.add(
        [solution, solution, solution, solution],
        [1.0, 10.0, 99.0, 101.0],
        [measures, measures, measures2, measures2],
    )

    assert (add_info["status"] == [0, 1, 0, 1]).all()
    # [1.0 - 2.0, 10.0 - 2.0, 99.0 - 100.0, 101.0 - 100.0]
    assert np.isclose(add_info["value"], [-1.0, 8.0, -1.0, 1.0]).all()


def test_add_batch_wrong_shapes(data):
    with pytest.raises(ValueError):
        data.archive.add(
            solution=[[1, 1]],  # 2D instead of 3D solution.
            objective=[0],
            measures=[[0, 0]],
        )
    with pytest.raises(ValueError):
        data.archive.add(
            solution=[[0, 0, 0]],
            objective=[[1]],  # Array instead of scalar objective.
            measures=[[0, 0]],
        )
    with pytest.raises(ValueError):
        data.archive.add(
            solution=[[0, 0, 0]],
            objective=[0],
            measures=[[1, 1, 1]],  # 3D instead of 2D measures.
        )


def test_add_batch_zero_length(data):
    """Nothing should happen when adding a batch with length 0."""
    add_info = data.archive.add(
        solution=np.ones((0, 3)),
        objective=np.ones((0,)),
        measures=np.ones((0, 2)),
    )

    assert len(add_info["status"]) == 0
    assert len(add_info["value"]) == 0
    assert len(data.archive) == 0


def test_add_batch_wrong_batch_size(data):
    with pytest.raises(ValueError):
        data.archive.add(
            solution=[[0, 0, 0]],
            objective=[1, 1],  # 2 objectives.
            measures=[[0, 0, 0]],
        )
    with pytest.raises(ValueError):
        data.archive.add(
            solution=[[0, 0, 0]],
            objective=[0, 0, 0],
            measures=[[1, 1, 1], [1, 1, 1]],  # 2 measures.
        )


def test_grid_to_int_index(data):
    assert data.archive.grid_to_int_index([data.grid_indices])[0] == data.int_index


def test_grid_to_int_index_wrong_shape(data):
    with pytest.raises(ValueError):
        data.archive.grid_to_int_index([data.grid_indices[:-1]])


def test_int_to_grid_index(data):
    assert np.all(
        data.archive.int_to_grid_index([data.int_index])[0] == data.grid_indices
    )


def test_int_to_grid_index_wrong_shape(data):
    with pytest.raises(ValueError):
        data.archive.int_to_grid_index(data.int_index)


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
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


def test_nonfinite_inputs(data):
    data.solution[0] = np.inf
    data.measures[0] = np.nan

    with pytest.raises(ValueError):
        data.archive.add([data.solution], -np.inf, [data.measures])
    with pytest.raises(ValueError):
        data.archive.add_single(data.solution, -np.inf, data.measures)
    with pytest.raises(ValueError):
        data.archive.retrieve([data.measures])
    with pytest.raises(ValueError):
        data.archive.retrieve_single(data.measures)
    with pytest.raises(ValueError):
        data.archive.index_of([data.measures])
    with pytest.raises(ValueError):
        data.archive.index_of_single(data.measures)


def test_retessellate_bad_learning_rate():
    archive = GridArchive(
        solution_dim=3,
        dims=[2, 2],
        ranges=[(-1, 1), (-1, 1)],
        # We only support retessellating with learning_rate=1.0.
        learning_rate=0.1,
        threshold_min=0.0,
    )
    with pytest.raises(ValueError):
        archive.retessellate([1, 2, 3])


def test_retessellate_wrong_dims():
    archive = GridArchive(
        solution_dim=3,
        dims=[2, 2],
        ranges=[(-1, 1), (-1, 1)],
    )
    with pytest.raises(ValueError):
        # This is a 3D measure space when the original measure space was 2D.
        archive.retessellate([1, 2, 3])


def test_retessellate_into_bigger_dims():
    archive = GridArchive(
        solution_dim=3,
        dims=[2, 2],
        ranges=[(-1, 1), (-1, 1)],
    )
    archive.add(
        solution=[[1, 2, 3], [4, 5, 6]],
        objective=[1.0, 2.0],
        measures=[[0.75, 0.75], [-0.75, -0.75]],
    )

    assert np.all(archive.dims == [2, 2])
    assert len(archive.boundaries) == 2
    assert np.isclose(archive.boundaries[0], [-1, 0, 1]).all()
    assert np.isclose(archive.boundaries[1], [-1, 0, 1]).all()
    assert_archive_elites(
        archive=archive,
        batch_size=2,
        solution_batch=[[1, 2, 3], [4, 5, 6]],
        objective_batch=[1.0, 2.0],
        measures_batch=[[0.75, 0.75], [-0.75, -0.75]],
        grid_indices_batch=[[1, 1], [0, 0]],
    )

    archive.retessellate([4, 4])

    assert np.all(archive.dims == [4, 4])
    assert len(archive.boundaries) == 2
    assert np.isclose(archive.boundaries[0], [-1, -0.5, 0, 0.5, 1]).all()
    assert np.isclose(archive.boundaries[1], [-1, -0.5, 0, 0.5, 1]).all()
    assert_archive_elites(
        archive=archive,
        batch_size=2,
        solution_batch=[[1, 2, 3], [4, 5, 6]],
        objective_batch=[1.0, 2.0],
        measures_batch=[[0.75, 0.75], [-0.75, -0.75]],
        grid_indices_batch=[[3, 3], [0, 0]],
    )


def test_retessellate_into_smaller_dims():
    archive = GridArchive(
        solution_dim=3,
        dims=[2, 2],
        ranges=[(-1, 1), (-1, 1)],
    )
    archive.add(
        solution=[[1, 2, 3], [4, 5, 6]],
        objective=[1.0, 2.0],
        measures=[[0.75, 0.75], [-0.75, -0.75]],
    )

    archive.retessellate([1, 1])

    assert np.all(archive.dims == [1, 1])
    assert len(archive.boundaries) == 2
    assert np.isclose(archive.boundaries[0], [-1, 1]).all()
    assert np.isclose(archive.boundaries[1], [-1, 1]).all()
    # Only the elite with higher objective should be kept.
    assert_archive_elites(
        archive=archive,
        batch_size=1,
        solution_batch=[[4, 5, 6]],
        objective_batch=[2.0],
        measures_batch=[[-0.75, -0.75]],
        grid_indices_batch=[[0, 0]],
    )


def test_scalar_solutions():
    archive = GridArchive(solution_dim=(), dims=[10, 20], ranges=[(-1, 1), (-2, 2)])
    assert archive.solution_dim == ()

    add_info = archive.add(
        solution=[1, 2, 3, 4],
        # The first two solutions end up in separate cells, and the next two end
        # up in the same cell.
        objective=[0, 0, 0, 1],
        measures=[[0, 0], [0.25, 0.25], [0.5, 0.5], [0.5, 0.5]],
    )
    assert (add_info["status"] == 2).all()
    assert np.isclose(add_info["value"], [0, 0, 0, 1]).all()

    assert_archive_elites(
        archive=archive,
        batch_size=3,
        solution_batch=[1, 2, 4],
        objective_batch=[0, 0, 1],
        measures_batch=[[0, 0], [0.25, 0.25], [0.5, 0.5]],
        grid_indices_batch=[[5, 10], [6, 11], [7, 12]],
    )


def test_str_solutions():
    archive = GridArchive(
        solution_dim=(),
        dims=[10, 20],
        ranges=[(-1, 1), (-2, 2)],
        dtype={"solution": object, "objective": np.float32, "measures": np.float32},
    )
    assert archive.solution_dim == ()
    assert archive.dtypes["solution"] == np.object_

    add_info = archive.add(
        solution=["One", "Two", "Three", "Four"],
        # The first two solutions end up in separate cells, and the next two end
        # up in the same cell.
        objective=[0, 0, 0, 1],
        measures=[[0, 0], [0.25, 0.25], [0.5, 0.5], [0.5, 0.5]],
    )
    assert (add_info["status"] == 2).all()
    assert np.isclose(add_info["value"], [0, 0, 0, 1]).all()

    assert_archive_elites(
        archive=archive,
        batch_size=3,
        solution_batch=["One", "Two", "Four"],
        objective_batch=[0, 0, 1],
        measures_batch=[[0, 0], [0.25, 0.25], [0.5, 0.5]],
        grid_indices_batch=[[5, 10], [6, 11], [7, 12]],
    )


def test_multi_dim_solutions():
    archive = GridArchive(solution_dim=(2, 3), dims=[10, 20], ranges=[(-1, 1), (-2, 2)])
    assert archive.solution_dim == (2, 3)

    add_info = archive.add(
        solution=np.arange(4 * 2 * 3).reshape((4, 2, 3)),
        # The first two solutions end up in separate cells, and the next two end
        # up in the same cell.
        objective=[0, 0, 0, 1],
        measures=[[0, 0], [0.25, 0.25], [0.5, 0.5], [0.5, 0.5]],
    )
    assert (add_info["status"] == 2).all()
    assert np.isclose(add_info["value"], [0, 0, 0, 1]).all()

    assert_archive_elites(
        archive=archive,
        batch_size=3,
        solution_batch=np.arange(4 * 2 * 3).reshape((4, 2, 3))[[0, 1, 3]],
        objective_batch=[0, 0, 1],
        measures_batch=[[0, 0], [0.25, 0.25], [0.5, 0.5]],
        grid_indices_batch=[[5, 10], [6, 11], [7, 12]],
    )
