"""Tests for the UnstructuredArchive."""
import numpy as np
import pytest

from ribs.archives import AddStatus, UnstructuredArchive
from tests.archives.conftest import get_archive_data

# pylint: disable = redefined-outer-name


@pytest.fixture
def data():
    """Data for grid archive tests."""
    return get_archive_data("UnstructuredArchive")


def assert_archive_elite(archive, solution, objective, measures, grid_indices):
    """Asserts that the archive has one specific elite."""
    assert len(archive) == 1
    elite = list(archive)[0]
    assert np.isclose(elite["solution"], solution).all()
    assert np.isclose(elite["objective"], objective).all()
    assert np.isclose(elite["measures"], measures).all()
    assert elite["index"] == np.array([grid_indices], dtype=np.int32)


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
        index_batch = np.array(grid_indices_batch, dtype=np.int32)

    # Enforce a one-to-one correspondence between entries in the archive and in
    # the provided input -- see
    # https://www.geeksforgeeks.org/check-two-unsorted-array-duplicates-allowed-elements/
    archive_covered = [False for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(len(data["index"])):
            if archive_covered[j]:
                continue

            solution_match = (solution_batch is None or np.isclose(
                data["solution"][j], solution_batch[i]).all())
            objective_match = (objective_batch is None or np.isclose(
                data["objective"][j], objective_batch[i]))
            measures_match = (measures_batch is None or np.isclose(
                data["measures"][j], measures_batch[i]).all())
            index_match = (grid_indices_batch is None or
                           data["index"][j] == index_batch[i])  # pylint: disable=E

            # Used for testing custom fields.
            metadata_match = (metadata_batch is None or
                              data["metadata"][j] == metadata_batch[i])

            if (solution_match and objective_match and measures_match and
                    index_match and metadata_match):
                archive_covered[j] = True

    assert np.all(archive_covered)


# def test_fails_on_dim_mismatch():
#     with pytest.raises(ValueError):
#         UnstructuredArchive(
#             solution_dim=10,  # arbitrary
#             measure_dim=3,  # arbitrary
#             k_neighbors=5,
#             novelty_threshold=1,
#         )


def test_properties_are_correct(data):
    data.archive.add(
        solution=[[1, 2, 3]] * 2,
        objective=[0, 0],
        measures=[[0, 0], [2, 2]],
    )

    # boundaries should reflect min/max measures
    assert np.all(data.archive.lower_bounds == [0, 0])
    assert np.all(data.archive.upper_bounds == [2, 2])

    # test boundaries contain [0, 2] using the midpoints and endpoints
    test_boundaries = np.array([0, 1, 2], dtype=data.archive.dtypes["measures"])
    boundaries = data.archive.boundaries

    assert len(boundaries) == 2
    assert np.isclose(boundaries[0], test_boundaries).all()
    assert np.isclose(boundaries[1], test_boundaries).all()

    data.archive.add(
        solution=[[1, 2, 3]] * 2,
        objective=[0] * 2,
        measures=[[-2, -2], [4, 4]],
    )

    # boundaries should update to the new min/max measures
    assert np.all(data.archive.lower_bounds == [-2, -2])
    assert np.all(data.archive.upper_bounds == [4, 4])

    # test boundaries contain [-2, 0, 2, 4] using the midpoints and endpoints
    test_boundaries = np.array([-2, -1, 1, 3, 4],
                               dtype=data.archive.dtypes["measures"])
    boundaries = data.archive.boundaries

    assert len(boundaries) == 2
    assert np.isclose(boundaries[0], test_boundaries).all()
    assert np.isclose(boundaries[1], test_boundaries).all()


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
    assert_archive_elite(data.archive_with_elite, data.solution, data.objective,
                         data.measures, data.grid_indices)


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
    assert_archive_elite(data.archive_with_elite, data.solution, data.objective,
                         data.measures, data.grid_indices)


def test_add_single_and_overwrite(data, add_mode):
    """Test adding a new solution with a higher objective value."""
    arbitrary_sol = data.solution + 1
    high_objective = data.objective + 1.0

    if add_mode == "single":
        add_info = data.archive_with_elite.add_single(arbitrary_sol,
                                                      high_objective,
                                                      data.measures)
    else:
        add_info = data.archive_with_elite.add([arbitrary_sol],
                                               [high_objective],
                                               [data.measures])

    assert add_info["status"] == AddStatus.IMPROVE_EXISTING
    assert np.isclose(add_info["value"], high_objective - data.objective)
    assert_archive_elite(data.archive_with_elite, arbitrary_sol, high_objective,
                         data.measures, data.grid_indices)


def test_add_single_without_overwrite(data, add_mode):
    """Test adding a new solution with a lower objective value."""
    arbitrary_sol = data.solution + 1
    low_objective = data.objective - 1.0

    if add_mode == "single":
        add_info = data.archive_with_elite.add_single(arbitrary_sol,
                                                      low_objective,
                                                      data.measures)
    else:
        add_info = data.archive_with_elite.add([arbitrary_sol], [low_objective],
                                               [data.measures])

    assert add_info["status"] == AddStatus.NOT_ADDED
    assert np.isclose(add_info["value"], low_objective - data.objective)
    assert_archive_elite(data.archive_with_elite, data.solution, data.objective,
                         data.measures, data.grid_indices)


def test_add_single_threshold_update(add_mode):
    archive = UnstructuredArchive(
        solution_dim=3,
        measure_dim=2,
        k_neighbors=5,
        novelty_threshold=1.0,
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
    add_info = data.archive.add_single(data.solution, data.objective,
                                       data.measures)

    assert add_info["status"] == 2
    assert add_info["value"] == data.objective

    data.archive.clear()

    add_info = data.archive.add_single(data.solution, data.objective,
                                       data.measures)

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
        solution=[[1, 2, 3]] * 5,
        # The first two solutions end up in separate cells, and the next two end
        # up in the same cell.
        objective=[0, 0, 1, 0, 1],
        measures=[[0, 0], [0.25, 0.25], [0.25, 0.25], [0.5, 0.5], [0.5, 0.5]],
    )
    assert (add_info["status"] == 2).all()
    assert np.isclose(add_info["value"], [0, 0, 1, 0, 1]).all()

    assert_archive_elites(
        archive=data.archive,
        batch_size=3,
        solution_batch=[[1, 2, 3]] * 3,
        objective_batch=[0, 1, 1],
        measures_batch=[[0, 0], [0.25, 0.25], [0.5, 0.5]],
        grid_indices_batch=[0, 1, 2],
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
            [2, 2],
            [2, 2],
        ],
    )
    assert (add_info["status"] == [0, 0, 1, 1, 2, 2]).all()
    assert np.isclose(add_info["value"], [-1, -2, 1, 2, 1, 2]).all()

    assert_archive_elites(
        archive=data.archive_with_elite,
        batch_size=2,
        solution_batch=[[1, 2, 3]] * 2,
        objective_batch=[data.objective + 2.0, 2.0],
        measures_batch=[data.measures, [2, 2]],
        grid_indices_batch=[data.grid_indices, [1]],
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
            [2, 2],
            [2, 2],
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
        measures_batch=[data.measures, [2, 2]],
        grid_indices_batch=[data.grid_indices, [1]],
    )


def test_add_batch_not_inserted_if_below_threshold_min():
    archive = UnstructuredArchive(
        solution_dim=3,
        measure_dim=2,
        k_neighbors=5,
        novelty_threshold=1.0,
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
        grid_indices_batch=[[0]],
    )


def test_add_batch_threshold_update():
    archive = UnstructuredArchive(
        solution_dim=3,
        measure_dim=2,
        k_neighbors=5,
        novelty_threshold=1.0,
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
        add_info["value"],
        [1.0, 2.0, 3.0, 11.0, 101.0, -9.0]).all()  # [...] - (-1.0)

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
    archive = UnstructuredArchive(
        solution_dim=3,
        measure_dim=2,
        k_neighbors=5,
        novelty_threshold=1.0,
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
    assert np.isclose(add_info["value"],
                      [0.0, 1.0, 2.0, -10.0, 10.0, 100.0]).all()

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


# def test_int_to_grid_index(data):
#     assert np.all(
#         data.archive.int_to_grid_index([data.int_index])[0] ==
#         data.grid_indices)

# def test_int_to_grid_index_wrong_shape(data):
#     with pytest.raises(ValueError):
#         data.archive.int_to_grid_index(data.int_index)


@pytest.mark.parametrize("dtype", [np.float64, np.float32],
                         ids=["float64", "float32"])
def test_values_go_to_correct_bin(dtype):
    """Bins tend to be a bit fuzzy at the edges due to floating point precision
    errors, so this test checks if we can get everything to land in the correct
    bin."""
    archive = UnstructuredArchive(
        solution_dim=0,
        measure_dim=1,
        k_neighbors=5,
        novelty_threshold=1.0,
        dtype=dtype,
    )
    archive.add(
        solution=[[]] * 2,
        objective=[0, 0],
        measures=[[0], [2]],
    )

    # closest index is 0
    assert archive.index_of_single([0.0]) == 0
    assert archive.index_of_single([0.99]) == 0
    assert archive.index_of_single([1.0]) == 0
    assert archive._store.capacity == 2  # pylint: disable=W0212

    # closest index is 1
    assert archive.index_of_single([1.01]) == 1
    assert archive.index_of_single([1.99]) == 1
    assert archive.index_of_single([2.0]) == 1
    assert archive._store.capacity == 2  # pylint: disable=W0212

    # closest index is 0, and the archive is allowed to grow but
    # the solution is in an overdense region
    assert archive.index_of_single([0.0], resize=True) == 0
    assert archive.index_of_single([0.99], resize=True) == 0
    assert archive.index_of_single([1.0], resize=True) == 0
    assert archive._store.capacity == 2  # pylint: disable=W0212

    # closest index is 1, and the archive is allowed to grow but
    # the solution is in an overdense region
    assert archive.index_of_single([1.01], resize=True) == 1
    assert archive.index_of_single([1.99], resize=True) == 1
    assert archive.index_of_single([2.0], resize=True) == 1
    assert archive._store.capacity == 2  # pylint: disable=W0212

    # closest index is 1, but the solution is in a sparse region
    # so the archive grows
    assert archive.index_of_single([4.0], resize=True) == 2
    assert archive._store.capacity == 3  # pylint: disable=W0212


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


def test_cqd_score_detects_wrong_shapes(data):
    with pytest.raises(ValueError):
        data.archive.cqd_score(
            iterations=1,
            target_points=np.array([1.0]),  # Should be 3D.
            penalties=2,
            obj_min=0.0,
            obj_max=1.0,
        )

    with pytest.raises(ValueError):
        data.archive.cqd_score(
            iterations=1,
            target_points=3,
            penalties=[[1.0, 1.0]],  # Should be 1D.
            obj_min=0.0,
            obj_max=1.0,
        )


def test_cqd_score_with_one_elite():
    archive = UnstructuredArchive(
        solution_dim=2,
        measure_dim=2,
        k_neighbors=5,
        novelty_threshold=1.0,
    )
    archive.add_single([4.0, 4.0], 1.0, [0.0, 0.0])

    score = archive.cqd_score(
        iterations=1,
        # With this target point, the solution above at [0, 0] has a normalized
        # distance of 0.5, since it is halfway between the archive bounds of
        # (-1, -1) and (1, 1).
        target_points=np.array([[[1.0, 1.0]]]),
        penalties=2,
        obj_min=0.0,
        obj_max=1.0,
    ).mean

    # For theta=0, the score should be 1.0 - 0 * 0.5 = 1.0
    # For theta=1, the score should be 1.0 - 1 * 0.5 = 0.5
    assert np.isclose(score, 1.0 + 0.5)


def test_cqd_score_with_max_dist():
    archive = UnstructuredArchive(
        solution_dim=2,
        measure_dim=2,
        k_neighbors=5,
        novelty_threshold=1.0,
    )
    archive.add_single([4.0, 4.0], 0.5, [0.0, 1.0])

    score = archive.cqd_score(
        iterations=1,
        # With this target point and dist_max, the solution above at [0, 1]
        # has a normalized distance of 0.5, since it is one unit away.
        target_points=np.array([[[1.0, 1.0]]]),
        penalties=2,
        obj_min=0.0,
        obj_max=1.0,
        dist_max=2.0,
    ).mean

    # For theta=0, the score should be 0.5 - 0 * 0.5 = 0.5
    # For theta=1, the score should be 0.5 - 1 * 0.5 = 0.0
    assert np.isclose(score, 0.5 + 0.0)


def test_cqd_score_l1_norm():
    archive = UnstructuredArchive(
        solution_dim=2,
        measure_dim=2,
        k_neighbors=5,
        novelty_threshold=1.0,
    )
    archive.add_single([4.0, 4.0], 0.5, [0.0, 0.0])

    score = archive.cqd_score(
        iterations=1,
        # With this target point and dist_max, the solution above at [0, 0]
        # has a normalized distance of 1.0, since it is two units away.
        target_points=np.array([[[1.0, 1.0]]]),
        penalties=2,
        obj_min=0.0,
        obj_max=1.0,
        dist_max=2.0,
        # L1 norm.
        dist_ord=1,
    ).mean

    # For theta=0, the score should be 0.5 - 0 * 1.0 = 0.5
    # For theta=1, the score should be 0.5 - 1 * 1.0 = -0.5
    assert np.isclose(score, 0.5 + -0.5)


def test_cqd_score_full_output():
    archive = UnstructuredArchive(
        solution_dim=2,
        measure_dim=2,
        k_neighbors=5,
        novelty_threshold=1.0,
    )
    archive.add_single([4.0, 4.0], 1.0, [0.0, 0.0])

    result = archive.cqd_score(
        iterations=5,
        # With this target point, the solution above at [0, 0] has a normalized
        # distance of 0.5, since it is halfway between the archive bounds of
        # (-1, -1) and (1, 1).
        target_points=np.array([
            [[1.0, 1.0]],
            [[1.0, 1.0]],
            [[1.0, 1.0]],
            [[-1.0, -1.0]],
            [[-1.0, -1.0]],
        ]),
        penalties=2,
        obj_min=0.0,
        obj_max=1.0,
    )

    # For theta=0, the score should be 1.0 - 0 * 0.5 = 1.0
    # For theta=1, the score should be 1.0 - 1 * 0.5 = 0.5
    assert result.iterations == 5
    assert np.isclose(result.mean, 1.0 + 0.5)
    assert np.all(np.isclose(result.scores, 1.0 + 0.5))
    assert np.all(
        np.isclose(
            result.target_points,
            np.array([
                [[1.0, 1.0]],
                [[1.0, 1.0]],
                [[1.0, 1.0]],
                [[-1.0, -1.0]],
                [[-1.0, -1.0]],
            ])))
    assert np.all(np.isclose(result.penalties, [0.0, 1.0]))
    assert np.isclose(result.obj_min, 0.0)
    assert np.isclose(result.obj_max, 1.0)
    # Distance from (-1,-1) to (1,1).
    assert np.isclose(result.dist_max, 2 * np.sqrt(2))
    assert result.dist_ord is None


def test_cqd_score_with_two_elites():
    archive = UnstructuredArchive(
        solution_dim=2,
        measure_dim=2,
        k_neighbors=5,
        novelty_threshold=1.0,
    )
    archive.add_single([4.0, 4.0], 0.25, [-2.0, -2.0])  # Elite 2.
    archive.add_single([4.0, 4.0], 0.0, [2.0, 2.0])  # Elite 3.

    score = archive.cqd_score(
        iterations=1,
        # With this target point, Elite 1 has a normalized distance of 1, since
        # it is exactly at [-2, -2].
        #
        # Elite 2 has a normalized distance of 0, since it is exactly at [2, 2].
        target_points=np.array([[[2.0, 2.0]]]),
        penalties=2,  # Penalties of 0 and 1.
        obj_min=0.0,
        obj_max=1.0,
    ).mean
    # For theta=0, the score should be max(0.25 - 0 * 1.0, 0 - 0 * 0.0) = 0.25
    # For theta=1, the score should be max(0.25 - 1 * 1.0, 0 - 1 * 0.0) = 0.0
    assert np.isclose(score, 0.25 + 0)
