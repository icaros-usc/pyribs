"""Tests for the ProximityArchive."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ribs.archives import AddStatus, ProximityArchive
from tests.archives.conftest import get_archive_data

# pylint: disable = redefined-outer-name


@pytest.fixture
def data():
    """Data for grid archive tests."""
    return get_archive_data("ProximityArchive")


def assert_archive_elite(archive, solution, objective, measures):
    """Asserts that the archive has one specific elite."""
    assert len(archive) == 1
    elite = list(archive)[0]
    assert np.isclose(elite["solution"], solution).all()
    assert np.isclose(elite["objective"], objective).all()
    assert np.isclose(elite["measures"], measures).all()


def assert_archive_elites(
    archive,
    batch_size,
    solution_batch=None,
    objective_batch=None,
    measures_batch=None,
    metadata_batch=None,
):
    """Asserts that the archive contains a batch of elites.

    Any of the batch items may be excluded by setting to None.
    """
    data = archive.data()

    # Check the number of solutions.
    assert len(data["index"]) == batch_size

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

            # Used for testing custom fields.
            metadata_match = (metadata_batch is None or
                              data["metadata"][j] == metadata_batch[i])

            if (solution_match and objective_match and measures_match and
                    metadata_match):
                archive_covered[j] = True

    assert np.all(archive_covered)


def test_properties_are_correct(data):
    # Without elite.
    assert data.archive.capacity == data.capacity
    assert len(data.archive) == 0
    assert data.archive.cells == 0
    assert data.archive.empty
    assert data.archive.k_neighbors == data.k_neighbors
    assert data.archive.novelty_threshold == data.novelty_threshold
    # Undefined when there are no solutions.
    with pytest.raises(RuntimeError):
        data.archive.lower_bounds  # pylint: disable = pointless-statement
    with pytest.raises(RuntimeError):
        data.archive.upper_bounds  # pylint: disable = pointless-statement

    # With elite.
    assert data.archive_with_elite.capacity == data.capacity
    assert len(data.archive_with_elite) == 1
    assert data.archive_with_elite.cells == 1
    assert not data.archive_with_elite.empty
    assert data.archive_with_elite.k_neighbors == data.k_neighbors
    assert data.archive_with_elite.novelty_threshold == data.novelty_threshold
    assert_allclose(data.archive_with_elite.lower_bounds, data.measures)
    assert_allclose(data.archive_with_elite.upper_bounds, data.measures)


def test_bounds(data):
    data.archive.add(
        solution=[[1, 2, 3]] * 2,
        objective=None,
        measures=[[0, 0], [2, 2]],
    )

    # Boundaries should reflect min/max measures.
    assert np.all(data.archive.lower_bounds == [0, 0])
    assert np.all(data.archive.upper_bounds == [2, 2])

    data.archive.add(
        solution=[[1, 2, 3]] * 2,
        objective=None,
        measures=[[-2, -2], [4, 4]],
    )

    # Boundaries should update to the new min/max measures.
    assert np.all(data.archive.lower_bounds == [-2, -2])
    assert np.all(data.archive.upper_bounds == [4, 4])


def test_resizing_with_add_one_at_a_time():
    archive = ProximityArchive(
        solution_dim=3,
        measure_dim=2,
        k_neighbors=5,
        novelty_threshold=1.0,
        initial_capacity=1,
    )

    for i in range(1, 129):
        archive.add_single([1, 2, 3], None, [0, i])

        if i in [2, 4, 8, 16, 32, 64, 128]:
            assert archive.capacity == i


def test_resizing_with_add_multiple():
    archive = ProximityArchive(
        solution_dim=3,
        measure_dim=2,
        k_neighbors=5,
        novelty_threshold=1.0,
        initial_capacity=2,
    )

    archive.add(
        [[1, 2, 3]] * 5,
        None,
        np.stack((np.arange(5), np.arange(5)), axis=1),  # [0, 0], [1, 1], ...
    )

    assert len(archive) == 5
    assert archive.capacity == 8

    archive.add(
        [[1, 2, 3]] * 27,
        None,
        np.stack((np.arange(5, 32), np.arange(5, 32)), axis=1),
    )

    assert len(archive) == 32
    assert archive.capacity == 32


@pytest.mark.parametrize("use_list", [True, False], ids=["list", "ndarray"])
def test_add_single(data, use_list, add_mode):
    solution = data.solution
    measures = data.measures

    if use_list:
        solution = list(data.solution)
        measures = list(data.measures)

    if add_mode == "single":
        add_info = data.archive.add_single(solution, None, measures)
    else:
        add_info = data.archive.add([solution], None, [measures])

    # Objective should default to 0.0.
    assert_archive_elite(data.archive, data.solution, 0.0, data.measures)
    assert add_info["status"] == AddStatus.NEW
    assert add_info["novelty"] == np.inf


def test_add_single_after_clear(data):
    """After clearing, we should still get the same status and value when adding
    to the archive.

    https://github.com/icaros-usc/pyribs/pull/260
    """
    add_info = data.archive.add_single(data.solution, None, data.measures)

    assert add_info["status"] == 2
    assert add_info["novelty"] == np.inf

    data.archive.clear()

    add_info = data.archive.add_single(data.solution, None, data.measures)

    assert add_info["status"] == 2
    assert add_info["novelty"] == np.inf


def test_add_novel_solution():
    archive = ProximityArchive(
        solution_dim=3,
        measure_dim=2,
        k_neighbors=1,
        novelty_threshold=1.0,
        initial_capacity=1,
    )

    archive.add_single([1, 2, 3], 1.0, [0, 0])

    # Should be added since novelty threshold is 1.0.
    add_info = archive.add_single([1, 2, 3], None, [1, 0])

    assert_archive_elites(archive, 2, measures_batch=[[0, 0], [1, 0]])

    assert add_info["status"] == 2
    assert add_info["novelty"] == 1.0


def test_add_non_novel_solution():
    archive = ProximityArchive(
        solution_dim=3,
        measure_dim=2,
        k_neighbors=1,
        novelty_threshold=1.0,
        initial_capacity=1,
    )

    archive.add_single([1, 2, 3], None, [0, 0])

    # Should not be added since threshold is 1.0.
    add_info = archive.add_single([1, 2, 3], None, [0.5, 0])

    assert_archive_elites(archive, 1, measures_batch=[[0, 0]])

    assert add_info["status"] == 0
    assert_allclose(add_info["novelty"], 0.5)


@pytest.mark.parametrize("point", [[0.1, 0], [0.5, 0], [0.9, 0], [-0.1, 0.1]])
def test_add_with_multiple_neighbors(point):
    archive = ProximityArchive(
        solution_dim=3,
        measure_dim=2,
        k_neighbors=2,
        novelty_threshold=1.0,
        initial_capacity=1,
    )

    archive.add_single([1, 2, 3], None, [0, 0])
    archive.add_single([1, 2, 3], None, [2, 0])

    # Should be added since threshold is 1.0 and the point is in between [0, 0]
    # and [2, 0], so its average distance is always at least 1.0.
    add_info = archive.add_single([1, 2, 3], None, point)

    assert_archive_elites(archive, 3, measures_batch=[[0, 0], [2, 0], point])
    assert add_info["status"] == 2
    assert_allclose(
        add_info["novelty"],
        np.mean(np.linalg.norm(np.array(point)[None] - [[0, 0], [2, 0]],
                               axis=1)),
    )


def test_add_single_wrong_shapes(data):
    with pytest.raises(ValueError):
        data.archive.add_single(
            solution=[1, 1],  # 2D instead of 3D solution.
            objective=None,
            measures=[0, 0],
        )
    with pytest.raises(ValueError):
        data.archive.add_single(
            solution=[0, 0, 0],
            objective=None,
            measures=[1, 1, 1],  # 3D instead of 2D measures.
        )


def test_add_batch_all_new():
    archive = ProximityArchive(
        solution_dim=3,
        measure_dim=2,
        k_neighbors=2,
        novelty_threshold=1.0,
        initial_capacity=1,
    )

    # Initial points.
    archive.add([[1, 2, 3]] * 2, None, [[0, 0], [1, 0]])

    add_info = archive.add(
        solution=[[1, 2, 3]] * 3,
        objective=None,
        measures=[[-1, 0], [0, 1], [2, 0]],
    )

    assert_archive_elites(
        archive=archive,
        batch_size=5,
        solution_batch=[[1, 2, 3]] * 5,
        measures_batch=[[0, 0], [1, 0], [-1, 0], [0, 1], [2, 0]],
    )

    assert (add_info["status"] == 2).all()
    assert_allclose(add_info["novelty"], [
        np.mean([1, 2]),
        np.mean([1, np.sqrt(2)]),
        np.mean([2, 1]),
    ])


def test_add_batch_none_inserted():
    archive = ProximityArchive(
        solution_dim=3,
        measure_dim=2,
        k_neighbors=2,
        novelty_threshold=1.0,
        initial_capacity=1,
    )

    # Initial points.
    archive.add([[1, 2, 3]] * 2, None, [[0, 0], [1, 0]])

    add_info = archive.add(
        solution=[[1, 2, 3]] * 2,
        objective=None,
        measures=[[0.4, 0], [0, 0.1]],
    )

    assert_archive_elites(
        archive=archive,
        batch_size=2,
        solution_batch=[[1, 2, 3]] * 2,
        measures_batch=[[0, 0], [1, 0]],
    )

    assert (add_info["status"] == 0).all()
    assert_allclose(add_info["novelty"], [
        np.mean([0.4, 0.6]),
        np.mean([0.1, np.sqrt(0.1**2 + 1)]),
    ])


def test_add_batch_mixed_statuses():
    archive = ProximityArchive(
        solution_dim=3,
        measure_dim=2,
        k_neighbors=2,
        novelty_threshold=1.0,
        initial_capacity=1,
    )

    # Initial points.
    archive.add([[1, 2, 3]] * 2, None, [[0, 0], [1, 0]])

    add_info = archive.add(
        solution=[[1, 2, 3]] * 5,
        objective=None,
        measures=[[-1, 0], [0.4, 0], [0, 1], [0, 0.1], [2, 0]],
    )

    assert_archive_elites(
        archive=archive,
        batch_size=5,
        solution_batch=[[1, 2, 3]] * 5,
        measures_batch=[[0, 0], [1, 0], [-1, 0], [0, 1], [2, 0]],
    )

    assert (add_info["status"] == [2, 0, 2, 0, 2]).all()
    assert_allclose(add_info["novelty"], [
        np.mean([1, 2]),
        np.mean([0.4, 0.6]),
        np.mean([1, np.sqrt(2)]),
        np.mean([0.1, np.sqrt(0.1**2 + 1)]),
        np.mean([2, 1]),
    ])


def test_add_batch_wrong_shapes(data):
    with pytest.raises(ValueError):
        data.archive.add(
            solution=[[1, 1]],  # 2D instead of 3D solution.
            objective=None,
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
            objective=None,
            measures=[[1, 1, 1]],  # 3D instead of 2D measures.
        )


def test_add_batch_zero_length(data):
    """Nothing should happen when adding a batch with length 0."""
    add_info = data.archive.add(
        solution=np.ones((0, 3)),
        objective=None,
        measures=np.ones((0, 2)),
    )

    assert len(add_info["status"]) == 0
    assert len(add_info["novelty"]) == 0
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
            objective=None,
            measures=[[1, 1, 1], [1, 1, 1]],  # 2 measures.
        )


def test_retrieve():
    """Indirectly tests that index_of is retrieving the nearest solutions in
    measure space."""
    archive = ProximityArchive(
        solution_dim=0,
        measure_dim=2,
        k_neighbors=5,
        novelty_threshold=1.0,
    )

    # Add four measures in a square.
    archive.add(
        solution=[[]] * 4,
        objective=None,
        measures=[[0, 0], [1, 0], [1, 1], [0, 1]],
    )

    occupied, data = archive.retrieve([[0.1, 0.1], [10.0, 0], [1.5, 1.5],
                                       [0, 0.55]])

    assert np.all(occupied)
    assert_allclose(data["measures"], [[0, 0], [1, 0], [1, 1], [0, 1]])


def test_index_of_errors_on_empty(data):
    with pytest.raises(RuntimeError):
        data.archive.index_of([[1, 2]])


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


def test_cqd_score_with_max_dist():
    archive = ProximityArchive(
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
