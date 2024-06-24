"""Tests for the UnstructuredArchive."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ribs.archives import AddStatus, UnstructuredArchive
from tests.archives.conftest import get_archive_data

# pylint: disable = redefined-outer-name


@pytest.fixture
def data():
    """Data for grid archive tests."""
    return get_archive_data("UnstructuredArchive")


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
    assert not data.archive.compare_to_batch
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
    assert not data.archive_with_elite.compare_to_batch
    assert not data.archive_with_elite.empty
    assert data.archive_with_elite.k_neighbors == data.k_neighbors
    assert data.archive_with_elite.novelty_threshold == data.novelty_threshold
    assert_allclose(data.archive_with_elite.lower_bounds, data.measures)
    assert_allclose(data.archive_with_elite.upper_bounds, data.measures)


def test_bounds(data):
    data.archive.add(
        solution=[[1, 2, 3]] * 2,
        objective=[0, 0],
        measures=[[0, 0], [2, 2]],
    )

    # Boundaries should reflect min/max measures.
    assert np.all(data.archive.lower_bounds == [0, 0])
    assert np.all(data.archive.upper_bounds == [2, 2])

    data.archive.add(
        solution=[[1, 2, 3]] * 2,
        objective=[0] * 2,
        measures=[[-2, -2], [4, 4]],
    )

    # Boundaries should update to the new min/max measures.
    assert np.all(data.archive.lower_bounds == [-2, -2])
    assert np.all(data.archive.upper_bounds == [4, 4])


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
                         data.measures)


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


def test_resizing_with_add_one_at_a_time():
    archive = UnstructuredArchive(
        solution_dim=3,
        measure_dim=2,
        k_neighbors=5,
        novelty_threshold=1.0,
        initial_capacity=1,
    )

    for i in range(1, 129):
        archive.add_single([1, 2, 3], 1.0, [0, i])

        if i in [2, 4, 8, 16, 32, 64, 128]:
            assert archive.capacity == i


def test_resizing_with_add_multiple():
    archive = UnstructuredArchive(
        solution_dim=3,
        measure_dim=2,
        k_neighbors=5,
        novelty_threshold=1.0,
        initial_capacity=2,
    )

    archive.add(
        [[1, 2, 3]] * 5,
        np.ones(5),
        np.stack((np.arange(5), np.arange(5)), axis=1),  # [0, 0], [1, 1], ...
    )

    assert len(archive) == 5
    assert archive.capacity == 8

    archive.add(
        [[1, 2, 3]] * 27,
        np.ones(27),
        np.stack((np.arange(5, 32), np.arange(5, 32)), axis=1),
    )

    assert len(archive) == 32
    assert archive.capacity == 32


# TODO: Test addition
# TODO: Test compare_to_batch

#  def test_add_single_wrong_shapes(data):
#      with pytest.raises(ValueError):
#          data.archive.add_single(
#              solution=[1, 1],  # 2D instead of 3D solution.
#              objective=0,
#              measures=[0, 0],
#          )
#      with pytest.raises(ValueError):
#          data.archive.add_single(
#              solution=[0, 0, 0],
#              objective=0,
#              measures=[1, 1, 1],  # 3D instead of 2D measures.
#          )

#  def test_add_batch_all_new(data):
#      add_info = data.archive.add(
#          # 4 solutions of arbitrary value.
#          solution=[[1, 2, 3]] * 5,
#          # The first two solutions end up in separate cells, and the next two end
#          # up in the same cell.
#          objective=[0, 0, 1, 0, 1],
#          measures=[[0, 0], [0.25, 0.25], [0.25, 0.25], [0.5, 0.5], [0.5, 0.5]],
#      )
#      assert (add_info["status"] == 2).all()
#      assert np.isclose(add_info["value"], [0, 0, 1, 0, 1]).all()

#      assert_archive_elites(
#          archive=data.archive,
#          batch_size=3,
#          solution_batch=[[1, 2, 3]] * 3,
#          objective_batch=[0, 1, 1],
#          measures_batch=[[0, 0], [0.25, 0.25], [0.5, 0.5]],
#          grid_indices_batch=[0, 1, 2],
#      )

#  def test_add_batch_none_inserted(data):
#      add_info = data.archive_with_elite.add(
#          solution=[[1, 2, 3]] * 4,
#          objective=[data.objective - 1 for _ in range(4)],
#          measures=[data.measures for _ in range(4)],
#      )

#      # All solutions were inserted into the same cell as the elite already in the
#      # archive and had objective value 1 less.
#      assert (add_info["status"] == 0).all()
#      assert np.isclose(add_info["value"], -1.0).all()

#      assert_archive_elites(
#          archive=data.archive_with_elite,
#          batch_size=1,
#          solution_batch=[data.solution],
#          objective_batch=[data.objective],
#          measures_batch=[data.measures],
#          grid_indices_batch=[data.grid_indices],
#      )

#  def test_add_batch_with_improvement(data):
#      add_info = data.archive_with_elite.add(
#          solution=[[1, 2, 3]] * 4,
#          objective=[data.objective + 1 for _ in range(4)],
#          measures=[data.measures for _ in range(4)],
#      )

#      # All solutions were inserted into the same cell as the elite already in the
#      # archive and had objective value 1 greater.
#      assert (add_info["status"] == 1).all()
#      assert np.isclose(add_info["value"], 1.0).all()

#      assert_archive_elites(
#          archive=data.archive_with_elite,
#          batch_size=1,
#          solution_batch=[[1, 2, 3]],
#          objective_batch=[data.objective + 1],
#          measures_batch=[data.measures],
#          grid_indices_batch=[data.grid_indices],
#      )

#  def test_add_batch_mixed_statuses(data):
#      add_info = data.archive_with_elite.add(
#          solution=[[1, 2, 3]] * 6,
#          objective=[
#              # Not added.
#              data.objective - 1.0,
#              # Not added.
#              data.objective - 2.0,
#              # Improve but not added.
#              data.objective + 1.0,
#              # Improve and added since it has higher objective.
#              data.objective + 2.0,
#              # New but not added.
#              1.0,
#              # New and added.
#              2.0,
#          ],
#          measures=[
#              data.measures,
#              data.measures,
#              data.measures,
#              data.measures,
#              [2, 2],
#              [2, 2],
#          ],
#      )
#      assert (add_info["status"] == [0, 0, 1, 1, 2, 2]).all()
#      assert np.isclose(add_info["value"], [-1, -2, 1, 2, 1, 2]).all()

#      assert_archive_elites(
#          archive=data.archive_with_elite,
#          batch_size=2,
#          solution_batch=[[1, 2, 3]] * 2,
#          objective_batch=[data.objective + 2.0, 2.0],
#          measures_batch=[data.measures, [2, 2]],
#          grid_indices_batch=[data.grid_indices, [1]],
#      )


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


def test_retrieve():
    """Indirectly tests that index_of is retrieving the nearest solutions in
    measure space."""
    archive = UnstructuredArchive(
        solution_dim=0,
        measure_dim=2,
        k_neighbors=5,
        novelty_threshold=1.0,
    )

    # Add four measures in a square.
    archive.add(
        solution=[[]] * 4,
        objective=[0, 0, 0, 0],
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


#  def test_cqd_score_detects_wrong_shapes(data):
#      with pytest.raises(ValueError):
#          data.archive.cqd_score(
#              iterations=1,
#              target_points=np.array([1.0]),  # Should be 3D.
#              penalties=2,
#              obj_min=0.0,
#              obj_max=1.0,
#          )

#      with pytest.raises(ValueError):
#          data.archive.cqd_score(
#              iterations=1,
#              target_points=3,
#              penalties=[[1.0, 1.0]],  # Should be 1D.
#              obj_min=0.0,
#              obj_max=1.0,
#          )

#  def test_cqd_score_with_one_elite():
#      archive = UnstructuredArchive(
#          solution_dim=2,
#          measure_dim=2,
#          k_neighbors=5,
#          novelty_threshold=1.0,
#      )
#      archive.add_single([4.0, 4.0], 1.0, [0.0, 0.0])

#      score = archive.cqd_score(
#          iterations=1,
#          # With this target point, the solution above at [0, 0] has a normalized
#          # distance of 0.5, since it is halfway between the archive bounds of
#          # (-1, -1) and (1, 1).
#          target_points=np.array([[[1.0, 1.0]]]),
#          penalties=2,
#          obj_min=0.0,
#          obj_max=1.0,
#      ).mean

#      # For theta=0, the score should be 1.0 - 0 * 0.5 = 1.0
#      # For theta=1, the score should be 1.0 - 1 * 0.5 = 0.5
#      assert np.isclose(score, 1.0 + 0.5)

#  def test_cqd_score_with_max_dist():
#      archive = UnstructuredArchive(
#          solution_dim=2,
#          measure_dim=2,
#          k_neighbors=5,
#          novelty_threshold=1.0,
#      )
#      archive.add_single([4.0, 4.0], 0.5, [0.0, 1.0])

#      score = archive.cqd_score(
#          iterations=1,
#          # With this target point and dist_max, the solution above at [0, 1]
#          # has a normalized distance of 0.5, since it is one unit away.
#          target_points=np.array([[[1.0, 1.0]]]),
#          penalties=2,
#          obj_min=0.0,
#          obj_max=1.0,
#          dist_max=2.0,
#      ).mean

#      # For theta=0, the score should be 0.5 - 0 * 0.5 = 0.5
#      # For theta=1, the score should be 0.5 - 1 * 0.5 = 0.0
#      assert np.isclose(score, 0.5 + 0.0)

#  def test_cqd_score_l1_norm():
#      archive = UnstructuredArchive(
#          solution_dim=2,
#          measure_dim=2,
#          k_neighbors=5,
#          novelty_threshold=1.0,
#      )
#      archive.add_single([4.0, 4.0], 0.5, [0.0, 0.0])

#      score = archive.cqd_score(
#          iterations=1,
#          # With this target point and dist_max, the solution above at [0, 0]
#          # has a normalized distance of 1.0, since it is two units away.
#          target_points=np.array([[[1.0, 1.0]]]),
#          penalties=2,
#          obj_min=0.0,
#          obj_max=1.0,
#          dist_max=2.0,
#          # L1 norm.
#          dist_ord=1,
#      ).mean

#      # For theta=0, the score should be 0.5 - 0 * 1.0 = 0.5
#      # For theta=1, the score should be 0.5 - 1 * 1.0 = -0.5
#      assert np.isclose(score, 0.5 + -0.5)

#  def test_cqd_score_full_output():
#      archive = UnstructuredArchive(
#          solution_dim=2,
#          measure_dim=2,
#          k_neighbors=5,
#          novelty_threshold=1.0,
#      )
#      archive.add_single([4.0, 4.0], 1.0, [0.0, 0.0])

#      result = archive.cqd_score(
#          iterations=5,
#          # With this target point, the solution above at [0, 0] has a normalized
#          # distance of 0.5, since it is halfway between the archive bounds of
#          # (-1, -1) and (1, 1).
#          target_points=np.array([
#              [[1.0, 1.0]],
#              [[1.0, 1.0]],
#              [[1.0, 1.0]],
#              [[-1.0, -1.0]],
#              [[-1.0, -1.0]],
#          ]),
#          penalties=2,
#          obj_min=0.0,
#          obj_max=1.0,
#      )

#      # For theta=0, the score should be 1.0 - 0 * 0.5 = 1.0
#      # For theta=1, the score should be 1.0 - 1 * 0.5 = 0.5
#      assert result.iterations == 5
#      assert np.isclose(result.mean, 1.0 + 0.5)
#      assert np.all(np.isclose(result.scores, 1.0 + 0.5))
#      assert np.all(
#          np.isclose(
#              result.target_points,
#              np.array([
#                  [[1.0, 1.0]],
#                  [[1.0, 1.0]],
#                  [[1.0, 1.0]],
#                  [[-1.0, -1.0]],
#                  [[-1.0, -1.0]],
#              ])))
#      assert np.all(np.isclose(result.penalties, [0.0, 1.0]))
#      assert np.isclose(result.obj_min, 0.0)
#      assert np.isclose(result.obj_max, 1.0)
#      # Distance from (-1,-1) to (1,1).
#      assert np.isclose(result.dist_max, 2 * np.sqrt(2))
#      assert result.dist_ord is None

#  def test_cqd_score_with_two_elites():
#      archive = UnstructuredArchive(
#          solution_dim=2,
#          measure_dim=2,
#          k_neighbors=5,
#          novelty_threshold=1.0,
#      )
#      archive.add_single([4.0, 4.0], 0.25, [-2.0, -2.0])  # Elite 2.
#      archive.add_single([4.0, 4.0], 0.0, [2.0, 2.0])  # Elite 3.

#      score = archive.cqd_score(
#          iterations=1,
#          # With this target point, Elite 1 has a normalized distance of 1, since
#          # it is exactly at [-2, -2].
#          #
#          # Elite 2 has a normalized distance of 0, since it is exactly at [2, 2].
#          target_points=np.array([[[2.0, 2.0]]]),
#          penalties=2,  # Penalties of 0 and 1.
#          obj_min=0.0,
#          obj_max=1.0,
#      ).mean
#      # For theta=0, the score should be max(0.25 - 0 * 1.0, 0 - 0 * 0.0) = 0.25
#      # For theta=1, the score should be max(0.25 - 1 * 1.0, 0 - 1 * 0.0) = 0.0
#      assert np.isclose(score, 0.25 + 0)
