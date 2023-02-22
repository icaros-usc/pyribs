"""Tests for ArchiveBase."""
import numpy as np
import pytest

from ribs.archives import GridArchive

from .conftest import ARCHIVE_NAMES, get_archive_data

# pylint: disable = redefined-outer-name

#
# Test the dtypes of all classes.
#


@pytest.mark.parametrize("name", ARCHIVE_NAMES)
@pytest.mark.parametrize("dtype", [("f", np.float32), ("d", np.float64)],
                         ids=["f", "d"])
def test_str_dtype_float(name, dtype):
    str_dtype, np_dtype = dtype
    archive = get_archive_data(name, str_dtype).archive
    assert archive.dtype == np_dtype


def test_invalid_dtype():
    with pytest.raises(ValueError):
        GridArchive(solution_dim=0,
                    dims=[20, 20],
                    ranges=[(-1, 1)] * 2,
                    dtype=np.int32)


#
# Tests for iteration -- only GridArchive for simplicity.
#


def test_iteration():
    data = get_archive_data("GridArchive")
    for elite in data.archive_with_elite:
        assert np.isclose(elite.solution, data.solution).all()
        assert np.isclose(elite.objective, data.objective)
        assert np.isclose(elite.measures, data.measures).all()
        assert elite.index == data.archive_with_elite.grid_to_int_index(
            [data.grid_indices])[0]
        assert elite.metadata == data.metadata


def test_add_during_iteration(add_mode):
    # Even with just one entry, adding during iteration should still raise an
    # error, just like it does in set.
    data = get_archive_data("GridArchive")
    with pytest.raises(RuntimeError):
        for _ in data.archive_with_elite:
            if add_mode == "single":
                data.archive_with_elite.add_single(data.solution,
                                                   data.objective + 1,
                                                   data.measures)
            else:
                data.archive_with_elite.add([data.solution],
                                            [data.objective + 1],
                                            [data.measures])


def test_clear_during_iteration():
    data = get_archive_data("GridArchive")
    with pytest.raises(RuntimeError):
        for _ in data.archive_with_elite:
            data.archive_with_elite.clear()


def test_clear_and_add_during_iteration(add_mode):
    data = get_archive_data("GridArchive")
    with pytest.raises(RuntimeError):
        for _ in data.archive_with_elite:
            data.archive_with_elite.clear()
            if add_mode == "single":
                data.archive_with_elite.add_single(data.solution,
                                                   data.objective + 1,
                                                   data.measures)
            else:
                data.archive_with_elite.add([data.solution],
                                            [data.objective + 1],
                                            [data.measures])


#
# Statistics and best elite tests -- just GridArchive for simplicity.
#


@pytest.mark.parametrize("dtype", [np.float64, np.float32],
                         ids=["float64", "float32"])
def test_stats_dtype(dtype):
    data = get_archive_data("GridArchive", dtype=dtype)
    assert isinstance(data.archive_with_elite.stats.num_elites, int)
    assert isinstance(data.archive_with_elite.stats.coverage, dtype)
    assert isinstance(data.archive_with_elite.stats.qd_score, dtype)
    assert isinstance(data.archive_with_elite.stats.norm_qd_score, dtype)
    assert isinstance(data.archive_with_elite.stats.obj_max, dtype)
    assert isinstance(data.archive_with_elite.stats.obj_mean, dtype)


@pytest.mark.parametrize("qd_score_offset", [0.0, -1.0])
def test_stats_multiple_add(add_mode, qd_score_offset):
    archive = GridArchive(solution_dim=3,
                          dims=[10, 20],
                          ranges=[(-1, 1), (-2, 2)],
                          qd_score_offset=qd_score_offset)
    if add_mode == "single":
        archive.add_single([1, 2, 3], 1.0, [0, 0])
        archive.add_single([1, 2, 3], 2.0, [0.25, 0.25])
        archive.add_single([1, 2, 3], 3.0, [-0.25, -0.25])
    else:
        solution_batch = [[1, 2, 3]] * 3
        objective_batch = [1.0, 2.0, 3.0]
        measures_batch = [[0, 0], [0.25, 0.25], [-0.25, -0.25]]
        archive.add(solution_batch, objective_batch, measures_batch)

    assert archive.stats.num_elites == 3
    assert np.isclose(archive.stats.coverage, 3 / 200)
    if qd_score_offset == 0.0:
        assert np.isclose(archive.stats.qd_score, 6.0)
        assert np.isclose(archive.stats.norm_qd_score, 6.0 / 200)
    else:
        # -1 is subtracted from every objective.
        assert np.isclose(archive.stats.qd_score, 9.0)
        assert np.isclose(archive.stats.norm_qd_score, 9.0 / 200)
    assert np.isclose(archive.stats.obj_max, 3.0)
    assert np.isclose(archive.stats.obj_mean, 2.0)


@pytest.mark.parametrize("qd_score_offset", [0.0, -1.0])
def test_stats_add_and_overwrite(add_mode, qd_score_offset):
    archive = GridArchive(solution_dim=3,
                          dims=[10, 20],
                          ranges=[(-1, 1), (-2, 2)],
                          qd_score_offset=qd_score_offset)
    if add_mode == "single":
        archive.add_single([1, 2, 3], 1.0, [0, 0])
        archive.add_single([1, 2, 3], 2.0, [0.25, 0.25])
        archive.add_single([1, 2, 3], 3.0, [-0.25, -0.25])
        archive.add_single([1, 2, 3], 5.0,
                           [0.25, 0.25])  # Overwrites the second add.
    else:
        solution_batch = [[1, 2, 3]] * 4
        objective_batch = [1.0, 2.0, 3.0, 5.0]
        measures_batch = [[0, 0], [0.25, 0.25], [-0.25, -0.25], [0.25, 0.25]]
        archive.add(solution_batch, objective_batch, measures_batch)

    assert archive.stats.num_elites == 3
    assert np.isclose(archive.stats.coverage, 3 / 200)
    if qd_score_offset == 0.0:
        assert np.isclose(archive.stats.qd_score, 9.0)
        assert np.isclose(archive.stats.norm_qd_score, 9.0 / 200)
    else:
        # -1 is subtracted from every objective.
        assert np.isclose(archive.stats.qd_score, 12.0)
        assert np.isclose(archive.stats.norm_qd_score, 12.0 / 200)
    assert np.isclose(archive.stats.obj_max, 5.0)
    assert np.isclose(archive.stats.obj_mean, 3.0)


def test_best_elite(add_mode):
    archive = GridArchive(solution_dim=3,
                          dims=[10, 20],
                          ranges=[(-1, 1), (-2, 2)])

    # Initial elite is None.
    assert archive.best_elite is None

    # Add an elite.
    if add_mode == "single":
        archive.add_single([1, 2, 3], 1.0, [0, 0])
    else:
        archive.add([[1, 2, 3]], [1.0], [[0, 0]])

    assert np.isclose(archive.best_elite.solution, [1, 2, 3]).all()
    assert np.isclose(archive.best_elite.objective, 1.0)
    assert np.isclose(archive.best_elite.measures, [0, 0]).all()
    assert np.isclose(archive.stats.obj_max, 1.0)

    # Add an elite into the same cell as the previous elite -- best_elite should
    # now be overwritten.
    if add_mode == "single":
        archive.add_single([4, 5, 6], 2.0, [0, 0])
    else:
        archive.add([[4, 5, 6]], [2.0], [[0, 0]])

    assert np.isclose(archive.best_elite.solution, [4, 5, 6]).all()
    assert np.isclose(archive.best_elite.objective, 2.0).all()
    assert np.isclose(archive.best_elite.measures, [0, 0]).all()
    assert np.isclose(archive.stats.obj_max, 2.0)


def test_best_elite_with_threshold(add_mode):
    archive = GridArchive(solution_dim=3,
                          dims=[10, 20],
                          ranges=[(-1, 1), (-2, 2)],
                          learning_rate=0.1,
                          threshold_min=0.0)

    # Add an elite.
    if add_mode == "single":
        archive.add_single([1, 2, 3], 1.0, [0, 0])
    else:
        archive.add([[1, 2, 3]], [1.0], [[0, 0]])

    # Threshold should now be 0.1 * 1 + (1 - 0.1) * 0.

    assert np.isclose(archive.best_elite.solution, [1, 2, 3]).all()
    assert np.isclose(archive.best_elite.objective, 1.0).all()
    assert np.isclose(archive.best_elite.measures, [0, 0]).all()
    assert np.isclose(archive.stats.obj_max, 1.0)

    # Add an elite with lower objective value than best elite but higher
    # objective value than threshold.
    if add_mode == "single":
        archive.add_single([4, 5, 6], 0.2, [0, 0])
    else:
        archive.add([[4, 5, 6]], [0.2], [[0, 0]])

    # Best elite remains the same even though this is a non-elitist archive and
    # the best elite is no longer in the archive.
    assert np.isclose(archive.best_elite.solution, [1, 2, 3]).all()
    assert np.isclose(archive.best_elite.objective, 1.0)
    assert np.isclose(archive.best_elite.measures, [0, 0]).all()
    assert np.isclose(archive.stats.obj_max, 1.0)


#
# index_of() and index_of_single() tests
#


def test_index_of_wrong_shape(data):
    with pytest.raises(ValueError):
        data.archive.index_of([data.measures[:-1]])


# Only test on GridArchive.
def test_index_of_single():
    data = get_archive_data("GridArchive")
    assert data.archive.index_of_single(data.measures) == data.int_index


def test_index_of_single_wrong_shape(data):
    with pytest.raises(ValueError):
        data.archive.retrieve_single(data.measures[:-1])


#
# General tests -- should work for all archive classes.
#


@pytest.fixture(params=ARCHIVE_NAMES)
def data(request):
    """Provides data for testing all kinds of archives."""
    return get_archive_data(request.param)


def test_length(data):
    assert len(data.archive) == 0
    assert len(data.archive_with_elite) == 1


def test_new_archive_is_empty(data):
    assert data.archive.empty


def test_archive_with_elite_is_not_empty(data):
    assert not data.archive_with_elite.empty


def test_archive_is_empty_after_clear(data):
    data.archive_with_elite.clear()
    assert data.archive_with_elite.empty


def test_cells_correct(data):
    assert data.archive.cells == data.cells


def test_measure_dim_correct(data):
    assert data.archive.measure_dim == len(data.measures)


def test_solution_dim_correct(data):
    assert data.archive.solution_dim == len(data.solution)


def test_learning_rate_correct(data):
    assert data.archive.learning_rate == 1.0  # Default value.


def test_threshold_min_correct(data):
    assert data.archive.threshold_min == -np.inf  # Default value.


def test_qd_score_offset_correct(data):
    assert data.archive.qd_score_offset == 0.0  # Default value.


def test_basic_stats(data):
    assert data.archive.stats.num_elites == 0
    assert data.archive.stats.coverage == 0.0
    assert data.archive.stats.qd_score == 0.0
    assert data.archive.stats.norm_qd_score == 0.0
    assert data.archive.stats.obj_max is None
    assert data.archive.stats.obj_mean is None

    assert data.archive_with_elite.stats.num_elites == 1
    assert data.archive_with_elite.stats.coverage == 1 / data.cells
    assert data.archive_with_elite.stats.qd_score == data.objective
    assert (data.archive_with_elite.stats.norm_qd_score == data.objective /
            data.cells)
    assert data.archive_with_elite.stats.obj_max == data.objective
    assert data.archive_with_elite.stats.obj_mean == data.objective


def test_retrieve_gets_correct_elite(data):
    elite_batch = data.archive_with_elite.retrieve([data.measures])
    assert np.all(elite_batch.solution_batch[0] == data.solution)
    assert elite_batch.objective_batch[0] == data.objective
    assert np.all(elite_batch.measures_batch[0] == data.measures)
    # Avoid checking elite_batch.idx since the meaning varies by archive.
    assert elite_batch.metadata_batch[0] == data.metadata


def test_retrieve_empty_values(data):
    elite_batch = data.archive.retrieve([data.measures])
    assert np.all(np.isnan(elite_batch.solution_batch[0]))
    assert np.isnan(elite_batch.objective_batch)
    assert np.all(np.isnan(elite_batch.measures_batch[0]))
    assert elite_batch.index_batch[0] == -1
    assert elite_batch.metadata_batch[0] is None


def test_retrieve_wrong_shape(data):
    with pytest.raises(ValueError):
        data.archive.retrieve([data.measures[:-1]])


def test_retrieve_single_gets_correct_elite(data):
    elite = data.archive_with_elite.retrieve_single(data.measures)
    assert np.all(elite.solution == data.solution)
    assert elite.objective == data.objective
    assert np.all(elite.measures == data.measures)
    # Avoid checking elite.idx since the meaning varies by archive.
    assert elite.metadata == data.metadata


def test_retrieve_single_empty_values(data):
    elite = data.archive.retrieve_single(data.measures)
    assert np.all(np.isnan(elite.solution))
    assert np.isnan(elite.objective)
    assert np.all(np.isnan(elite.measures))
    assert elite.index == -1
    assert elite.metadata is None


def test_retrieve_single_wrong_shape(data):
    with pytest.raises(ValueError):
        data.archive.retrieve_single(data.measures[:-1])


def test_sample_elites_gets_single_elite(data):
    elite_batch = data.archive_with_elite.sample_elites(2)
    assert np.all(elite_batch.solution_batch == data.solution)
    assert np.all(elite_batch.objective_batch == data.objective)
    assert np.all(elite_batch.measures_batch == data.measures)
    # Avoid checking elite.idx since the meaning varies by archive.
    assert np.all(elite_batch.metadata_batch == data.metadata)


def test_sample_elites_fails_when_empty(data):
    with pytest.raises(IndexError):
        data.archive.sample_elites(1)


@pytest.mark.parametrize("name", ARCHIVE_NAMES)
@pytest.mark.parametrize("with_elite", [True, False], ids=["nonempty", "empty"])
@pytest.mark.parametrize("include_solutions", [True, False],
                         ids=["solutions", "no_solutions"])
@pytest.mark.parametrize("include_metadata", [True, False],
                         ids=["metadata", "no_metadata"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32],
                         ids=["float64", "float32"])
def test_as_pandas(name, with_elite, include_solutions, include_metadata,
                   dtype):
    data = get_archive_data(name, dtype)

    # Set up expected columns and data types.
    measure_cols = [f"measure_{i}" for i in range(len(data.measures))]
    expected_cols = ["index"] + measure_cols + ["objective"]
    expected_dtypes = [np.int32, *[dtype for _ in measure_cols], dtype]
    if include_solutions:
        solution_cols = [f"solution_{i}" for i in range(len(data.solution))]
        expected_cols += solution_cols
        expected_dtypes += [dtype for _ in solution_cols]
    if include_metadata:
        expected_cols.append("metadata")
        expected_dtypes.append(object)

    # Retrieve the dataframe.
    if with_elite:
        df = data.archive_with_elite.as_pandas(
            include_solutions=include_solutions,
            include_metadata=include_metadata)
    else:
        df = data.archive.as_pandas(include_solutions=include_solutions,
                                    include_metadata=include_metadata)

    # Check columns and data types.
    assert (df.columns == expected_cols).all()
    assert (df.dtypes == expected_dtypes).all()

    if with_elite:
        if name.startswith("CVTArchive-"):
            # For CVTArchive, we check the centroid because the index can vary.
            index = df.loc[0, "index"]
            assert (data.archive_with_elite.centroids[index] == data.centroid
                   ).all()
        else:
            # Other archives have expected grid indices.
            assert df.loc[0, "index"] == data.archive.grid_to_int_index(
                [data.grid_indices])[0]

        expected_data = [*data.measures, data.objective]
        if include_solutions:
            expected_data += list(data.solution)
        if include_metadata:
            expected_data.append(data.metadata)
        assert (df.loc[0, "measure_0":] == expected_data).all()
