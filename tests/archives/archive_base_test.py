"""Tests that are common across all archives."""

import numpy as np
import pytest

from ribs.archives import (
    CategoricalArchive,
    CVTArchive,
    GridArchive,
    ProximityArchive,
    SlidingBoundariesArchive,
)

from .conftest import ARCHIVE_NAMES, get_archive_data

MAE_ARCHIVES = (
    CategoricalArchive,
    CVTArchive,
    GridArchive,
)


#
# Test the dtypes of all classes.
#


@pytest.mark.parametrize("name", ARCHIVE_NAMES)
@pytest.mark.parametrize(
    "dtype", [("f", np.float32), ("d", np.float64)], ids=["f", "d"]
)
def test_str_dtype_float(name, dtype):
    str_dtype, np_dtype = dtype
    archive = get_archive_data(name, dtype=str_dtype).archive
    assert archive.dtypes["solution"] == np_dtype
    assert archive.dtypes["objective"] == np_dtype

    if isinstance(archive, CategoricalArchive):
        assert archive.dtypes["measures"] == np.object_
    else:
        assert archive.dtypes["measures"] == np_dtype

    if isinstance(archive, MAE_ARCHIVES):
        assert archive.dtypes["threshold"] == np_dtype

    assert archive.dtypes["index"] == np.int32


def test_default_dtypes():
    archive = GridArchive(
        solution_dim=3,
        dims=[10, 10],
        ranges=[(-1, 1), (-2, 2)],
    )

    assert archive.dtypes["solution"] == np.float64
    assert archive.dtypes["objective"] == np.float64
    assert archive.dtypes["measures"] == np.float64
    assert archive.dtypes["threshold"] == np.float64
    assert archive.dtypes["index"] == np.int32


def test_different_dtypes():
    archive = GridArchive(
        solution_dim=3,
        dims=[10, 10],
        ranges=[(-1, 1), (-2, 2)],
        solution_dtype=object,
        objective_dtype=np.float32,
        measures_dtype=np.float32,
    )

    assert archive.dtypes["solution"] == np.object_
    assert archive.dtypes["objective"] == np.float32
    assert archive.dtypes["measures"] == np.float32
    assert archive.dtypes["threshold"] == np.float32
    assert archive.dtypes["index"] == np.int32


def test_dtype_parameter():
    archive = GridArchive(
        solution_dim=3,
        dims=[10, 10],
        ranges=[(-1, 1), (-2, 2)],
        dtype=np.float32,
    )

    assert archive.dtypes["solution"] == np.float32
    assert archive.dtypes["objective"] == np.float32
    assert archive.dtypes["measures"] == np.float32
    assert archive.dtypes["threshold"] == np.float32
    assert archive.dtypes["index"] == np.int32


def test_simultaneous_dtypes():
    with pytest.raises(
        ValueError, match=r"dtype cannot be used at the same time as .*"
    ):
        GridArchive(
            solution_dim=3,
            dims=[10, 10],
            ranges=[(-1, 1), (-2, 2)],
            dtype=np.float32,
            solution_dtype=np.float32,
        )
    with pytest.raises(
        ValueError, match=r"dtype cannot be used at the same time as .*"
    ):
        GridArchive(
            solution_dim=3,
            dims=[10, 10],
            ranges=[(-1, 1), (-2, 2)],
            dtype=np.float32,
            objective_dtype=np.float32,
        )
    with pytest.raises(
        ValueError, match=r"dtype cannot be used at the same time as .*"
    ):
        GridArchive(
            solution_dim=3,
            dims=[10, 10],
            ranges=[(-1, 1), (-2, 2)],
            dtype=np.float32,
            measures_dtype=np.float32,
        )


def test_dtype_dict_deprecated():
    with pytest.raises(
        ValueError,
        match=r"Passing a dict as `dtype` is deprecated in pyribs 0\.9\.0\..*",
    ):
        GridArchive(
            solution_dim=3,
            dims=[10, 10],
            ranges=[(-1, 1), (-2, 2)],
            dtype={
                "solution": object,
                "objective": np.float32,
                "measures": np.float32,
            },
        )


#
# Tests for iteration -- only GridArchive for simplicity.
#


def test_iteration():
    data = get_archive_data("GridArchive")
    for elite in data.archive_with_elite:
        assert np.isclose(elite["solution"], data.solution).all()
        assert np.isclose(elite["objective"], data.objective)
        assert np.isclose(elite["measures"], data.measures).all()
        assert (
            elite["index"]
            == data.archive_with_elite.grid_to_int_index([data.grid_indices])[0]
        )


def test_add_during_iteration(add_mode):
    # Even with just one entry, adding during iteration should still raise an
    # error, just like it does in set.
    data = get_archive_data("GridArchive")
    with pytest.raises(RuntimeError):  # noqa: PT012
        for _ in data.archive_with_elite:
            if add_mode == "single":
                data.archive_with_elite.add_single(
                    data.solution, data.objective + 1, data.measures
                )
            else:
                data.archive_with_elite.add(
                    [data.solution], [data.objective + 1], [data.measures]
                )


def test_clear_during_iteration():
    data = get_archive_data("GridArchive")
    with pytest.raises(RuntimeError):  # noqa: PT012
        for _ in data.archive_with_elite:
            data.archive_with_elite.clear()


def test_clear_and_add_during_iteration(add_mode):
    data = get_archive_data("GridArchive")
    with pytest.raises(RuntimeError):  # noqa: PT012
        for _ in data.archive_with_elite:
            data.archive_with_elite.clear()
            if add_mode == "single":
                data.archive_with_elite.add_single(
                    data.solution, data.objective + 1, data.measures
                )
            else:
                data.archive_with_elite.add(
                    [data.solution], [data.objective + 1], [data.measures]
                )


#
# Statistics and best elite tests -- just GridArchive for simplicity.
#


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
def test_stats_dtype(dtype):
    data = get_archive_data("GridArchive", dtype=dtype)
    assert isinstance(data.archive_with_elite.stats.num_elites, int)
    assert data.archive_with_elite.stats.coverage.dtype == dtype
    assert data.archive_with_elite.stats.qd_score.dtype == dtype
    assert data.archive_with_elite.stats.norm_qd_score.dtype == dtype
    assert data.archive_with_elite.stats.obj_max.dtype == dtype
    assert data.archive_with_elite.stats.obj_mean.dtype == dtype


@pytest.mark.parametrize("qd_score_offset", [0.0, -1.0])
def test_stats_multiple_add(add_mode, qd_score_offset):
    archive = GridArchive(
        solution_dim=3,
        dims=[10, 20],
        ranges=[(-1, 1), (-2, 2)],
        qd_score_offset=qd_score_offset,
    )
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
    archive = GridArchive(
        solution_dim=3,
        dims=[10, 20],
        ranges=[(-1, 1), (-2, 2)],
        qd_score_offset=qd_score_offset,
    )
    if add_mode == "single":
        archive.add_single([1, 2, 3], 1.0, [0, 0])
        archive.add_single([1, 2, 3], 2.0, [0.25, 0.25])
        archive.add_single([1, 2, 3], 3.0, [-0.25, -0.25])
        archive.add_single([1, 2, 3], 5.0, [0.25, 0.25])  # Overwrites the second add.
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


@pytest.mark.parametrize("name", ARCHIVE_NAMES)
def test_best_elite_basic(name):
    data = get_archive_data(name)
    assert np.isclose(
        data.archive_with_elite.best_elite["solution"], data.solution
    ).all()
    assert np.isclose(
        data.archive_with_elite.best_elite["objective"], data.objective
    ).all()

    if isinstance(data.archive_with_elite, CategoricalArchive):
        assert np.all(data.archive_with_elite.best_elite["measures"] == data.measures)
    else:
        assert np.isclose(
            data.archive_with_elite.best_elite["measures"], data.measures
        ).all()


def test_best_elite_extended(add_mode):
    archive = GridArchive(solution_dim=3, dims=[10, 20], ranges=[(-1, 1), (-2, 2)])

    # Initial elite is None.
    assert archive.best_elite is None

    # Add an elite.
    if add_mode == "single":
        archive.add_single([1, 2, 3], 1.0, [0, 0])
    else:
        archive.add([[1, 2, 3]], [1.0], [[0, 0]])

    assert archive.best_elite.keys() == {
        "solution",
        "objective",
        "measures",
        "threshold",
        "index",
    }

    assert archive.best_elite["solution"].shape == (3,)
    assert archive.best_elite["objective"].shape == ()
    assert archive.best_elite["measures"].shape == (2,)
    assert archive.best_elite["threshold"].shape == ()
    assert archive.stats.obj_max.shape == ()  # ty: ignore[possibly-unbound-attribute]

    assert np.isclose(archive.best_elite["solution"], [1, 2, 3]).all()
    assert np.isclose(archive.best_elite["objective"], 1.0)
    assert np.isclose(archive.best_elite["measures"], [0, 0]).all()
    assert np.isclose(archive.best_elite["threshold"], 1.0).all()
    assert np.isclose(archive.stats.obj_max, 1.0)

    # Add an elite into the same cell as the previous elite -- best_elite should
    # now be overwritten.
    if add_mode == "single":
        archive.add_single([4, 5, 6], 2.0, [0, 0])
    else:
        archive.add([[4, 5, 6]], [2.0], [[0, 0]])

    assert np.isclose(archive.best_elite["solution"], [4, 5, 6]).all()
    assert np.isclose(archive.best_elite["objective"], 2.0).all()
    assert np.isclose(archive.best_elite["measures"], [0, 0]).all()
    assert np.isclose(archive.best_elite["threshold"], 2.0).all()
    assert np.isclose(archive.stats.obj_max, 2.0)


def test_best_elite_with_threshold(add_mode):
    archive = GridArchive(
        solution_dim=3,
        dims=[10, 20],
        ranges=[(-1, 1), (-2, 2)],
        learning_rate=0.1,
        threshold_min=0.0,
    )

    # Add an elite.
    if add_mode == "single":
        archive.add_single([1, 2, 3], 1.0, [0, 0])
    else:
        archive.add([[1, 2, 3]], [1.0], [[0, 0]])

    # Threshold should now be 0.1 * 1 + (1 - 0.1) * 0.

    assert np.isclose(archive.best_elite["solution"], [1, 2, 3]).all()
    assert np.isclose(archive.best_elite["objective"], 1.0).all()
    assert np.isclose(archive.best_elite["measures"], [0, 0]).all()
    assert np.isclose(archive.stats.obj_max, 1.0)

    # Add an elite with lower objective value than best elite but higher
    # objective value than threshold.
    if add_mode == "single":
        archive.add_single([4, 5, 6], 0.2, [0, 0])
    else:
        archive.add([[4, 5, 6]], [0.2], [[0, 0]])

    # Best elite remains the same even though this is a non-elitist archive and
    # the best elite is no longer in the archive.
    assert np.isclose(archive.best_elite["solution"], [1, 2, 3]).all()
    assert np.isclose(archive.best_elite["objective"], 1.0)
    assert np.isclose(archive.best_elite["measures"], [0, 0]).all()
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
        data.archive.index_of_single(data.measures[:-1])


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
    if isinstance(data.archive, MAE_ARCHIVES):
        assert data.archive.learning_rate == 1.0  # Default value.


def test_threshold_min_correct(data):
    if isinstance(data.archive, MAE_ARCHIVES):
        assert data.archive.threshold_min == -np.inf  # Default value.


def test_qd_score_offset_correct(data):
    assert data.archive.qd_score_offset == 0.0  # Default value.


def test_field_list_correct(data):
    if isinstance(data.archive, MAE_ARCHIVES):
        assert data.archive.field_list == [
            "solution",
            "objective",
            "measures",
            "threshold",
            "index",
        ]
    else:
        assert data.archive.field_list == ["solution", "objective", "measures", "index"]


def test_basic_stats(data):
    assert data.archive.stats.num_elites == 0
    assert data.archive.stats.coverage == 0.0
    assert data.archive.stats.qd_score == 0.0
    assert data.archive.stats.norm_qd_score == 0.0
    assert data.archive.stats.obj_max is None
    assert data.archive.stats.obj_mean is None

    assert data.archive_with_elite.stats.num_elites == 1

    if data.name == "ProximityArchive":
        assert data.archive_with_elite.stats.coverage == 1.0
        assert data.archive_with_elite.stats.norm_qd_score == data.objective
    else:
        assert data.archive_with_elite.stats.coverage == 1 / data.cells
        assert (
            data.archive_with_elite.stats.norm_qd_score == data.objective / data.cells
        )
    assert data.archive_with_elite.stats.qd_score == data.objective
    assert data.archive_with_elite.stats.obj_max == data.objective
    assert data.archive_with_elite.stats.obj_mean == data.objective


def test_unstructured_stats_after_none_objective():
    archive = ProximityArchive(
        solution_dim=3, measure_dim=2, k_neighbors=1, novelty_threshold=1.0
    )
    archive.add_single([1, 2, 3], None, [0, 0])

    assert archive.stats.coverage == 1.0
    assert archive.stats.qd_score == 0.0
    assert archive.stats.norm_qd_score == 0.0
    assert archive.stats.obj_max == 0.0
    assert archive.stats.obj_mean == 0.0


def test_retrieve_gets_correct_elite(data):
    occupied, elites = data.archive_with_elite.retrieve([data.measures])
    assert occupied[0]
    assert np.all(elites["solution"][0] == data.solution)
    assert elites["objective"][0] == data.objective
    assert np.all(elites["measures"][0] == data.measures)
    if isinstance(data.archive_with_elite, MAE_ARCHIVES):
        assert elites["threshold"][0] == data.objective
    # Avoid checking elites["index"] since the meaning varies by archive.


def test_retrieve_empty_values(data):
    if data.name == "ProximityArchive":
        # No solutions in the archive, so ProximityArchive cannot retrieve
        # anything.
        with pytest.raises(RuntimeError):
            data.archive.retrieve([data.measures])
    else:
        occupied, elites = data.archive.retrieve([data.measures])
        assert not occupied[0]
        assert np.all(np.isnan(elites["solution"][0]))
        assert np.isnan(elites["objective"])

        if isinstance(data.archive, CategoricalArchive):
            assert all(m is None for m in elites["measures"][0])
        else:
            assert np.all(np.isnan(elites["measures"][0]))

        if isinstance(data.archive, MAE_ARCHIVES):
            assert np.isnan(elites["threshold"])

        assert elites["index"][0] == -1


def test_retrieve_wrong_shape(data):
    with pytest.raises(ValueError):
        data.archive.retrieve([data.measures[:-1]])


def test_retrieve_single_gets_correct_elite(data):
    occupied, elite = data.archive_with_elite.retrieve_single(data.measures)
    assert occupied
    assert np.all(elite["solution"] == data.solution)
    assert elite["objective"] == data.objective
    assert np.all(elite["measures"] == data.measures)
    if isinstance(data.archive_with_elite, MAE_ARCHIVES):
        assert elite["threshold"] == data.objective
    # Avoid checking elite["index"] since the meaning varies by archive.


def test_retrieve_single_empty_values(data):
    if data.name == "ProximityArchive":
        # No solutions in the archive, so ProximityArchive cannot retrieve
        # anything.
        with pytest.raises(RuntimeError):
            data.archive.retrieve_single(data.measures)
    else:
        occupied, elite = data.archive.retrieve_single(data.measures)
        assert not occupied
        assert np.all(np.isnan(elite["solution"]))
        assert np.isnan(elite["objective"])

        if isinstance(data.archive, CategoricalArchive):
            assert all(m is None for m in elite["measures"])
        else:
            assert np.all(np.isnan(elite["measures"]))

        if isinstance(data.archive_with_elite, MAE_ARCHIVES):
            assert np.isnan(elite["threshold"])

        assert elite["index"] == -1


def test_retrieve_single_wrong_shape(data):
    with pytest.raises(ValueError):
        data.archive.retrieve_single(data.measures[:-1])


def test_sample_elites_gets_single_elite(data):
    elites = data.archive_with_elite.sample_elites(2)
    assert np.all(elites["solution"] == data.solution)
    assert np.all(elites["objective"] == data.objective)
    assert np.all(elites["measures"] == data.measures)
    # Avoid checking elite["index"] since the meaning varies by archive.


def test_sample_elites_fails_when_empty(data):
    with pytest.raises(IndexError):
        data.archive.sample_elites(1)


@pytest.mark.parametrize("setting_for_n", ["enough_n", "too_many_n"])
def test_sample_elites_with_replacement(data, setting_for_n):
    if isinstance(data.archive, CategoricalArchive):
        data.archive.add(
            solution=np.zeros((3, 3)),
            objective=[1, 2, 3],
            measures=[["A", "One"], ["A", "Two"], ["A", "Three"]],
        )
    else:
        data.archive.add(
            solution=np.zeros((3, 3)),
            objective=[1, 2, 3],
            measures=[[-1, -1], [-1, 1], [1, 1]],
        )

    if setting_for_n == "enough_n":
        # Sampling exactly 3 with replace=False should cause the 3 elites to be sampled.
        elites = data.archive.sample_elites(3, replace=False)
        assert np.allclose(np.sort(elites["objective"]), [1, 2, 3])
    elif setting_for_n == "too_many_n":
        # Sampling more than the number of elites  with replace=False throws an error.
        with pytest.raises(
            ValueError,
            match=r"Cannot take a larger sample than the number of elites in the archive .*",
        ):
            elites = data.archive.sample_elites(4, replace=False)
    else:
        raise ValueError


@pytest.mark.parametrize("name", ARCHIVE_NAMES)
@pytest.mark.parametrize("with_elite", [True, False], ids=["nonempty", "empty"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
def test_pandas_data(name, with_elite, dtype):
    data = get_archive_data(name, dtype=dtype)

    # Set up expected columns and data types.
    solution_dim = len(data.solution)
    measure_dim = len(data.measures)
    expected_cols = (
        [f"solution_{i}" for i in range(solution_dim)]
        + ["objective"]
        + [f"measures_{i}" for i in range(measure_dim)]
    )

    expected_dtypes = [dtype for _ in range(solution_dim)] + [dtype]
    if isinstance(data.archive, CategoricalArchive):
        expected_dtypes += [np.object_ for _ in range(measure_dim)]
    else:
        expected_dtypes += [dtype for _ in range(measure_dim)]

    expected_data = [*data.solution, data.objective, *data.measures]

    if isinstance(data.archive, MAE_ARCHIVES):
        expected_cols += ["threshold", "index"]
        expected_dtypes += [dtype, np.int32]
        expected_data.append(data.objective)  # Append the threshold.
    else:
        expected_cols += ["index"]
        expected_dtypes += [np.int32]

    # Retrieve the dataframe.
    if with_elite:
        df = data.archive_with_elite.data(return_type="pandas")
    else:
        df = data.archive.data(return_type="pandas")

    # Check columns and data types.
    assert (df.columns == expected_cols).all()
    assert (df.dtypes == expected_dtypes).all()

    if with_elite:
        if isinstance(data.archive_with_elite, CVTArchive):
            # For CVTArchive, we check the centroid because the index can vary.
            index = df.loc[0, "index"]
            assert (data.archive_with_elite.centroids[index] == data.centroid).all()
        elif isinstance(
            data.archive_with_elite,
            (
                CategoricalArchive,
                GridArchive,
                SlidingBoundariesArchive,
            ),
        ):
            # These archives have expected grid indices.
            assert (
                df.loc[0, "index"]
                == data.archive.grid_to_int_index([data.grid_indices])[0]
            )
        else:
            # Archives where indices can't be tested.
            assert isinstance(data.archive_with_elite, (ProximityArchive,))

        # Comparing the df to the list of expected data seems to make things be
        # marked unequal when the dtypes are mixed between object and scalar.
        assert list(df.iloc[0, : len(expected_data)]) == expected_data
