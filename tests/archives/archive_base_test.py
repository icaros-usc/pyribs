"""Tests for ArchiveBase."""
import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.archives._archive_base import RandomBuffer

from .conftest import ARCHIVE_NAMES, get_archive_data

# pylint: disable = redefined-outer-name

#
# RandomBuffer tests -- note this is an internal class.
#


def test_random_buffer_in_range():
    buffer = RandomBuffer(buf_size=10)
    for _ in range(10):
        assert buffer.get(1) == 0
    for _ in range(10):
        assert buffer.get(2) in [0, 1]


def test_random_buffer_not_repeating():
    buf_size = 100
    buffer = RandomBuffer(buf_size=buf_size)

    # Both lists contain random integers in the range [0,100), and we triggered
    # a reset by retrieving more than buf_size integers. It should be nearly
    # impossible for the lists to be equal unless something is wrong.
    x1 = [buffer.get(100) for _ in range(buf_size)]
    x2 = [buffer.get(100) for _ in range(buf_size)]

    assert x1 != x2


#
# Tests for the require_init decorator. Just need to make sure it works on a few
# methods, as it is too much to test on all.
#


def test_add_requires_init():
    archive = GridArchive([20, 20], [(-1, 1)] * 2)
    with pytest.raises(RuntimeError):
        archive.add(np.array([1, 2, 3]), 1.0, np.array([1.0, 1.0]))


def test_solution_dim_requires_init():
    archive = GridArchive([20, 20], [(-1, 1)] * 2)
    with pytest.raises(RuntimeError):
        _ = archive.solution_dim


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
        GridArchive([20, 20], [(-1, 1)] * 2, dtype=np.int32)


#
# ArchiveBase tests -- should work for all archive classes.
#


@pytest.fixture(params=ARCHIVE_NAMES)
def data(request):
    """Provides data for testing all kinds of archives."""
    return get_archive_data(request.param)


def test_archive_cannot_reinit(data):
    with pytest.raises(RuntimeError):
        data.archive.initialize(len(data.solution))


def test_new_archive_is_empty(data):
    assert data.archive.empty


def test_archive_with_elite_is_not_empty(data):
    assert not data.archive_with_elite.empty


def test_archive_is_empty_after_clear(data):
    data.archive_with_elite.clear()
    assert data.archive_with_elite.empty


def test_bins_correct(data):
    assert data.archive.bins == data.bins


def test_behavior_dim_correct(data):
    assert data.archive.behavior_dim == len(data.behavior_values)


def test_solution_dim_correct(data):
    assert data.archive.solution_dim == len(data.solution)


def test_elite_with_behavior_gets_correct_elite(data):
    elite = data.archive_with_elite.elite_with_behavior(data.behavior_values)
    assert np.all(elite.sol == data.solution)
    assert elite.obj == data.objective_value
    assert np.all(elite.beh == data.behavior_values)
    assert isinstance(elite.idx, (int, tuple))  # Exact val depends on archive.
    assert elite.meta == data.metadata


def test_elite_with_behavior_returns_none(data):
    elite = data.archive.elite_with_behavior(data.behavior_values)
    assert elite.sol is None
    assert elite.obj is None
    assert elite.beh is None
    assert elite.idx is None
    assert elite.meta is None


def test_random_elite_gets_single_elite(data):
    elite = data.archive_with_elite.get_random_elite()
    assert np.all(elite.sol == data.solution)
    assert elite.obj == data.objective_value
    assert np.all(elite.beh == data.behavior_values)
    assert isinstance(elite.idx, (int, tuple))  # Exact val depends on archive.
    assert elite.meta == data.metadata


def test_random_elite_fails_when_empty(data):
    with pytest.raises(IndexError):
        data.archive.get_random_elite()


def test_data(data):
    """General checks for data() method.

    The assert_archive_elite method in the other archive tests already tests the
    correctness of data().
    """
    (all_sols, all_objs, all_behs, all_idxs,
     all_meta) = data.archive_with_elite.data()
    assert len(all_sols) == 1
    assert len(all_objs) == 1
    assert len(all_behs) == 1
    assert len(all_idxs) == 1
    assert isinstance(all_idxs, list)
    assert len(all_meta) == 1


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
    is_cvt = name.startswith("CVTArchive-")

    # Set up expected columns and data types.
    num_index_cols = 1 if is_cvt else len(data.behavior_values)
    index_cols = [f"index_{i}" for i in range(num_index_cols)]
    behavior_cols = [f"behavior_{i}" for i in range(len(data.behavior_values))]
    expected_cols = index_cols + behavior_cols + ["objective"]
    expected_dtypes = [
        *[int for _ in index_cols],
        *[dtype for _ in behavior_cols],
        dtype,
    ]
    if include_solutions:
        solution_cols = [f"solution_{i}" for i in range(len(data.solution))]
        expected_cols += solution_cols
        expected_dtypes += [dtype for _ in solution_cols]
    if include_metadata:
        expected_cols.append("metadata")
        expected_dtypes.append(object)

    # Retrieve the dataframe.
    if with_elite:
        df = data.archive_with_elite.as_pandas(include_solutions,
                                               include_metadata)
    else:
        df = data.archive.as_pandas(include_solutions, include_metadata)

    # Check columns and data types.
    assert (df.columns == expected_cols).all()
    assert (df.dtypes == expected_dtypes).all()

    if with_elite:
        if is_cvt:
            # For CVTArchive, we check the centroid because the index can vary.
            index = df.loc[0, "index_0"]
            assert (data.archive_with_elite.centroids[index] == data.centroid
                   ).all()
        else:
            # Other archives have expected grid indices.
            assert (df.loc[0, index_cols[0]:index_cols[-1]] == list(
                data.grid_indices)).all()

        expected_data = [*data.behavior_values, data.objective_value]
        if include_solutions:
            expected_data += list(data.solution)
        if include_metadata:
            expected_data.append(data.metadata)
        assert (df.loc[0, "behavior_0":] == expected_data).all()
