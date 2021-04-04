"""Tests for ArchiveBase."""
import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.archives._archive_base import RandomBuffer

from .conftest import ARCHIVE_NAMES, get_archive_data

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


def test_solution_dim_requires_init(_data):
    archive = GridArchive([20, 20], [(-1, 1)] * 2)
    with pytest.raises(RuntimeError):
        _ = archive.solution_dim


#
# Test the dtypes of all classes, as some classes use the dtype in their
# constructor.
#


@pytest.mark.parametrize("name", ARCHIVE_NAMES)
@pytest.mark.parametrize("dtype", [("f", np.float32), ("d", np.float64)],
                         ids=["f", "d"])
def test_str_dtype_float(name, dtype):
    str_dtype, np_dtype = dtype
    data = get_archive_data(name, str_dtype)
    assert data.archive.dtype == np_dtype


def test_invalid_dtype():
    with pytest.raises(ValueError):
        GridArchive([20, 20], [(-1, 1)] * 2, dtype=np.int32)


#
# ArchiveBase tests -- should work for all archive classes.
#

# pylint: disable = redefined-outer-name


@pytest.fixture(params=ARCHIVE_NAMES)
def _data(request):
    """Provides data for testing all kinds of archives."""
    return get_archive_data(request.param)


def test_archive_cannot_reinit(_data):
    with pytest.raises(RuntimeError):
        _data.archive.initialize(len(_data.solution))


def test_new_archive_is_empty(_data):
    assert _data.archive.empty


def test_archive_with_entry_is_not_empty(_data):
    assert not _data.archive_with_entry.empty


def test_bins_correct(_data):
    assert _data.archive.bins == _data.bins


def test_behavior_dim_correct(_data):
    assert _data.archive.behavior_dim == len(_data.behavior_values)


def test_solution_dim_correct(_data):
    assert _data.archive.solution_dim == len(_data.solution)


def test_elite_with_behavior_gets_correct_elite(_data):
    sol, obj, beh, meta = _data.archive_with_entry.elite_with_behavior(
        _data.behavior_values)
    assert (sol == _data.solution).all()
    assert obj == _data.objective_value
    assert (beh == _data.behavior_values).all()
    assert meta == _data.metadata


def test_elite_with_behavior_returns_none(_data):
    sol, obj, beh, meta = _data.archive.elite_with_behavior(
        _data.behavior_values)
    assert sol is None
    assert obj is None
    assert beh is None
    assert meta is None


def test_random_elite_gets_single_elite(_data):
    sol, obj, beh, meta = _data.archive_with_entry.get_random_elite()
    assert np.all(sol == _data.solution)
    assert obj == _data.objective_value
    assert np.all(beh == _data.behavior_values)
    assert meta == _data.metadata


def test_random_elite_fails_when_empty(_data):
    with pytest.raises(IndexError):
        _data.archive.get_random_elite()


@pytest.mark.parametrize("name", ARCHIVE_NAMES)
@pytest.mark.parametrize("with_entry", [True, False], ids=["nonempty", "empty"])
@pytest.mark.parametrize("include_solutions", [True, False],
                         ids=["solutions", "no_solutions"])
@pytest.mark.parametrize("include_metadata", [True, False],
                         ids=["metadata", "no_metadata"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32],
                         ids=["float64", "float32"])
def test_as_pandas(name, with_entry, include_solutions, include_metadata,
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
    if with_entry:
        df = data.archive_with_entry.as_pandas(include_solutions,
                                               include_metadata)
    else:
        df = data.archive.as_pandas(include_solutions, include_metadata)

    # Check columns and data types.
    assert (df.columns == expected_cols).all()
    assert (df.dtypes == expected_dtypes).all()

    if with_entry:
        if is_cvt:
            # For CVTArchive, we check the centroid because the index can vary.
            index = df.loc[0, "index_0"]
            assert (data.archive_with_entry.centroids[index] == data.centroid
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
