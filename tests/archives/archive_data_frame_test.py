"""Tests for ArchiveDataFrame."""
import numpy as np
import pytest

from ribs.archives import ArchiveDataFrame

# pylint: disable = redefined-outer-name


@pytest.fixture
def data():
    """Data for an archive."""
    solution_batch = np.arange(5).reshape(5, 1)
    objective_batch = 2 * np.arange(5)
    measures_batch = 3 * np.arange(5).reshape(5, 1)
    index_batch = 4 * np.arange(5, dtype=int)
    metadata_batch = [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}, {"e": 5}]
    return (solution_batch, objective_batch, measures_batch, index_batch,
            metadata_batch)


@pytest.fixture
def df(data):
    """Mimics the ArchiveDataFrame which an as_pandas method would generate."""
    (solution_batch, objective_batch, measures_batch, index_batch,
     metadata_batch) = data
    return ArchiveDataFrame({
        "index": index_batch,
        "objective": objective_batch,
        "measure_0": measures_batch[:, 0],
        "solution_0": solution_batch[:, 0],
        "metadata": metadata_batch,
    })


def test_iterelites(data, df):
    for elite, (solution, objective, measures, index,
                metadata) in zip(df.iterelites(), zip(*data)):
        assert np.isclose(elite.solution, solution).all()
        assert np.isclose(elite.objective, objective)
        assert np.isclose(elite.measures, measures).all()
        assert elite.index == index
        assert elite.metadata == metadata


def test_batch_methods(data, df):
    (solution_batch, objective_batch, measures_batch, index_batch,
     metadata_batch) = data
    assert np.isclose(df.solution_batch(), solution_batch).all()
    assert np.isclose(df.objective_batch(), objective_batch).all()
    assert np.isclose(df.measures_batch(), measures_batch).all()
    assert (df.index_batch() == index_batch).all()
    assert (df.metadata_batch() == metadata_batch).all()


@pytest.mark.parametrize(
    "remove",
    ["index", "objective", "measure_0", "metadata", "solution_0"],
    ids=["index", "objective", "measures", "metadata", "solutions"],
)
def test_batch_methods_can_be_none(df, remove):
    """Removes a column so that the corresponding batch method returns None."""
    del df[remove]

    method = {
        "solution_0": df.solution_batch,
        "objective": df.objective_batch,
        "measure_0": df.measures_batch,
        "index": df.index_batch,
        "metadata": df.metadata_batch,
    }[remove]

    assert method() is None


def test_correct_constructor(df):
    """Checks that we defined the _constructor property.

    Essentially, methods which return a DataFrame should now return an
    ArchiveDataFrame.
    """
    assert isinstance(df.iloc[[0, 1]], ArchiveDataFrame)
    assert isinstance(df[["objective", "metadata"]], ArchiveDataFrame)
