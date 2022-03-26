"""Tests for ArchiveDataFrame."""
import numpy as np
import pytest

from ribs.archives import ArchiveDataFrame

# pylint: disable = redefined-outer-name


@pytest.fixture
def data():
    """Data for an archive."""
    solutions = np.arange(5).reshape(5, 1)
    objectives = 2 * np.arange(5)
    behaviors = 3 * np.arange(5).reshape(5, 1)
    indices = 4 * np.arange(5, dtype=int)
    metadata = [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}, {"e": 5}]
    return solutions, objectives, behaviors, indices, metadata


@pytest.fixture
def df(data):
    """Mimics the ArchiveDataFrame an as_pandas method would generate."""
    solutions, objectives, behaviors, indices, metadata = data
    return ArchiveDataFrame({
        "index": indices,
        "objective": objectives,
        "behavior_0": behaviors[:, 0],
        "solution_0": solutions[:, 0],
        "metadata": metadata,
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
    solutions, objectives, behaviors, indices, metadata = data
    assert np.isclose(df.batch_solutions(), solutions).all()
    assert np.isclose(df.batch_objectives(), objectives).all()
    assert np.isclose(df.batch_behaviors(), behaviors).all()
    assert (df.batch_indices() == indices).all()
    assert (df.batch_metadata() == metadata).all()


@pytest.mark.parametrize(
    "remove",
    ["index", "objective", "behavior_0", "metadata", "solution_0"],
    ids=["indices", "objectives", "behaviors", "metadata", "solutions"],
)
def test_batch_methods_can_be_none(df, remove):
    """Removes a column so that the corresponding batch method returns None."""
    del df[remove]

    method = {
        "solution_0": df.batch_solutions,
        "objective": df.batch_objectives,
        "behavior_0": df.batch_behaviors,
        "index": df.batch_indices,
        "metadata": df.batch_metadata,
    }[remove]

    assert method() is None


def test_correct_constructor(df):
    """Checks that we defined the _constructor property.

    Essentially, methods which return a DataFrame should now return an
    ArchiveDataFrame.
    """
    assert isinstance(df.iloc[[0, 1]], ArchiveDataFrame)
    assert isinstance(df[["objective", "metadata"]], ArchiveDataFrame)
