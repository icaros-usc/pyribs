"""Tests for ArchiveDataFrame."""

import numpy as np
import pytest

from ribs.archives import ArchiveDataFrame


@pytest.fixture
def data():
    """Data for an archive."""
    solution_batch = np.arange(5).reshape(5, 1)
    objective_batch = 2 * np.arange(5)
    measures_batch = 3 * np.arange(5).reshape(5, 1)
    index_batch = 4 * np.arange(5, dtype=int)
    metadata_batch = [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}, {"e": 5}]
    return (
        solution_batch,
        objective_batch,
        measures_batch,
        index_batch,
        metadata_batch,
    )


@pytest.fixture
def df(data):
    """Mimics an ArchiveDataFrame that a data method would generate."""
    (solution_batch, objective_batch, measures_batch, index_batch, metadata_batch) = (
        data
    )
    return ArchiveDataFrame(
        {
            "solution_0": solution_batch[:, 0],
            "objective": objective_batch,
            # Fancy name to test field handling.
            "foo__bar_measures_3_0": measures_batch[:, 0],
            "metadata": metadata_batch,
            "index": index_batch,
        }
    )


def test_iterelites(data, df):
    for elite, (solution, objective, measures, index, metadata) in zip(
        df.iterelites(), zip(*data)
    ):
        assert np.isclose(elite["solution"], solution).all()
        assert np.isclose(elite["objective"], objective)
        assert np.isclose(elite["foo__bar_measures_3"], measures).all()
        assert elite["metadata"] == metadata
        assert elite["index"] == index


def test_get_field(data, df):
    (solution_batch, objective_batch, measures_batch, index_batch, metadata_batch) = (
        data
    )
    assert np.isclose(df.get_field("solution"), solution_batch).all()
    assert np.isclose(df.get_field("objective"), objective_batch).all()
    assert np.isclose(df.get_field("foo__bar_measures_3"), measures_batch).all()
    assert (df.get_field("metadata") == metadata_batch).all()
    assert (df.get_field("index") == index_batch).all()


@pytest.mark.parametrize(
    "field_col",
    [
        ["solution", "solution_0"],
        ["objective", "objective"],
        ["measures", "foo__bar_measures_3_0"],
        ["metadata", "metadata"],
        ["index", "index"],
    ],
    ids=[
        "solutions",
        "objective",
        "measures",
        "metadata",
        "index",
    ],
)
def test_field_not_found(df, field_col):
    """Removes a column so that get_field has KeyError."""
    field, col = field_col
    del df[col]
    with pytest.raises(KeyError, match=f"Field '{field}' was not found."):
        df.get_field(field)


def test_correct_constructor(df):
    """Checks that we defined the _constructor property.

    Essentially, methods which return a DataFrame should now return an
    ArchiveDataFrame.
    """
    assert isinstance(df.iloc[[0, 1]], ArchiveDataFrame)
    assert isinstance(df[["objective", "metadata"]], ArchiveDataFrame)
