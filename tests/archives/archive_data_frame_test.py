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
    indices = [(idx,) for idx in 4 * np.arange(5)]
    metadata = [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}, {"e": 5}]
    return solutions, objectives, behaviors, indices, metadata


@pytest.fixture
def df(data):
    """Mimics the ArchiveDataFrame an as_pandas method would generate."""
    solutions, objectives, behaviors, indices, metadata = data
    return ArchiveDataFrame({
        "index_0": [idx[0] for idx in indices],
        "objective": objectives,
        "behavior_0": behaviors[:, 0],
        "solution_0": solutions[:, 0],
        "metadata": metadata,
    })


def test_iterelites(data, df):
    for elite, (sol, obj, beh, idx, meta) in zip(df.iterelites(), zip(*data)):
        assert np.isclose(elite.sol, sol).all()
        assert np.isclose(elite.obj, obj)
        assert np.isclose(elite.beh, beh).all()
        assert elite.idx == idx
        assert elite.meta == meta


def test_batch_attributes(data, df):
    solutions, objectives, behaviors, indices, metadata = data
    assert np.isclose(df.batch_solutions(), solutions).all()
    assert np.isclose(df.batch_objectives(), objectives).all()
    assert np.isclose(df.batch_behaviors(), behaviors).all()
    assert df.batch_indices() == indices
    assert (df.batch_metadata() == metadata).all()


def test_batch_solutions_is_none(data):
    _, objectives, behaviors, indices, metadata = data
    df = ArchiveDataFrame({
        "index_0": [idx[0] for idx in indices],
        "objective": objectives,
        "behavior_0": behaviors[:, 0],
        "metadata": metadata,
    })

    assert df.batch_solutions() is None


def test_batch_metadata_is_none(data):
    solutions, objectives, behaviors, indices, _ = data
    df = ArchiveDataFrame({
        "index_0": [idx[0] for idx in indices],
        "objective": objectives,
        "behavior_0": behaviors[:, 0],
        "solution_0": solutions[:, 0],
    })

    assert df.batch_metadata() is None
