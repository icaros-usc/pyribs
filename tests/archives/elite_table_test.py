"""Tests for EliteTable."""
import numpy as np
import pytest

from ribs.archives import Elite, EliteTable

# pylint: disable = redefined-outer-name


def assert_elite_eq(a, b):
    """Checks that two Elite's are equal."""
    assert np.all(np.isclose(a.sol, b.sol))
    assert np.isclose(a.obj, b.obj)
    assert np.all(np.isclose(a.beh, b.beh))
    assert a.idx == b.idx
    assert a.meta == b.meta


@pytest.fixture
def data():
    """EliteTable with 5 elites."""
    return EliteTable(
        np.array([[1, 2, 3], [4, 5, 6], [4, 5, 6], [10, 11, 12], [7, 8, 9]],
                 dtype=float),
        np.array([1, 2, 2, 4, 3], dtype=float),
        np.array([[1, 2], [4, 5], [4, 5], [10, 11], [7, 8]], dtype=float),
        np.array([[0, 1], [1, 0], [1, 0], [2, 2], [3, 4]], dtype=int),
        np.array([
            {
                "a": 1
            },
            {
                "b": 2
            },
            {
                "b": 2
            },
            None,
            None,
        ]),
    )


def test_init_fails_diff_length():
    with pytest.raises(ValueError):
        EliteTable(
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([1]),
            np.array([[1, 2]]),
            np.array([[3, 4]]),
            np.array([None]),
        )


def test_length(data):
    assert len(data) == 5


def test_item():
    data = EliteTable(
        np.array([[1, 2, 3]], dtype=float),
        np.array([1], dtype=float),
        np.array([[1, 2]], dtype=float),
        np.array([[0, 1]], dtype=int),
        np.array([{
            "a": 1
        }]),
    )
    elite = data.item()
    assert_elite_eq(
        elite,
        Elite(np.array([1, 2, 3]), 1.0, np.array([1, 2]), (0, 1), {"a": 1}))


def test_item_fails_when_more_than_one_elite(data):
    with pytest.raises(ValueError):
        data.item()


def test_iteration(data):
    for (elite, (sol, obj, beh, idx, meta)) in zip(
            data,
            zip(data.solutions, data.objective_values, data.behavior_values,
                data.indices, data.metadata)):
        assert_elite_eq(elite, Elite(sol, obj, beh, tuple(idx), meta))
