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
def table():
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
        # Two solutions but only 1 of everything else.
        EliteTable(
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([1]),
            np.array([[1, 2]]),
            np.array([[3, 4]]),
            np.array([None]),
        )


def test_length(table):
    assert len(table) == 5


def test_iteration(table):
    for (elite, (sol, obj, beh, idx, meta)) in zip(
            table,
            zip(table.solutions, table.objective_values, table.behavior_values,
                table.indices, table.metadata)):
        assert_elite_eq(elite, Elite(sol, obj, beh, tuple(idx), meta))


def test_index_int(table):
    assert_elite_eq(
        table[0],
        Elite(table.solutions[0],
              table.objective_values[0], table.behavior_values[0],
              tuple(table.indices[0]), table.metadata[0]))


def test_index_array(table):
    table2 = table[[0, 4]]
    assert len(table2) == 2
    assert_elite_eq(
        table2[0],
        Elite(table.solutions[0],
              table.objective_values[0], table.behavior_values[0],
              tuple(table.indices[0]), table.metadata[0]))
    assert_elite_eq(
        table2[1],
        Elite(table.solutions[4],
              table.objective_values[4], table.behavior_values[4],
              tuple(table.indices[4]), table.metadata[4]))


def test_index_with_comparisons(table):
    table2 = table[table.objective_values == 2.0]
    assert len(table2) == 2
    assert_elite_eq(
        table2[0],
        Elite(table.solutions[1], 2.0, table.behavior_values[1],
              tuple(table.indices[1]), table.metadata[1]))
    assert_elite_eq(
        table2[1],
        Elite(table.solutions[2], 2.0, table.behavior_values[2],
              tuple(table.indices[2]), table.metadata[2]))


def test_item():
    table = EliteTable(
        np.array([[1, 2, 3]], dtype=float),
        np.array([1], dtype=float),
        np.array([[1, 2]], dtype=float),
        np.array([[0, 1]], dtype=int),
        np.array([{
            "a": 1
        }]),
    )
    elite = table.item()
    assert_elite_eq(
        elite,
        Elite(np.array([1, 2, 3]), 1.0, np.array([1, 2]), (0, 1), {"a": 1}))


def test_item_fails_when_more_than_one_elite(table):
    with pytest.raises(ValueError):
        table.item()


def test_filter(table):
    table2 = table.filter(lambda elite: elite.meta is None)
    assert len(table2) == 2
    assert_elite_eq(
        table2[0],
        Elite(table.solutions[3], table.objective_values[3],
              table.behavior_values[3], tuple(table.indices[3]), None))
    assert_elite_eq(
        table2[1],
        Elite(table.solutions[4], table.objective_values[4],
              table.behavior_values[4], tuple(table.indices[4]), None))
