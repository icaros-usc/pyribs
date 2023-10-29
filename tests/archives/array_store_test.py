"""Tests for ArrayStore."""
import numpy as np
import pytest

from ribs.archives import ArrayStore

# pylint: disable = redefined-outer-name


def test_init():
    capacity = 10
    store = ArrayStore(
        {
            "objective": ((), np.float32),
            "measures": ((2,), np.float32),
            "solution": ((10,), np.float32),
        },
        capacity,
    )

    assert len(store) == 0
    assert store.capacity == capacity
    assert np.all(~store.occupied)
    assert len(store.occupied_list) == 0


@pytest.fixture
def store():
    """Simple ArrayStore for testing."""
    return ArrayStore(
        field_desc={
            "objective": ((), np.float32),
            "measures": ((2,), np.float32),
            "solution": ((10,), np.float32),
        },
        capacity=10,
    )


def test_add_wrong_keys(store):
    with pytest.raises(ValueError):
        store.add(
            [0, 1],
            {
                "objective": [1.0, 2.0],
                "measures": [[1.0, 2.0], [3.0, 4.0]],
                # Missing `solution` key.
            },
            [],  # Empty transforms.
        )


def test_add_mismatch_indices(store):
    with pytest.raises(ValueError):
        store.add(
            [0, 1],
            {
                "objective": [1.0, 2.0, 3.0],  # Length 3 instead of 2.
                "measures": [[1.0, 2.0], [3.0, 4.0]],
                "solution": [np.zeros(10), np.ones(10)],
            },
            [],  # Empty transforms.
        )


def test_simple_add_and_retrieve(store):
    """Add without transforms and then retrieve the data."""
    store.add(
        [3, 5],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": [np.zeros(10), np.ones(10)],
        },
        [],  # Empty transforms.
    )

    assert len(store) == 2
    assert np.all(store.occupied == [0, 0, 0, 1, 0, 1, 0, 0, 0, 0])
    assert np.all(np.sort(store.occupied_list) == [3, 5])

    occupied, data = store.retrieve([5, 3])

    assert np.all(occupied == [True, True])
    assert data.keys() == set(["objective", "measures", "solution"])
    assert np.all(data["objective"] == [2.0, 1.0])
    assert np.all(data["measures"] == [[3.0, 4.0], [1.0, 2.0]])
    assert np.all(data["solution"] == [np.ones(10), np.zeros(10)])


def test_add_duplicate_indices(store):
    store.add(
        [3, 3],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": [np.zeros(10), np.ones(10)],
        },
        [],  # Empty transforms.
    )

    assert len(store) == 1
    assert np.all(store.occupied == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    assert np.all(store.occupied_list == [3])


def test_retrieve_duplicate_indices(store):
    store.add(
        [3],
        {
            "objective": [2.0],
            "measures": [[3.0, 4.0]],
            "solution": [np.ones(10)],
        },
        [],  # Empty transforms.
    )

    occupied, data = store.retrieve([3, 3])

    assert np.all(occupied == [True, True])
    assert data.keys() == set(["objective", "measures", "solution"])
    assert np.all(data["objective"] == [2.0, 2.0])
    assert np.all(data["measures"] == [[3.0, 4.0], [3.0, 4.0]])
    assert np.all(data["solution"] == [np.ones(10), np.ones(10)])
