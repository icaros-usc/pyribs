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


def test_simple_add_retrieve_clear(store):
    """Add without transforms, retrieve the data, and clear the archive."""
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

    store.clear()

    assert len(store) == 0
    assert np.all(store.occupied == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert len(store.occupied_list) == 0


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


def test_add_simple_transform(store):

    def obj_meas(indices, new_data, add_info, occupied, cur_data):
        # pylint: disable = unused-argument
        new_data["objective"] = np.sum(new_data["solution"], axis=1)
        new_data["measures"] = np.asarray(new_data["solution"])[:, :2]
        return indices, new_data, {"foo": 5}

    add_info = store.add(
        [3, 5],
        {
            "solution": [np.ones(10), 2 * np.ones(10)],
        },
        [obj_meas],
    )

    assert add_info == {"foo": 5}

    assert len(store) == 2
    assert np.all(store.occupied == [0, 0, 0, 1, 0, 1, 0, 0, 0, 0])
    assert np.all(np.sort(store.occupied_list) == [3, 5])

    occupied, data = store.retrieve([3, 5])

    assert np.all(occupied == [True, True])
    assert data.keys() == set(["objective", "measures", "solution"])
    assert np.all(data["objective"] == [10.0, 20.0])
    assert np.all(data["measures"] == [[1.0, 1.0], [2.0, 2.0]])
    assert np.all(data["solution"] == [np.ones(10), 2 * np.ones(10)])


def test_as_dict(store):
    store.add(
        [3, 5],
        {
            "objective": [1.0, 2.0],
            "measures": [[1.0, 2.0], [3.0, 4.0]],
            "solution": [np.zeros(10), np.ones(10)],
        },
        [],  # Empty transforms.
    )

    d = store.as_dict()

    assert d.keys() == set([
        "props.capacity",
        "props.occupied",
        "props.n_occupied",
        "props.occupied_list",
        "fields.objective",
        "fields.measures",
        "fields.solution",
    ])
    assert d["props.capacity"] == 10
    assert np.all(d["props.occupied"] == [0, 0, 0, 1, 0, 1, 0, 0, 0, 0])
    assert d["props.n_occupied"] == 2
    assert np.all(np.sort(d["props.occupied_list"][:2]) == [3, 5])
    assert np.all(d["fields.objective"][[3, 5]] == [1.0, 2.0])
    assert np.all(d["fields.measures"][[3, 5]] == [[1.0, 2.0], [3.0, 4.0]])
    assert np.all(d["fields.solution"][[3, 5]] == [np.zeros(10), np.ones(10)])
