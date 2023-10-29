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
    return ArrayStore(
        field_desc={
            "objective": ((), np.float32),
            "measures": ((2,), np.float32),
            "solution": ((10,), np.float32),
        },
        capacity=10,
    )


# TODO test retrieve


def test_add_wrong_keys(store):
    with pytest.raises(ValueError):
        store.add(
            [0, 1],
            {
                "objective": [1.0, 2.0],
                "measures": [[1.0, 2.0], [3.0, 4.0]]
                # Missing `solution` key.
            },
            [],  # Empty transforms.
        )
