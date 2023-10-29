"""Tests for ArrayStore."""
import numpy as np

from ribs.archives import ArrayStore


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
