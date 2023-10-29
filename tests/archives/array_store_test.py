"""Tests for ArrayStore."""
import numpy as np

from ribs.archives import ArrayStore


def test_init():
    ArrayStore(
        {
            "objective": ((), np.float32),
            "measures": ((2,), np.float32),
            "solution": ((10,), np.float32),
        },
        10,
    )
