"""Tests for transforms."""
import numpy as np

from ribs.archives._transforms import compute_best_index


def test_best_index():
    _, _, add_info = compute_best_index(
        indices=np.array([0, 10, 6]),
        new_data={"objective": np.array([1.0, 5.0, 3.0])},
        add_info={},
        extra_args={},
        occupied=[0, 1, 2],
        cur_data={"objective": np.array([0.0, 2.0, 4.0])},
    )

    assert "best_index" in add_info
    assert add_info["best_index"] == 10


def test_best_index_no_indices():
    _, _, add_info = compute_best_index(
        indices=np.array([]),
        new_data={"objective": np.array([])},
        add_info={},
        extra_args={},
        occupied=[],
        cur_data={"objective": np.array([])},
    )

    assert "best_index" in add_info
    assert add_info["best_index"] is None
