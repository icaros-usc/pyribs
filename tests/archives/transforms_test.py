"""Tests for transforms."""
import numpy as np

from ribs.archives._transforms import compute_best_index, compute_objective_sum


def test_objective_sum():
    _, _, add_info = compute_objective_sum(
        indices=np.array([0, 10, 6]),
        new_data={"objective": np.array([1.0, 5.0, 3.0])},
        add_info={},
        extra_args={"objective_sum": 10.0},
        occupied=np.array([True, False, True]),
        cur_data={"objective": np.array([0.0, 2.0, 4.0])},
    )

    assert "objective_sum" in add_info
    assert np.isclose(add_info["objective_sum"],
                      (10.0 + (1.0 - 0.0) + (5.0 - 0.0) + (3.0 - 4.0)))


def test_objective_sum_no_indices():
    _, _, add_info = compute_objective_sum(
        indices=np.array([]),
        new_data={"objective": np.array([])},
        add_info={},
        extra_args={"objective_sum": 10.0},
        occupied=np.array([]),
        cur_data={"objective": np.array([])},
    )

    assert "objective_sum" in add_info
    assert add_info["objective_sum"] == 10.0


def test_best_index():
    _, _, add_info = compute_best_index(
        indices=np.array([0, 10, 6]),
        new_data={"objective": np.array([1.0, 5.0, 3.0])},
        add_info={},
        extra_args={},
        occupied=np.array([True, False, True]),
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
        occupied=np.array([]),
        cur_data={"objective": np.array([])},
    )

    assert "best_index" in add_info
    assert add_info["best_index"] is None
