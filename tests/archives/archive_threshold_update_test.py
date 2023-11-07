"""Tests for theshold update in archive."""
import numpy as np
import pytest

from ribs.archives import GridArchive
from ribs.archives._transforms import _compute_thresholds

# pylint: disable = redefined-outer-name, missing-function-docstring


def update_threshold(threshold, f_val, learning_rate):
    return (1.0 - learning_rate) * threshold + learning_rate * f_val


def calc_geom(base_value, exponent):
    if base_value == 1.0:
        return exponent
    top = 1 - base_value**exponent
    bottom = 1 - base_value
    return top / bottom


def calc_expected_threshold(additions, cell_value, learning_rate):
    k = len(additions)
    geom = calc_geom(1.0 - learning_rate, k)
    f_star = sum(additions)
    term1 = learning_rate * f_star * geom / k
    term2 = cell_value * (1.0 - learning_rate)**k
    return term1 + term2


@pytest.mark.parametrize("learning_rate", [0, 0.001, 0.01, 0.1, 1])
def test_threshold_update_for_one_cell(learning_rate):
    cur_threshold = np.full(5, -3.1)
    objective = np.array([0.1, 0.3, 0.9, 400.0, 42.0])
    indices = np.zeros(5, dtype=np.int32)

    result_test = _compute_thresholds(indices, objective, cur_threshold,
                                      learning_rate, np.float64)
    result_true = calc_expected_threshold(objective, cur_threshold[0],
                                          learning_rate)

    # The result should have 5 duplicate entries with the new threshold.
    assert result_test.shape == (5,)
    assert np.all(np.isclose(result_test, result_true))


@pytest.mark.parametrize("learning_rate", [0, 0.001, 0.01, 0.1, 1])
def test_threshold_update_for_multiple_cells(learning_rate):
    cur_threshold = np.repeat([-3.1, 0.4, 2.9], 5)
    objective = np.array([
        0.1, 0.3, 0.9, 400.0, 42.0, 0.44, 0.53, 0.51, 0.80, 0.71, 33.6, 61.78,
        81.71, 83.48, 41.18
    ])  # 15 values.
    indices = np.repeat([0, 1, 2], 5)

    result_test = _compute_thresholds(indices, objective, cur_threshold,
                                      learning_rate, np.float64)
    result_true = np.repeat([
        calc_expected_threshold(objective[5 * i:5 * (i + 1)],
                                cur_threshold[5 * i], learning_rate)
        for i in range(3)
    ], 5)

    assert result_test.shape == (15,)
    assert np.all(np.isclose(result_test, result_true))


def test_threshold_update_for_empty_objective_and_index():
    cur_threshold = np.array([])
    objective = np.array([])  # Empty objective.
    indices = np.array([], dtype=np.int32)  # Empty index.

    new_threshold = _compute_thresholds(indices, objective, cur_threshold, 0.1,
                                        np.float64)

    assert new_threshold.shape == (0,)


def test_init_learning_rate_and_threshold_min():
    # Setting threshold_min while not setting the learning_rate should not
    # raise an error.
    _ = GridArchive(solution_dim=2,
                    dims=[10, 20],
                    ranges=[(-1, 1), (-2, 2)],
                    threshold_min=0)

    # Setting both learning_rate and threshold_min should not raise an error.
    _ = GridArchive(solution_dim=2,
                    dims=[10, 20],
                    ranges=[(-1, 1), (-2, 2)],
                    learning_rate=0.1,
                    threshold_min=0)

    # Setting learning_rate while not setting the threshold_min should raise an
    # error.
    with pytest.raises(ValueError):
        _ = GridArchive(solution_dim=2,
                        dims=[10, 20],
                        ranges=[(-1, 1), (-2, 2)],
                        learning_rate=0.1)
