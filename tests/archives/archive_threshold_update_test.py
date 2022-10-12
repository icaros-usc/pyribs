"""Tests for theshold update in archive."""
import numpy as np
import pytest

from .conftest import get_archive_data


@pytest.fixture
def data():
    """Data for grid archive tests."""
    return get_archive_data("GridArchive")


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
def test_threshold_update_for_one_cell(data, learning_rate):
    archive = data.archive

    threshold_arr = np.array([-3.1])
    objective_batch = np.array([0.1, 0.3, 0.9, 400.0, 42.0])
    index_batch = np.array([0, 0, 0, 0, 0])

    result_test, _ = archive._compute_new_thresholds(threshold_arr,
                                                     objective_batch,
                                                     index_batch, learning_rate)
    result_true = calc_expected_threshold(objective_batch, threshold_arr[0],
                                          learning_rate)

    assert pytest.approx(result_test[0]) == result_true


@pytest.mark.parametrize("learning_rate", [0, 0.001, 0.01, 0.1, 1])
def test_consistent_multi_update(data, learning_rate):
    archive = data.archive

    update_size = 3
    old_threshold = [-3.1]
    objective = [0.1, 0.3, 0.9, 400.0, 42.0]

    threshold_arr = np.array(old_threshold * update_size)
    objective_batch = np.array(objective * update_size)
    index_batch = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

    result_test, _ = archive._compute_new_thresholds(threshold_arr,
                                                     objective_batch,
                                                     index_batch, learning_rate)

    result_true = calc_expected_threshold(objective, old_threshold[0],
                                          learning_rate)

    for result in result_test:
        assert pytest.approx(result) == result_true
