"""Tests for theshold update in archive."""
import numpy as np
import pytest

from ribs.archives import GridArchive

from .conftest import get_archive_data

# pylint: disable = redefined-outer-name


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

    # pylint: disable = protected-access
    result_test, _ = archive._compute_new_thresholds(threshold_arr,
                                                     objective_batch,
                                                     index_batch, learning_rate)
    result_true = calc_expected_threshold(objective_batch, threshold_arr[0],
                                          learning_rate)

    assert pytest.approx(result_test[0]) == result_true


@pytest.mark.parametrize("learning_rate", [0, 0.001, 0.01, 0.1, 1])
def test_threshold_update_for_multiple_cells(data, learning_rate):
    archive = data.archive

    threshold_arr = np.array([-3.1, 0.4, 2.9])
    objective_batch = np.array([
        0.1, 0.3, 0.9, 400.0, 42.0, 0.44, 0.53, 0.51, 0.80, 0.71, 33.6, 61.78,
        81.71, 83.48, 41.18
    ])
    index_batch = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

    # pylint: disable = protected-access
    result_test, _ = archive._compute_new_thresholds(threshold_arr,
                                                     objective_batch,
                                                     index_batch, learning_rate)

    result_true = [
        calc_expected_threshold(objective_batch[5 * i:5 * (i + 1)],
                                threshold_arr[i], learning_rate)
        for i in range(3)
    ]

    assert np.all(np.isclose(result_test, result_true))


def test_threshold_update_for_empty_objective_and_index(data):
    archive = data.archive

    threshold_arr = np.array([-3.1, 0.4, 2.9])
    objective_batch = np.array([])  # Empty objective.
    index_batch = np.array([])  # Empty index.

    # pylint: disable = protected-access
    new_threshold_batch, threshold_update_indices = (
        archive._compute_new_thresholds(threshold_arr, objective_batch,
                                        index_batch, 0.1))

    assert new_threshold_batch.size == 0
    assert threshold_update_indices.size == 0


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
