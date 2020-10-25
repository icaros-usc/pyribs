"""Tests for the Individual."""

import numpy as np

from ribs.archives import Individual

# pylint: disable = invalid-name


def test_individuals_are_equal():
    ind1 = Individual(10.0, np.array([1, 2, 3, 4]),
                      np.array([5, 6, 7, 8, 9, 10]))
    ind2 = Individual(10.0, np.array([1, 2, 3, 4]),
                      np.array([5, 6, 7, 8, 9, 10]))
    assert ind1 == ind2


def test_individuals_are_not_equal_unless_same_shape():
    ind_with_bigger_shape = Individual(10.0, np.array([4, 4, 4, 4]),
                                       np.array([5, 5, 5]))
    ind_with_smaller_shape = Individual(10.0, np.array([4]), np.array([5]))
    assert ind_with_bigger_shape != ind_with_smaller_shape


def test_individuals_are_not_equal_unless_nparray():
    ind_with_array = Individual(10.0, np.array([1, 2, 3, 4]),
                                np.array([5, 6, 7, 8, 9, 10]))
    ind_with_list = Individual(10.0, [1, 2, 3, 4], [5, 6, 7, 8, 9, 10])
    assert ind_with_array != ind_with_list
