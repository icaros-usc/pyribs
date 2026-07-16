"""Tests for visualization utils."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ribs.visualize._utils import compute_vmin_vmax

# pylint: disable = redefined-outer-name


@pytest.fixture
def objectives():
    """Basic objective values with min of -1 and max of 1."""
    return np.asarray([-1.0, -0.7, 0.2, 0.8, 1.0])


class TestVminVmax:
    """Tests for compute_vmin_vmax."""

    def test_no_vmin_no_vmax_good_objs(self, objectives):
        # Uses min and max objectives directly.
        assert_allclose(compute_vmin_vmax(None, None, objectives), (-1.0, 1.0))

    def test_no_vmin_no_vmax_close_objs(self):
        # Min and max objective are both 0.5.
        assert_allclose(
            compute_vmin_vmax(None, None, np.asarray([0.5, 0.5, 0.5, 0.5, 0.5])),
            (0.4, 0.6),
        )

    def test_no_vmin_no_vmax_no_objs(self):
        assert_allclose(compute_vmin_vmax(None, None, np.asarray([])), (-0.1, 0.1))

    def test_yes_vmin_no_vmax_good_objs(self, objectives):
        assert_allclose(compute_vmin_vmax(-2.0, None, objectives), (-2.0, 1.0))

    def test_yes_vmin_no_vmax_bad_objs(self, objectives):
        # 2.0 > max(objectives)
        assert_allclose(compute_vmin_vmax(2.0, None, objectives), (2.0, 2.2))

    def test_yes_vmin_no_vmax_no_objs(self):
        assert_allclose(compute_vmin_vmax(2.0, None, np.asarray([])), (2.0, 2.2))

    def test_no_vmin_yes_vmax_good_objs(self, objectives):
        assert_allclose(compute_vmin_vmax(None, 2.0, objectives), (-1.0, 2.0))

    def test_no_vmin_yes_vmax_bad_objs(self, objectives):
        # -2.0 < min(objectives)
        assert_allclose(compute_vmin_vmax(None, -2.0, objectives), (-2.2, -2.0))

    def test_no_vmin_yes_vmax_no_objs(self):
        assert_allclose(compute_vmin_vmax(None, 2.0, np.asarray([])), (1.8, 2.0))

    def test_yes_vmin_yes_vmax(self, objectives):
        assert_allclose(compute_vmin_vmax(-2.0, 2.0, objectives), (-2.0, 2.0))

    def test_yes_vmin_yes_vmax_error(self, objectives):
        """vmin and vmax invalid since vmax < vmin"""
        with pytest.raises(
            ValueError, match=r"vmax .* must be greater than or equal to vmin .*"
        ):
            compute_vmin_vmax(3, -3, objectives)
