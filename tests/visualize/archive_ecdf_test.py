"""Tests for archive_ecdf.

See README.md for instructions on writing tests.
"""

import matplotlib.pyplot as plt
import pytest
from matplotlib.testing.decorators import image_comparison

from ribs.archives import DensityArchive
from ribs.visualize import archive_ecdf

# See https://github.com/astral-sh/ruff/issues/10662
# ruff: noqa: F401, F811
from .grid_archive_heatmap_test import (  # pylint: disable = unused-import
    grid_archive_2d,
)

# pylint: disable=redefined-outer-name


#
# Tests
#


def test_no_data_method_available():
    archive = DensityArchive(measure_dim=2)
    with pytest.raises(
        AttributeError,
        match=r"To use archive_ecdf, the archive must have the data\(\) method\.",
    ):
        archive_ecdf(archive)


@image_comparison(
    baseline_images=["ecdf"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_ecdf(grid_archive_2d):
    plt.figure(figsize=(8, 6))
    archive_ecdf(grid_archive_2d, stat="count")


@image_comparison(
    baseline_images=["eccdf"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_eccdf(grid_archive_2d):
    plt.figure(figsize=(8, 6))
    archive_ecdf(grid_archive_2d, complementary=True, stat="count")


@image_comparison(
    baseline_images=["eccdf_stretch"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_from_df(grid_archive_2d):
    df = grid_archive_2d.data(return_type="pandas")
    df["objective"] *= 2.0
    archive_ecdf(
        grid_archive_2d,
        df=df,
        complementary=True,
        stat="count",
    )


@image_comparison(
    baseline_images=["eccdf_double"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_plot_two_eccdfs(grid_archive_2d):
    df = grid_archive_2d.data(return_type="pandas")
    df["objective"] *= 2.0

    archive_ecdf(grid_archive_2d, complementary=True, stat="count")
    archive_ecdf(
        grid_archive_2d,
        df=df,
        complementary=True,
        stat="count",
    )
