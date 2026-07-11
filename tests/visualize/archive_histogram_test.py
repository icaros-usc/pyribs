"""Tests for archive_histogram.

See README.md for instructions on writing tests.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.testing.decorators import image_comparison

from ribs.archives import DensityArchive, GridArchive
from ribs.visualize import archive_histogram, grid_archive_heatmap

# See https://github.com/astral-sh/ruff/issues/10662
# ruff: noqa: F401, F811
from .grid_archive_heatmap_test import (  # pylint: disable = unused-import
    grid_archive_3d,
)

# pylint: disable=redefined-outer-name

#
# Fixtures
#


@pytest.fixture
def archive_2d() -> GridArchive:
    """Creates a 100x100 archive and fills it with the sphere function."""
    rng = np.random.default_rng(42)
    xxs, yys = np.meshgrid(
        np.linspace(-1, 1, 100),
        np.linspace(-1, 1, 100),
    )
    xxs, yys = xxs.ravel(), yys.ravel()

    coords = np.stack((xxs, yys), axis=1)
    sphere = xxs**2 + yys**2 + 0.1 * rng.standard_normal(xxs.shape)

    archive = GridArchive(solution_dim=2, dims=[100, 100], ranges=[(-1, 1), (-1, 1)])
    archive.add(
        solution=coords,
        objective=-sphere,  # Negative sphere.
        measures=coords,
    )
    return archive


#
# Tests
#


def test_no_data_method_available():
    archive = DensityArchive(measure_dim=2)
    with pytest.raises(
        AttributeError,
        match=r"To use archive_histogram, the archive must have the data\(\) method\.",
    ):
        archive_histogram(archive)


@image_comparison(
    baseline_images=["heatmap_reference"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_heatmap_reference(archive_2d):
    """Plots a heatmap so that we get a good sense of what the archive looks like."""
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive_2d)


@image_comparison(
    baseline_images=["basic"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_basic(archive_2d):
    plt.figure(figsize=(8, 6))
    archive_histogram(archive_2d)


@image_comparison(
    baseline_images=["basic"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_basic_custom_ax(archive_2d):
    _, ax = plt.subplots(figsize=(8, 6))
    archive_histogram(archive_2d, ax=ax)


@image_comparison(
    baseline_images=["basic"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_basic_with_df(archive_2d):
    plt.figure(figsize=(8, 6))
    df = archive_2d.data(["objective"], return_type="pandas")
    archive_histogram(archive_2d, df=df)


@image_comparison(
    baseline_images=["3d_archive"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_3d_archive(grid_archive_3d):
    """Try using a 3D archive just to make sure the visualization is invariant to dimensions."""
    plt.figure(figsize=(8, 6))
    archive_histogram(grid_archive_3d)


@image_comparison(
    baseline_images=["custom_bins_int"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_custom_bins_int(archive_2d):
    plt.figure(figsize=(8, 6))
    archive_histogram(archive_2d, bins=10)


@image_comparison(
    baseline_images=["custom_bins_list"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_custom_bins_list(archive_2d):
    plt.figure(figsize=(8, 6))
    archive_histogram(archive_2d, bins=[-2, -0.5, 0.0])


@image_comparison(
    baseline_images=["vmin"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_vmin(archive_2d):
    plt.figure(figsize=(8, 6))
    archive_histogram(archive_2d, vmin=-5, vmax=None)


@image_comparison(
    baseline_images=["vmax"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_vmax(archive_2d):
    plt.figure(figsize=(8, 6))
    archive_histogram(archive_2d, vmin=None, vmax=5)


@image_comparison(
    baseline_images=["vmin_vmax"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_vmin_vmax(archive_2d):
    plt.figure(figsize=(8, 6))
    archive_histogram(archive_2d, vmin=-5, vmax=-1)


@image_comparison(
    baseline_images=["ylim"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_ylim(archive_2d):
    plt.figure(figsize=(8, 6))
    archive_histogram(archive_2d, ylim=300)


@image_comparison(
    baseline_images=["color"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_color(archive_2d):
    plt.figure(figsize=(8, 6))
    archive_histogram(archive_2d, color="red")


@image_comparison(
    baseline_images=["cmap"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_cmap(archive_2d):
    plt.figure(figsize=(8, 6))
    archive_histogram(archive_2d, cmap="magma")


@image_comparison(
    baseline_images=["cmap"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_cmap_overrides_color(archive_2d):
    plt.figure(figsize=(8, 6))
    archive_histogram(archive_2d, color="red", cmap="magma")


@image_comparison(
    baseline_images=["rasterized"],
    remove_text=False,
    extensions=["pdf"],
    style="mpl20",
)
def test_rasterized(archive_2d):
    plt.figure(figsize=(8, 6))
    archive_histogram(archive_2d, rasterized=True)


@image_comparison(
    baseline_images=["integration"],
    remove_text=False,
    extensions=["pdf"],
    style="mpl20",
)
def test_integration(archive_2d):
    """Test combination of a couple of the features above."""
    plt.figure(figsize=(8, 6))
    archive_histogram(
        archive_2d,
        bins=50,
        vmin=-2.5,
        vmax=0.5,
        ylim=500,
        cmap="magma",
        rasterized=True,
    )


@image_comparison(
    baseline_images=["hist_kwargs"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_hist_kwargs(archive_2d):
    plt.figure(figsize=(8, 6))
    archive_histogram(archive_2d, bins=20, hist_kwargs={"edgecolor": "red"})
