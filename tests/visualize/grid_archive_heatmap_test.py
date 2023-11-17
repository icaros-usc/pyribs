"""Tests for grid_archive_heatmap.

See README.md for instructions on writing tests.
"""
import matplotlib.pyplot as plt
import pytest
from matplotlib.testing.decorators import image_comparison

from ribs.archives import GridArchive
from ribs.visualize import grid_archive_heatmap

from .conftest import (add_uniform_sphere_1d, add_uniform_sphere_2d,
                       add_uniform_sphere_3d)

# pylint: disable = redefined-outer-name

#
# Fixtures
#


@pytest.fixture(scope="module")
def grid_archive_1d():
    """Deterministically-created GridArchive with 1 measure."""
    # The archive must be low-res enough that we can tell if the number of cells
    # is correct, yet high-res enough that we can see different colors.
    archive = GridArchive(solution_dim=1, dims=[10], ranges=[(-1, 1)], seed=42)
    add_uniform_sphere_1d(archive, (-1, 1))
    return archive


@pytest.fixture(scope="module")
def grid_archive_2d():
    """Deterministically-created GridArchive."""
    # The archive must be low-res enough that we can tell if the number of cells
    # is correct, yet high-res enough that we can see different colors.
    archive = GridArchive(solution_dim=2,
                          dims=[10, 10],
                          ranges=[(-1, 1), (-1, 1)],
                          seed=42)
    add_uniform_sphere_2d(archive, (-1, 1), (-1, 1))
    return archive


@pytest.fixture(scope="module")
def grid_archive_2d_long():
    """Same as above, but the measure space is longer in one direction."""
    archive = GridArchive(solution_dim=2,
                          dims=[10, 10],
                          ranges=[(-2, 2), (-1, 1)],
                          seed=42)
    add_uniform_sphere_2d(archive, (-2, 2), (-1, 1))
    return archive


@pytest.fixture(scope="module")
def grid_archive_3d():
    """Deterministic archive, but there are three measure axes of different
    sizes, and some of the axes are not totally filled."""
    archive = GridArchive(solution_dim=3,
                          dims=[10, 10, 10],
                          ranges=[(-2, 2), (-1, 1), (-2, 1)],
                          seed=42)
    add_uniform_sphere_3d(archive, (0, 2), (-1, 1), (-1, 0))
    return archive


#
# 2D tests
#


@image_comparison(baseline_images=["2d"], remove_text=False, extensions=["png"])
def test_2d(grid_archive_2d):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive_2d)


@image_comparison(baseline_images=["2d"], remove_text=False, extensions=["png"])
def test_2d_custom_axis(grid_archive_2d):
    _, ax = plt.subplots(figsize=(8, 6))
    grid_archive_heatmap(grid_archive_2d, ax=ax)


@image_comparison(baseline_images=["2d_long"],
                  remove_text=False,
                  extensions=["png"])
def test_2d_long(grid_archive_2d_long):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive_2d_long)


@image_comparison(baseline_images=["2d_long_square"],
                  remove_text=False,
                  extensions=["png"])
def test_2d_long_square(grid_archive_2d_long):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive_2d_long, aspect="equal")


@image_comparison(baseline_images=["2d_long_transpose"],
                  remove_text=False,
                  extensions=["png"])
def test_2d_long_transpose(grid_archive_2d_long):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive_2d_long, transpose_measures=True)


@image_comparison(baseline_images=["limits"],
                  remove_text=False,
                  extensions=["png"])
def test_limits(grid_archive_2d):
    # Negative sphere function should have range (-2, 0). These limits should
    # give a more uniform-looking archive.
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive_2d, vmin=-1.0, vmax=-0.5)


@image_comparison(baseline_images=["listed_cmap"],
                  remove_text=False,
                  extensions=["png"])
def test_listed_cmap(grid_archive_2d):
    # cmap consists of primary red, green, and blue.
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive_2d,
                         cmap=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])


@image_comparison(baseline_images=["coolwarm_cmap"],
                  remove_text=False,
                  extensions=["png"])
def test_coolwarm_cmap(grid_archive_2d):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive_2d, cmap="coolwarm")


@image_comparison(baseline_images=["boundaries"],
                  remove_text=False,
                  extensions=["png"])
def test_boundaries(grid_archive_2d):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive_2d,
                         pcm_kwargs={
                             "edgecolor": "black",
                             "linewidth": 0.1
                         })


@image_comparison(baseline_images=["equal_aspect"],
                  remove_text=False,
                  extensions=["png"])
def test_equal_aspect(grid_archive_2d):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive_2d, aspect="equal")


@image_comparison(baseline_images=["aspect_greater_than_1"],
                  remove_text=False,
                  extensions=["png"])
def test_aspect_greater_than_1(grid_archive_2d):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive_2d, aspect=2.5)


@image_comparison(baseline_images=["aspect_less_than_1"],
                  remove_text=False,
                  extensions=["png"])
def test_aspect_less_than_1(grid_archive_2d):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive_2d, aspect=0.5)


@image_comparison(baseline_images=["no_cbar"],
                  remove_text=False,
                  extensions=["png"])
def test_no_cbar(grid_archive_2d):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive_2d, cbar=None)


@image_comparison(baseline_images=["custom_cbar_axis"],
                  remove_text=False,
                  extensions=["png"])
def test_custom_cbar_axis(grid_archive_2d):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
    grid_archive_heatmap(grid_archive_2d, ax=ax1, cbar=ax2)


@image_comparison(baseline_images=["rasterized"],
                  remove_text=False,
                  extensions=["pdf"])
def test_rasterized(grid_archive_2d):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive_2d, rasterized=True)


@image_comparison(baseline_images=["plot_with_df"],
                  remove_text=False,
                  extensions=["png"])
def test_plot_with_df(grid_archive_2d):
    plt.figure(figsize=(8, 6))
    df = grid_archive_2d.data(return_type="pandas")
    df["objective"] = -df["objective"]
    grid_archive_heatmap(grid_archive_2d, df=df)


#
# 1D tests
#


@image_comparison(baseline_images=["1d"], remove_text=False, extensions=["png"])
def test_1d(grid_archive_1d):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive_1d)


@image_comparison(baseline_images=["1d_aspect_greater_than_1"],
                  remove_text=False,
                  extensions=["png"])
def test_1d_aspect_greater_than_1(grid_archive_1d):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive_1d, aspect=2.5)


@image_comparison(baseline_images=["1d_aspect_less_than_1"],
                  remove_text=False,
                  extensions=["png"])
def test_1d_aspect_less_than_1(grid_archive_1d):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive_1d, aspect=0.1)
