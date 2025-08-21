"""Tests for parallel_axes_plot.

See README.md for instructions on writing tests.
"""

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison

from ribs.visualize import parallel_axes_plot

# See https://github.com/astral-sh/ruff/issues/10662
# ruff: noqa: F401, F811
from .grid_archive_heatmap_test import (
    grid_archive_2d,
    grid_archive_3d,
    grid_archive_3d_empty,
)

# pylint: disable = redefined-outer-name


@image_comparison(baseline_images=["2d"], remove_text=False, extensions=["png"])
def test_2d(grid_archive_2d):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(grid_archive_2d)


@image_comparison(baseline_images=["3d"], remove_text=False, extensions=["png"])
def test_3d(grid_archive_3d):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(grid_archive_3d)


@image_comparison(baseline_images=["3d"], remove_text=False, extensions=["png"])
def test_3d_custom_ax(grid_archive_3d):
    _, ax = plt.subplots(figsize=(8, 6))
    parallel_axes_plot(grid_archive_3d, ax=ax)


@image_comparison(
    baseline_images=["3d_custom_order"], remove_text=False, extensions=["png"]
)
def test_3d_custom_order(grid_archive_3d):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(grid_archive_3d, measure_order=[1, 2, 0])


@image_comparison(
    baseline_images=["3d_custom_names"], remove_text=False, extensions=["png"]
)
def test_3d_custom_names(grid_archive_3d):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(
        grid_archive_3d, measure_order=[(1, "One"), (2, "Two"), (0, "Zero")]
    )


@image_comparison(
    baseline_images=["3d_coolwarm"], remove_text=False, extensions=["png"]
)
def test_3d_coolwarm_cmap(grid_archive_3d):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(grid_archive_3d, cmap="coolwarm")


@image_comparison(
    baseline_images=["3d_width2_alpha2"], remove_text=False, extensions=["png"]
)
def test_3d_width2_alpha2(grid_archive_3d):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(grid_archive_3d, linewidth=2.0, alpha=0.2)


@image_comparison(
    baseline_images=["3d_custom_objective_limits"],
    remove_text=False,
    extensions=["png"],
)
def test_3d_custom_objective_limits(grid_archive_3d):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(grid_archive_3d, vmin=-2.0, vmax=-1.0)


@image_comparison(
    baseline_images=["3d_limits_when_empty"], remove_text=False, extensions=["png"]
)
def test_3d_limits_when_empty(grid_archive_3d_empty):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(grid_archive_3d_empty, vmin=None, vmax=None)


@image_comparison(
    baseline_images=["3d_sorted"],
    remove_text=False,
    extensions=["png"],
    # This image seems to have tiny differences for some reason, so make the
    # tolerance a bit higher.
    tol=1.0,
)
def test_3d_sorted(grid_archive_3d):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(grid_archive_3d, sort_archive=True)


@image_comparison(
    baseline_images=["3d_vertical_cbar"], remove_text=False, extensions=["png"]
)
def test_3d_vertical_cbar(grid_archive_3d):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(grid_archive_3d, cbar_kwargs={"orientation": "vertical"})


@image_comparison(
    baseline_images=["plot_with_df"], remove_text=False, extensions=["png"]
)
def test_plot_with_df(grid_archive_3d):
    plt.figure(figsize=(8, 6))
    df = grid_archive_3d.data(return_type="pandas")
    df["objective"] = -df["objective"]
    parallel_axes_plot(grid_archive_3d, df=df)
