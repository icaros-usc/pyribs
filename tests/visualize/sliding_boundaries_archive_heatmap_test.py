"""Tests for sliding_boundaries_archive_heatmap.

See README.md for instructions on writing tests.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.testing.decorators import image_comparison

from ribs.archives import SlidingBoundariesArchive
from ribs.visualize import sliding_boundaries_archive_heatmap

#
# Fixtures
#


def add_random_sphere(archive, x_range, y_range):
    """Adds 1000 random points from the negative sphere function.

    Solutions, measures, and ranges are same as in add_uniform_sphere.
    """
    # Use random measures to make the boundaries shift.
    rng = np.random.default_rng(10)
    solutions = rng.uniform(
        (x_range[0], y_range[0]),
        (x_range[1], y_range[1]),
        (1000, 2),
    )
    sphere = np.sum(np.square(solutions), axis=1)
    archive.add(
        solution=solutions,
        objective=-sphere,
        measures=solutions,
    )


@pytest.fixture(scope="module")
def sliding_archive_2d():
    """Deterministically-created SlidingBoundariesArchive."""
    archive = SlidingBoundariesArchive(
        solution_dim=2, dims=[10, 20], ranges=[(-1, 1), (-1, 1)], seed=42
    )
    add_random_sphere(archive, (-1, 1), (-1, 1))
    return archive


@pytest.fixture(scope="module")
def sliding_archive_2d_empty():
    """Same as above but without solutions."""
    archive = SlidingBoundariesArchive(
        solution_dim=2, dims=[10, 20], ranges=[(-1, 1), (-1, 1)], seed=42
    )
    return archive


@pytest.fixture(scope="module")
def sliding_archive_2d_long():
    """Same as above, but the measure space is longer in one direction."""
    archive = SlidingBoundariesArchive(
        solution_dim=2, dims=[10, 20], ranges=[(-2, 2), (-1, 1)], seed=42
    )
    add_random_sphere(archive, (-2, 2), (-1, 1))
    return archive


#
# Tests
#


@image_comparison(baseline_images=["2d"], remove_text=False, extensions=["png"])
def test_2d(sliding_archive_2d):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(sliding_archive_2d)


@image_comparison(baseline_images=["2d"], remove_text=False, extensions=["png"])
def test_2d_custom_axis(sliding_archive_2d):
    _, ax = plt.subplots(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(sliding_archive_2d, ax=ax)


@image_comparison(baseline_images=["2d_long"], remove_text=False, extensions=["png"])
def test_2d_long(sliding_archive_2d_long):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(sliding_archive_2d_long)


@image_comparison(
    baseline_images=["2d_long_square"], remove_text=False, extensions=["png"]
)
def test_2d_long_square(sliding_archive_2d_long):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(sliding_archive_2d_long, aspect="equal")


@image_comparison(
    baseline_images=["2d_long_transpose"], remove_text=False, extensions=["png"]
)
def test_2d_long_transpose(sliding_archive_2d_long):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(sliding_archive_2d_long, transpose_measures=True)


@image_comparison(baseline_images=["limits"], remove_text=False, extensions=["png"])
def test_limits(sliding_archive_2d):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(sliding_archive_2d, vmin=-1.0, vmax=-0.5)


@image_comparison(
    baseline_images=["limits_when_empty"], remove_text=False, extensions=["png"]
)
def test_limits_when_empty(sliding_archive_2d_empty):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(
        sliding_archive_2d_empty,
        # Intentionally don't provide vmin or vmax.
        vmin=None,
        vmax=None,
    )


@image_comparison(
    baseline_images=["listed_cmap"], remove_text=False, extensions=["png"]
)
def test_listed_cmap(sliding_archive_2d):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(
        sliding_archive_2d, cmap=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )


@image_comparison(
    baseline_images=["coolwarm_cmap"], remove_text=False, extensions=["png"]
)
def test_coolwarm_cmap(sliding_archive_2d):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(sliding_archive_2d, cmap="coolwarm")


@image_comparison(baseline_images=["boundaries"], remove_text=False, extensions=["png"])
def test_boundaries(sliding_archive_2d):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(sliding_archive_2d, boundary_lw=0.5)


@image_comparison(
    baseline_images=["mismatch_xy_with_boundaries"],
    remove_text=False,
    extensions=["png"],
)
def test_mismatch_xy_with_boundaries():
    """There was a bug caused by the boundary lines being assigned incorrectly.

    https://github.com/icaros-usc/pyribs/issues/270
    """
    archive = SlidingBoundariesArchive(
        solution_dim=2, dims=[10, 20], ranges=[(-1, 1), (-2, 2)], seed=42
    )
    add_random_sphere(archive, (-1, 1), (-2, 2))
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(archive, boundary_lw=0.5)


@image_comparison(baseline_images=["rasterized"], remove_text=False, extensions=["pdf"])
def test_rasterized(sliding_archive_2d):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(
        sliding_archive_2d, boundary_lw=1.0, rasterized=True
    )


@image_comparison(
    baseline_images=["plot_with_df"], remove_text=False, extensions=["png"]
)
def test_plot_with_df(sliding_archive_2d):
    plt.figure(figsize=(8, 6))
    df = sliding_archive_2d.data(return_type="pandas")
    df["objective"] = -df["objective"]
    sliding_boundaries_archive_heatmap(sliding_archive_2d, df=df)
