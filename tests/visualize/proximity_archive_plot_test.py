"""Tests for proximity_archive_plot.

See README.md for instructions on writing tests.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.testing.decorators import image_comparison

from ribs.archives import ProximityArchive
from ribs.visualize import proximity_archive_plot

# pylint: disable = redefined-outer-name

#
# Fixtures
#


def add_random_sphere(archive, x_range, y_range, use_objective=False):
    """Adds 1000 random points from the negative sphere function.

    Solutions, measures, and ranges are same as in add_uniform_sphere.
    """
    # Use random measures to make the boundaries shift.
    rng = np.random.default_rng(10)

    # Add over separate iterations so that novelty is actually considered.
    for _ in range(10):
        solutions = rng.uniform(
            (x_range[0], y_range[0]),
            (x_range[1], y_range[1]),
            (100, 2),
        )
        objective = (-np.sum(np.square(solutions), axis=1)
                     if use_objective else None)
        archive.add(solution=solutions, objective=objective, measures=solutions)


@pytest.fixture(scope="module")
def proximity_archive_2d():
    """Deterministically-created ProximityArchive."""
    archive = ProximityArchive(solution_dim=2,
                               measure_dim=2,
                               k_neighbors=5,
                               novelty_threshold=0.1,
                               seed=42)
    add_random_sphere(archive, (-1, 1), (-1, 1), use_objective=False)
    return archive


@pytest.fixture(scope="module")
def proximity_archive_2d_obj():
    """Same as above but with objectives."""
    archive = ProximityArchive(solution_dim=2,
                               measure_dim=2,
                               k_neighbors=5,
                               novelty_threshold=0.1,
                               seed=42)
    add_random_sphere(archive, (-1, 1), (-1, 1), use_objective=True)
    return archive


@pytest.fixture(scope="module")
def proximity_archive_2d_empty():
    """Same as above but without solutions."""
    archive = ProximityArchive(solution_dim=2,
                               measure_dim=2,
                               k_neighbors=5,
                               novelty_threshold=0.1,
                               seed=42)
    return archive


@pytest.fixture(scope="module")
def proximity_archive_2d_long():
    """Same as above, but the measure space is longer in one direction."""
    archive = ProximityArchive(solution_dim=2,
                               measure_dim=2,
                               k_neighbors=5,
                               novelty_threshold=0.1,
                               seed=42)
    add_random_sphere(archive, (-2, 2), (-1, 1), use_objective=False)
    return archive


#
# Tests
#


@image_comparison(baseline_images=["2d"], remove_text=False, extensions=["png"])
def test_2d(proximity_archive_2d):
    plt.figure(figsize=(8, 6))
    proximity_archive_plot(proximity_archive_2d,
                           cmap=[[0.5, 0.5, 0.5]],
                           cbar=None)


@image_comparison(baseline_images=["2d"], remove_text=False, extensions=["png"])
def test_2d_custom_axis(proximity_archive_2d):
    _, ax = plt.subplots(figsize=(8, 6))
    proximity_archive_plot(proximity_archive_2d,
                           ax=ax,
                           cmap=[[0.5, 0.5, 0.5]],
                           cbar=None)


@image_comparison(baseline_images=["2d_long"],
                  remove_text=False,
                  extensions=["png"])
def test_2d_long(proximity_archive_2d_long):
    plt.figure(figsize=(8, 6))
    proximity_archive_plot(proximity_archive_2d_long,
                           cmap=[[0.5, 0.5, 0.5]],
                           cbar=None)


@image_comparison(baseline_images=["2d_long_square"],
                  remove_text=False,
                  extensions=["png"])
def test_2d_long_square(proximity_archive_2d_long):
    plt.figure(figsize=(8, 6))
    proximity_archive_plot(proximity_archive_2d_long,
                           aspect="equal",
                           cmap=[[0.5, 0.5, 0.5]],
                           cbar=None)


@image_comparison(baseline_images=["2d_long_transpose"],
                  remove_text=False,
                  extensions=["png"])
def test_2d_long_transpose(proximity_archive_2d_long):
    plt.figure(figsize=(8, 6))
    proximity_archive_plot(proximity_archive_2d_long,
                           transpose_measures=True,
                           cmap=[[0.5, 0.5, 0.5]],
                           cbar=None)


@image_comparison(baseline_images=["bounds"],
                  remove_text=False,
                  extensions=["png"])
def test_bounds(proximity_archive_2d):
    plt.figure(figsize=(8, 6))
    proximity_archive_plot(proximity_archive_2d,
                           lower_bounds=[-5, -5],
                           upper_bounds=[5, 5],
                           cmap=[[0.5, 0.5, 0.5]],
                           cbar=None)


@image_comparison(baseline_images=["limits"],
                  remove_text=False,
                  extensions=["png"])
def test_limits(proximity_archive_2d_obj):
    # Negative sphere function should have range (-2, 0). These limits should
    # give a more uniform-looking archive.
    plt.figure(figsize=(8, 6))
    proximity_archive_plot(proximity_archive_2d_obj, vmin=-1.0, vmax=-0.5)


@image_comparison(baseline_images=["bounds_and_limits_when_empty"],
                  remove_text=False,
                  extensions=["png"])
def test_bounds_and_limits_when_empty(proximity_archive_2d_empty):
    plt.figure(figsize=(8, 6))
    proximity_archive_plot(
        proximity_archive_2d_empty,
        # Intentionally don't provide vmin or vmax.
        vmin=None,
        vmax=None,
    )


@image_comparison(baseline_images=["rasterized"],
                  remove_text=False,
                  extensions=["pdf"])
def test_rasterized(proximity_archive_2d):
    plt.figure(figsize=(8, 6))
    proximity_archive_plot(proximity_archive_2d, rasterized=True)
