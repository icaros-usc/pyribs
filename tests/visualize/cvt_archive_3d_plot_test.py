"""Tests for cvt_archive_3d_plot.

See README.md for instructions on writing tests.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.testing.decorators import image_comparison

from ribs.archives import CVTArchive
from ribs.visualize import cvt_archive_3d_plot

from .conftest import add_uniform_sphere_3d

# pylint: disable = redefined-outer-name

# Tolerance for root mean square difference between the pixels of the images,
# where 255 is the max value. We have a pretty high tolerance for
# `cvt_archive_3d_plot` since 3D rendering tends to vary a bit.
CVT_IMAGE_TOLERANCE = 1.0

#
# Fixtures
#


@pytest.fixture(scope="module")
def cvt_archive_3d():
    """Deterministically-created CVTArchive."""
    archive = CVTArchive(
        solution_dim=3,
        cells=500,
        ranges=np.array([(-1, 1), (-1, 1), (-1, 1)]),
        samples=10_000,
        seed=42,
    )
    add_uniform_sphere_3d(archive, (-1, 1), (-1, 1), (-1, 1))
    return archive


#
# Argument validation tests
#


def test_no_samples_error():
    # This archive has no samples since custom centroids were passed in.
    archive = CVTArchive(solution_dim=2,
                         cells=2,
                         ranges=[(-1, 1), (-1, 1)],
                         custom_centroids=[[0, 0], [1, 1]])

    # Thus, plotting samples on this archive should fail.
    with pytest.raises(ValueError):
        cvt_archive_3d_plot(archive, plot_samples=True)


#
# Tests on archive with (-1, 1) range.
#


@image_comparison(baseline_images=["3d"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_3d(cvt_archive_3d):
    plt.figure(figsize=(8, 6))
    cvt_archive_3d_plot(cvt_archive_3d)


@image_comparison(baseline_images=["3d"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_3d_custom_axis(cvt_archive_3d):
    ax = plt.axes(projection="3d")
    cvt_archive_3d_plot(cvt_archive_3d, ax=ax)


#  @image_comparison(baseline_images=["2d_long"],
#                    remove_text=False,
#                    extensions=["png"],
#                    tol=CVT_IMAGE_TOLERANCE)
#  def test_2d_long(cvt_archive_2d_long):
#      plt.figure(figsize=(8, 6))
#      cvt_archive_heatmap(cvt_archive_2d_long)

#  @image_comparison(baseline_images=["2d_long_square"],
#                    remove_text=False,
#                    extensions=["png"],
#                    tol=CVT_IMAGE_TOLERANCE)
#  def test_2d_long_square(cvt_archive_2d_long):
#      plt.figure(figsize=(8, 6))
#      cvt_archive_heatmap(cvt_archive_2d_long, aspect="equal")

#  @image_comparison(baseline_images=["2d_long_transpose"],
#                    remove_text=False,
#                    extensions=["png"],
#                    tol=CVT_IMAGE_TOLERANCE)
#  def test_2d_long_transpose(cvt_archive_2d_long):
#      plt.figure(figsize=(8, 6))
#      cvt_archive_heatmap(cvt_archive_2d_long, transpose_measures=True)

#  @image_comparison(baseline_images=["limits"],
#                    remove_text=False,
#                    extensions=["png"],
#                    tol=CVT_IMAGE_TOLERANCE)
#  def test_limits(cvt_archive_2d):
#      plt.figure(figsize=(8, 6))
#      cvt_archive_heatmap(cvt_archive_2d, vmin=-1.0, vmax=-0.5)


@image_comparison(baseline_images=["listed_cmap"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_listed_cmap(cvt_archive_3d):
    plt.figure(figsize=(8, 6))
    cvt_archive_3d_plot(cvt_archive_3d, cmap=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])


@image_comparison(baseline_images=["coolwarm_cmap"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_coolwarm_cmap(cvt_archive_3d):
    plt.figure(figsize=(8, 6))
    cvt_archive_3d_plot(cvt_archive_3d, cmap="coolwarm")


@image_comparison(baseline_images=["vmin_equals_vmax"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_vmin_equals_vmax(cvt_archive_3d):
    plt.figure(figsize=(8, 6))
    cvt_archive_3d_plot(cvt_archive_3d, vmin=-0.95, vmax=-0.95)


@image_comparison(baseline_images=["plot_centroids"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_plot_centroids(cvt_archive_3d):
    plt.figure(figsize=(8, 6))
    cvt_archive_3d_plot(cvt_archive_3d, plot_centroids=True, cell_alpha=0.1)


@image_comparison(baseline_images=["plot_samples"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_plot_samples(cvt_archive_3d):
    plt.figure(figsize=(8, 6))
    cvt_archive_3d_plot(cvt_archive_3d, plot_samples=True, cell_alpha=0.1)


@image_comparison(baseline_images=["voronoi_style"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_voronoi_style(cvt_archive_3d):
    plt.figure(figsize=(8, 6))
    cvt_archive_3d_plot(cvt_archive_3d, lw=3.0, ec="grey")


@image_comparison(baseline_images=["cell_alpha"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_cell_alpha(cvt_archive_3d):
    plt.figure(figsize=(8, 6))
    cvt_archive_3d_plot(cvt_archive_3d, cell_alpha=0.1)
