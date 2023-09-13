"""Tests for cvt_archive_heatmap.

See README.md for instructions on writing tests.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest
import shapely
from matplotlib.testing.decorators import image_comparison

from ribs.archives import CVTArchive
from ribs.visualize import cvt_archive_heatmap

from .conftest import add_uniform_sphere_1d, add_uniform_sphere_2d

# pylint: disable = redefined-outer-name

# Tolerance for root mean square difference between the pixels of the images,
# where 255 is the max value. We only have tolerance for `cvt_archive_heatmap`
# since it is a bit more finicky than the other plots.
CVT_IMAGE_TOLERANCE = 0.1

#
# Fixtures
#


@pytest.fixture(scope="module")
def cvt_archive_1d():
    """Deterministically-created CVTArchive with 1 measure."""
    # The archive must be low-res enough that we can tell if the number of cells
    # is correct, yet high-res enough that we can see different colors.

    # The centroids are chosen in a descending order so that their indices do
    # not line up with the indices in the heatmap.
    centroids = np.array(
        [0.95, 0.7, 0.35, 0.3, 0.0, -0.1, -0.3, -0.5, -0.8, -0.9])

    archive = CVTArchive(
        solution_dim=1,
        cells=10,
        ranges=[(-1, 1)],
        seed=42,
        custom_centroids=centroids[:, None],
    )

    # Add with gaps -- this way, some cells are left unoccupied so that we can
    # check unoccupied cells.
    archive.add(
        np.zeros((8, 1)),
        np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        np.array([-0.9, -0.8, -0.3, -0.1, 0.0, 0.3, 0.35, 0.95])[:, None],
    )

    return archive


@pytest.fixture(scope="module")
def cvt_archive_2d():
    """Deterministically-created CVTArchive."""
    archive = CVTArchive(solution_dim=2,
                         cells=100,
                         ranges=[(-1, 1), (-1, 1)],
                         samples=1000,
                         use_kd_tree=True,
                         seed=42)
    add_uniform_sphere_2d(archive, (-1, 1), (-1, 1))
    return archive


@pytest.fixture(scope="module")
def cvt_archive_2d_long():
    """Same as above, but the measure space is longer in one direction."""
    archive = CVTArchive(solution_dim=2,
                         cells=100,
                         ranges=[(-2, 2), (-1, 1)],
                         samples=1000,
                         use_kd_tree=True,
                         seed=42)
    add_uniform_sphere_2d(archive, (-2, 2), (-1, 1))
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
        cvt_archive_heatmap(archive, lw=3.0, ec="grey", plot_samples=True)


#
# 2D tests
#


@image_comparison(baseline_images=["2d"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_2d(cvt_archive_2d):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive_2d)


@image_comparison(baseline_images=["2d"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_2d_custom_axis(cvt_archive_2d):
    _, ax = plt.subplots(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive_2d, ax=ax)


@image_comparison(baseline_images=["2d_long"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_heatmap_long__cvt(cvt_archive_2d_long):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive_2d_long)


@image_comparison(baseline_images=["2d_long_square"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_2d_long_square(cvt_archive_2d_long):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive_2d_long, aspect="equal")


@image_comparison(baseline_images=["2d_long_transpose"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_2d_long_transpose(cvt_archive_2d_long):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive_2d_long, transpose_measures=True)


@image_comparison(baseline_images=["limits"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_limits(cvt_archive_2d):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive_2d, vmin=-1.0, vmax=-0.5)


@image_comparison(baseline_images=["listed_cmap"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_listed_cmap(cvt_archive_2d):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive_2d, cmap=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])


@image_comparison(baseline_images=["coolwarm_cmap"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_coolwarm_cmap(cvt_archive_2d):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive_2d, cmap="coolwarm")


@image_comparison(baseline_images=["vmin_equals_vmax"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_vmin_equals_vmax(cvt_archive_2d):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive_2d, vmin=-0.5, vmax=-0.5)


@image_comparison(baseline_images=["plot_centroids"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_plot_centroids(cvt_archive_2d):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive_2d, plot_centroids=True)


@image_comparison(baseline_images=["plot_samples"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_plot_samples(cvt_archive_2d):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive_2d, plot_samples=True)


@image_comparison(baseline_images=["voronoi_style"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_voronoi_style(cvt_archive_2d):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive_2d, lw=3.0, ec="grey")


@image_comparison(baseline_images=["rasterized"],
                  remove_text=False,
                  extensions=["pdf"])
def test_rasterized(cvt_archive_2d):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive_2d, rasterized=True)


#
# Tests for `clip` parameter
#


@image_comparison(baseline_images=["noclip"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_noclip(cvt_archive_2d):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive_2d, clip=False)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)


@image_comparison(baseline_images=["clip"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_clip(cvt_archive_2d):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive_2d, clip=True)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)


@image_comparison(baseline_images=["clip_polygon"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_clip_polygon(cvt_archive_2d):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(
        cvt_archive_2d,
        clip=shapely.Polygon(shell=np.array([
            [-0.75, -0.375],
            [-0.75, 0.375],
            [-0.375, 0.75],
            [0.375, 0.75],
            [0.75, 0.375],
            [0.75, -0.375],
            [0.375, -0.75],
            [-0.375, -0.75],
        ]),),
    )
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)


@image_comparison(baseline_images=["clip_polygon_with_hole"],
                  remove_text=False,
                  extensions=["png"],
                  tol=CVT_IMAGE_TOLERANCE)
def test_clip_polygon_with_hole(cvt_archive_2d):
    """This test will force some cells to be split in two."""
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(
        cvt_archive_2d,
        clip=shapely.Polygon(
            shell=np.array([
                [-0.75, -0.375],
                [-0.75, 0.375],
                [-0.375, 0.75],
                [0.375, 0.75],
                [0.75, 0.375],
                [0.75, -0.375],
                [0.375, -0.75],
                [-0.375, -0.75],
            ]),
            holes=[
                # Two holes that split some cells into two parts, and some cells
                # into three parts.
                np.array([
                    [-0.5, 0],
                    [-0.5, 0.05],
                    [0.5, 0.05],
                    [0.5, 0],
                ]),
                np.array([
                    [-0.5, 0.125],
                    [-0.5, 0.175],
                    [0.5, 0.175],
                    [0.5, 0.125],
                ]),
            ],
        ),
    )
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)


#
# 1D tests
#


@image_comparison(baseline_images=["1d"], remove_text=False, extensions=["png"])
def test_1d(cvt_archive_1d):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive_1d)


@image_comparison(baseline_images=["1d_style"],
                  remove_text=False,
                  extensions=["png"])
def test_1d_style(cvt_archive_1d):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive_1d, lw=3.0, ec="grey")


@image_comparison(baseline_images=["1d_with_points"],
                  remove_text=False,
                  extensions=["png"])
def test_1d_with_points():
    """Adds in centroids and samples to the plot."""
    archive = CVTArchive(
        solution_dim=1,
        cells=10,
        ranges=[(-1, 1)],
        seed=42,
        samples=100,
    )
    add_uniform_sphere_1d(archive, (-1, 1))

    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(archive,
                        plot_centroids=True,
                        plot_samples=True,
                        ms=10.0)


@image_comparison(baseline_images=["1d_aspect_greater_than_1"],
                  remove_text=False,
                  extensions=["png"])
def test_1d_aspect_greater_than_1(cvt_archive_1d):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive_1d, aspect=2.5)


@image_comparison(baseline_images=["1d_aspect_less_than_1"],
                  remove_text=False,
                  extensions=["png"])
def test_1d_aspect_less_than_1(cvt_archive_1d):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive_1d, aspect=0.1)


@image_comparison(baseline_images=["1d_sphere"],
                  remove_text=False,
                  extensions=["png"])
def test_1d_sphere():
    """More complex setting."""
    archive = CVTArchive(
        solution_dim=1,
        cells=20,
        ranges=[(-1, 1)],
        seed=42,
        samples=100,
    )
    add_uniform_sphere_1d(archive, (-1, 1))

    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(archive)
