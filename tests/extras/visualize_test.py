"""Tests for ribs.visualize.

If you are validating the image output from matplotlib, first read these
instructions:
https://matplotlib.org/3.3.1/devel/testing.html#writing-an-image-comparison-test.
Essentially, after running your new test for the first time (and assuming you
are running in the root directory of this repo), you will need to copy the image
output from result_images/visualize_test to
tests/extras/baseline_images/visualize_test. For instance, for
``test_cvt_archive_heatmap``, you will need to run::

    cp result_images/visualize_test/cvt_archive_heatmap.png \
        tests/extras/baseline_images/visualize_test/cvt_archive_heatmap.png

Assuming your output is what you expected (and assuming you have made your code
deterministic), the test should now pass when you re-run it.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.testing.decorators import image_comparison

from ribs.archives import CVTArchive, SlidingBoundaryArchive
from ribs.visualize import cvt_archive_heatmap, sliding_boundary_archive_heatmap

# pylint: disable = invalid-name


@pytest.fixture(autouse=True)
def _clean_matplotlib():
    """Cleans up matplotlib figures before and after each test."""
    # Before the test.
    plt.close("all")

    yield

    # After the test.
    plt.close("all")


@pytest.fixture(scope="module")  # Only run once to save time.
def _sliding_boundary_archive():
    """Deterministically created SlidingBoundaryArchive."""

    archive = SlidingBoundaryArchive([10, 20], [(-1, 1), (-1, 1)], seed=42)
    archive.initialize(solution_dim=2)

    rng = np.random.default_rng(10)
    for _ in range(1000):
        x, y = rng.uniform((-1, -1), (1, 1))
        archive.add(
            solution=rng.random(2),
            objective_value=-(x**2 + y**2),
            behavior_values=np.array([x, y]),
        )
    return archive


@pytest.fixture(scope="module")  # Only run once to save time.
def _long_sliding_boundary_archive():
    """Deterministically created SlidingBoundaryArchive."""

    archive = SlidingBoundaryArchive([10, 20], [(-2, 2), (-1, 1)], seed=42)
    archive.initialize(solution_dim=2)

    rng = np.random.default_rng(10)
    for _ in range(1000):
        x, y = rng.uniform((-2, -1), (2, 1))
        archive.add(
            solution=rng.random(2),
            objective_value=-(x**2 + y**2),
            behavior_values=np.array([x, y]),
        )
    return archive


@pytest.fixture(scope="module")  # Only run once to save time.
def _cvt_archive():
    """Deterministically created CVTArchive."""
    archive = CVTArchive([(-1, 1), (-1, 1)],
                         100,
                         samples=1000,
                         use_kd_tree=True,
                         seed=42)
    archive.initialize(solution_dim=2)

    # Add solutions.
    for x in np.linspace(-1, 1, 100):
        for y in np.linspace(-1, 1, 100):
            archive.add(
                solution=np.array([x, y]),
                # Objective is the negative sphere function.
                objective_value=-(x**2 + y**2),
                behavior_values=np.array([x, y]),
            )
    return archive


@pytest.fixture(scope="module")  # Only run once to save time.
def _long_cvt_archive():
    """Same as above, but the behavior space is longer in one direction."""
    archive = CVTArchive([(-2, 2), (-1, 1)],
                         100,
                         samples=1000,
                         use_kd_tree=True,
                         seed=42)
    archive.initialize(solution_dim=2)

    # Add solutions.
    for x in np.linspace(-2, 2, 100):
        for y in np.linspace(-1, 1, 100):
            archive.add(
                solution=np.array([x, y]),
                # Objective is the negative sphere function.
                objective_value=-(x**2 + y**2),
                behavior_values=np.array([x, y]),
            )
    return archive


@image_comparison(baseline_images=["sliding_boundary_heatmap"],
                  remove_text=False,
                  extensions=["png"])
def test_sliding_boundary_archive(_sliding_boundary_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundary_archive_heatmap(_sliding_boundary_archive)


@image_comparison(baseline_images=["sliding_boundary_heatmap_long"],
                  remove_text=False,
                  extensions=["png"])
def test_sliding_boundary_long(_long_sliding_boundary_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundary_archive_heatmap(_long_sliding_boundary_archive)


@image_comparison(baseline_images=["sliding_boundary_heatmap_long_square"],
                  remove_text=False,
                  extensions=["png"])
def test_sliding_boundary_long_square(_long_sliding_boundary_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundary_archive_heatmap(_long_sliding_boundary_archive,
                                     square=True)


@image_comparison(baseline_images=["sliding_boundary_heatmap_long_transpose"],
                  remove_text=False,
                  extensions=["png"])
def test_sliding_boundary_long_transpose(_long_sliding_boundary_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundary_archive_heatmap(_long_sliding_boundary_archive,
                                     transpose_bcs=True)



@image_comparison(baseline_images=["sliding_boundary_heatmap_with_listed_cmap"],
                  remove_text=False,
                  extensions=["png"])
def test_sliding_boundary_heatmap_with_listed_cmap(_sliding_boundary_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundary_archive_heatmap(_sliding_boundary_archive,
                                     cmap=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])


@image_comparison(baseline_images=["sliding_boundary_heatmap_with_limits"],
                  remove_text=False,
                  extensions=["png"])
def test_sliding_boundary_heatmap_with_limits(_sliding_boundary_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundary_archive_heatmap(_sliding_boundary_archive, vmin=-1.0, vmax=-0.5)


@image_comparison(baseline_images=["sliding_boundary_heatmap_with_coolwarm_cmap"],
                  remove_text=False,
                  extensions=["png"])
def test_sliding_boundary_heatmap_with_coolwarm_cmap(_sliding_boundary_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundary_archive_heatmap(_sliding_boundary_archive, cmap=matplotlib.cm.get_cmap("coolwarm"))


def test_cvt_archive_heatmap_fails_on_non_2d():
    archive = CVTArchive([(-1, 1), (-1, 1), (-1, 1)],
                         100,
                         use_kd_tree=True,
                         samples=100)
    archive.initialize(solution_dim=3)

    with pytest.raises(ValueError):
        cvt_archive_heatmap(archive)


@image_comparison(baseline_images=["cvt_archive_heatmap"],
                  remove_text=False,
                  extensions=["png"])
def test_cvt_archive_heatmap(_cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(_cvt_archive)


@image_comparison(baseline_images=["cvt_archive_heatmap"],
                  remove_text=False,
                  extensions=["png"])
def test_cvt_archive_heatmap_with_custom_axis(_cvt_archive):
    _, ax = plt.subplots(figsize=(8, 6))
    cvt_archive_heatmap(_cvt_archive, ax=ax)


@image_comparison(baseline_images=["cvt_archive_heatmap_with_coolwarm_cmap"],
                  remove_text=False,
                  extensions=["png"])
def test_cvt_archive_heatmap_with_coolwarm_cmap(_cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(_cvt_archive, cmap=matplotlib.cm.get_cmap("coolwarm"))


@image_comparison(baseline_images=["cvt_archive_heatmap_with_listed_cmap"],
                  remove_text=False,
                  extensions=["png"])
def test_cvt_archive_heatmap_with_listed_cmap(_cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(_cvt_archive, cmap=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])


@image_comparison(baseline_images=["cvt_archive_heatmap_with_samples"],
                  remove_text=False,
                  extensions=["png"])
def test_cvt_archive_heatmap_with_samples(_cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(_cvt_archive, plot_samples=True)


@image_comparison(baseline_images=["cvt_archive_heatmap_with_limits"],
                  remove_text=False,
                  extensions=["png"])
def test_cvt_archive_heatmap_with_limits(_cvt_archive):
    plt.figure(figsize=(8, 6))
    # Negative sphere function in the _cvt_archive should have range
    # (-2, 0). These limits should give us a very uniform-looking archive.
    cvt_archive_heatmap(_cvt_archive, vmin=-1.0, vmax=-0.5)


@image_comparison(baseline_images=["cvt_archive_heatmap_long"],
                  remove_text=False,
                  extensions=["png"])
def test_cvt_archive_heatmap_long(_long_cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(_long_cvt_archive)


@image_comparison(baseline_images=["cvt_archive_heatmap_transpose"],
                  remove_text=False,
                  extensions=["png"])
def test_cvt_archive_heatmap_transpose(_long_cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(_long_cvt_archive, transpose_bcs=True)


@image_comparison(baseline_images=["cvt_archive_heatmap_transpose_square"],
                  remove_text=False,
                  extensions=["png"])
def test_cvt_archive_heatmap_transpose_square(_long_cvt_archive):
    plt.figure(figsize=(4, 6))
    cvt_archive_heatmap(_long_cvt_archive, transpose_bcs=True, square=True)
