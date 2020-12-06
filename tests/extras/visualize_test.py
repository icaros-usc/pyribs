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

from ribs.archives import CVTArchive
from ribs.visualize import cvt_archive_heatmap

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
def _cvt_archive_fixture():
    """A deterministically created CVTArchive."""
    seed = 42
    np.random.seed(seed)  # Make scipy's k-means deterministic.
    archive = CVTArchive([(-1, 1), (-1, 1)],
                         100,
                         samples=1000,
                         use_kd_tree=False,
                         k_means_threshold=1e-6,
                         seed=seed)
    archive.initialize(solution_dim=2)

    # Add solutions.
    n_vals = 100_000
    for _ in range(n_vals):
        solution = np.random.uniform(-1, 1, 2)
        archive.add(
            solution=solution,
            # Objective is the negative sphere function.
            objective_value=-np.sum(np.square(solution)),
            behavior_values=solution,
        )
    return archive


def test_cvt_archive_heatmap_fails_on_non_2d():
    archive = CVTArchive([(-1, 1), (-1, 1), (-1, 1)],
                         100,
                         samples=100,
                         use_kd_tree=False,
                         k_means_threshold=1e-6)
    archive.initialize(solution_dim=3)

    with pytest.raises(ValueError):
        cvt_archive_heatmap(archive)


@image_comparison(baseline_images=["cvt_archive_heatmap"],
                  remove_text=False,
                  extensions=["png"])
def test_cvt_archive_heatmap(_cvt_archive_fixture):
    archive = _cvt_archive_fixture
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(archive)


@image_comparison(baseline_images=["cvt_archive_heatmap"],
                  remove_text=False,
                  extensions=["png"])
def test_cvt_archive_heatmap_with_custom_axis(_cvt_archive_fixture):
    archive = _cvt_archive_fixture
    _, ax = plt.subplots(figsize=(8, 6))
    cvt_archive_heatmap(archive, ax=ax)


@image_comparison(baseline_images=["cvt_archive_heatmap_with_coolwarm_cmap"],
                  remove_text=False,
                  extensions=["png"])
def test_cvt_archive_heatmap_with_coolwarm_cmap(_cvt_archive_fixture):
    archive = _cvt_archive_fixture
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(archive, cmap=matplotlib.cm.get_cmap("coolwarm"))


@image_comparison(baseline_images=["cvt_archive_heatmap_with_listed_cmap"],
                  remove_text=False,
                  extensions=["png"])
def test_cvt_archive_heatmap_with_listed_cmap(_cvt_archive_fixture):
    archive = _cvt_archive_fixture
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(archive, cmap=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])


@image_comparison(baseline_images=["cvt_archive_heatmap_with_samples"],
                  remove_text=False,
                  extensions=["png"])
def test_cvt_archive_heatmap_with_samples(_cvt_archive_fixture):
    archive = _cvt_archive_fixture
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(archive, plot_samples=True)


@image_comparison(baseline_images=["cvt_archive_heatmap_with_limits"],
                  remove_text=False,
                  extensions=["png"])
def test_cvt_archive_heatmap_with_limits(_cvt_archive_fixture):
    archive = _cvt_archive_fixture
    plt.figure(figsize=(8, 6))
    # Negative sphere function in the _cvt_archive_fixture should have range
    # (-2, 0). These limits should give us a very uniform-looking archive.
    cvt_archive_heatmap(archive, vmin=-1.0, vmax=-0.5)
