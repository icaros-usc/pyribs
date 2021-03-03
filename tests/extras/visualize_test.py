"""Tests for ribs.visualize.

For image comparison tests, read these instructions:
https://matplotlib.org/3.3.1/devel/testing.html#writing-an-image-comparison-test.
Essentially, after running a new test for the first time in the _root_ directory
of this repo, copy the image output from result_images/visualize_test to
tests/extras/baseline_images/visualize_test. For instance, for
``test_cvt_archive_heatmap_with_samples``, run::

    cp result_images/visualize_test/cvt_archive_heatmap_with_samples.png \
        tests/extras/baseline_images/visualize_test/

Assuming the output is as expected (and assuming the code is deterministic), the
test should now pass when it is re-run.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.testing.decorators import image_comparison

from ribs.archives import CVTArchive, GridArchive, SlidingBoundariesArchive
from ribs.visualize import (cvt_archive_heatmap, grid_archive_heatmap,
                            sliding_boundaries_archive_heatmap,
                            parallel_axes_plot)


@pytest.fixture(autouse=True)
def _clean_matplotlib():
    """Cleans up matplotlib figures before and after each test."""
    # Before the test.
    plt.close("all")

    yield

    # After the test.
    plt.close("all")


def _add_uniform_sphere(archive, x_range, y_range):
    """Adds points from the negative sphere function in a 100x100 grid.

    The solutions are the same as the BCs (the (x,y) coordinates).

    x_range and y_range are tuples of (lower_bound, upper_bound).
    """
    for x in np.linspace(x_range[0], x_range[1], 100):
        for y in np.linspace(y_range[0], y_range[1], 100):
            archive.add(
                solution=np.array([x, y]),
                objective_value=-(x**2 + y**2),  # Negative sphere.
                behavior_values=np.array([x, y]),
            )


def _add_uniform_3d_sphere(archive, x_range, y_range, z_range):
    """Adds points from the negative sphere function in a 100x100x100 grid.

    The solutions are the same as the BCs (the (x,y,z) coordinates).

    x_range, y_range, and z_range are tuples of (lower_bound, upper_bound).
    """
    for x in np.linspace(x_range[0], x_range[1], 40):
        for y in np.linspace(y_range[0], y_range[1], 40):
            for z in np.linspace(z_range[0], z_range[1], 40):
                archive.add(
                    solution=np.array([x, y, z]),
                    objective_value=-(x**2 + y**2 + z**2),  # Negative sphere.
                    behavior_values=np.array([x, y, z]),
                )


def _add_random_sphere(archive, x_range, y_range):
    """Adds 1000 random points from the negative sphere function.

    Solutions, BCs, and ranges are same as in _add_uniform_sphere.
    """
    # Use random BCs to make the boundaries shift.
    rng = np.random.default_rng(10)
    for _ in range(1000):
        x, y = rng.uniform((x_range[0], y_range[0]), (x_range[1], y_range[1]))
        archive.add(
            solution=np.array([x, y]),
            objective_value=-(x**2 + y**2),
            behavior_values=np.array([x, y]),
        )


#
# Archive fixtures.
#


@pytest.fixture(scope="module")
def _grid_archive():
    """Deterministically created GridArchive."""
    # The archive must be low-res enough that we can tell if the number of cells
    # is correct, yet high-res enough that we can see different colors.
    archive = GridArchive([10, 10], [(-1, 1), (-1, 1)], seed=42)
    archive.initialize(solution_dim=2)
    _add_uniform_sphere(archive, (-1, 1), (-1, 1))
    return archive


@pytest.fixture(scope="module")
def _long_grid_archive():
    """Same as above, but the behavior space is longer in one direction."""
    archive = GridArchive([10, 10], [(-2, 2), (-1, 1)], seed=42)
    archive.initialize(solution_dim=2)
    _add_uniform_sphere(archive, (-2, 2), (-1, 1))
    return archive


@pytest.fixture(scope="module")
def _3d_grid_archive():
    """Deterministic archive, but there are three behavior axes of different
    sizes, and some of the axes are not totally filled. 
    """
    archive = GridArchive([10, 10, 10], [(-2, 2), (-1, 1), (-2, 1)], seed=42)
    archive.initialize(solution_dim=3)
    _add_uniform_3d_sphere(archive, (0, 2), (-1, 1), (-1, 0))
    return archive


@pytest.fixture(scope="module")
def _cvt_archive():
    """Deterministically created CVTArchive."""
    archive = CVTArchive(100, [(-1, 1), (-1, 1)],
                         samples=1000,
                         use_kd_tree=True,
                         seed=42)
    archive.initialize(solution_dim=2)
    _add_uniform_sphere(archive, (-1, 1), (-1, 1))
    return archive


@pytest.fixture(scope="module")
def _long_cvt_archive():
    """Same as above, but the behavior space is longer in one direction."""
    archive = CVTArchive(100, [(-2, 2), (-1, 1)],
                         samples=1000,
                         use_kd_tree=True,
                         seed=42)
    archive.initialize(solution_dim=2)
    _add_uniform_sphere(archive, (-2, 2), (-1, 1))
    return archive


@pytest.fixture(scope="module")
def _sliding_archive():
    """Deterministically created SlidingBoundariesArchive."""
    archive = SlidingBoundariesArchive([10, 20], [(-1, 1), (-1, 1)], seed=42)
    archive.initialize(solution_dim=2)
    _add_random_sphere(archive, (-1, 1), (-1, 1))
    return archive


@pytest.fixture(scope="module")
def _long_sliding_archive():
    """Same as above, but the behavior space is longer in one direction."""
    archive = SlidingBoundariesArchive([10, 20], [(-2, 2), (-1, 1)], seed=42)
    archive.initialize(solution_dim=2)
    _add_random_sphere(archive, (-2, 2), (-1, 1))
    return archive


#
# Tests for all heatmap functions. Unfortunately, these tests are hard to
# parametrize because the image_comparison decorator needs the filename, and
# pytest does not seem to pass params to decorators.
#


@pytest.mark.parametrize("archive_type", ["grid", "cvt", "sliding"])
def test_heatmap_fails_on_non_2d(archive_type):
    archive = {
        "grid":
            lambda: GridArchive([20, 20, 20], [(-1, 1)] * 3),
        "cvt":
            lambda: CVTArchive(100, [(-1, 1)] * 3, samples=100),
        "sliding":
            lambda: SlidingBoundariesArchive([20, 20, 20], [(-1, 1)] * 3),
    }[archive_type]()
    archive.initialize(solution_dim=2)  # Arbitrary.

    with pytest.raises(ValueError):
        {
            "grid": grid_archive_heatmap,
            "cvt": cvt_archive_heatmap,
            "sliding": sliding_boundaries_archive_heatmap,
        }[archive_type](archive)


@image_comparison(
    baseline_images=[
        "grid_archive_heatmap",
        "cvt_archive_heatmap",
        "sliding_boundaries_heatmap",
    ],
    remove_text=False,
    extensions=["png"],
)
def test_heatmap_archive(_grid_archive, _cvt_archive, _sliding_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(_grid_archive)

    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(_cvt_archive)

    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(_sliding_archive)


@image_comparison(
    baseline_images=[
        "grid_archive_heatmap",
        "cvt_archive_heatmap",
        "sliding_boundaries_heatmap",
    ],
    remove_text=False,
    extensions=["png"],
)
def test_cvt_archive_heatmap_with_custom_axis(_grid_archive, _cvt_archive,
                                              _sliding_archive):
    _, ax = plt.subplots(figsize=(8, 6))
    grid_archive_heatmap(_grid_archive, ax=ax)

    _, ax = plt.subplots(figsize=(8, 6))
    cvt_archive_heatmap(_cvt_archive, ax=ax)

    _, ax = plt.subplots(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(_sliding_archive, ax=ax)


@image_comparison(
    baseline_images=[
        "grid_archive_heatmap_long",
        "cvt_archive_heatmap_long",
        "sliding_boundaries_heatmap_long",
    ],
    remove_text=False,
    extensions=["png"],
)
def test_heatmap_long(_long_grid_archive, _long_cvt_archive,
                      _long_sliding_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(_long_grid_archive)

    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(_long_cvt_archive)

    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(_long_sliding_archive)


@image_comparison(
    baseline_images=[
        "grid_archive_heatmap_long_square",
        "cvt_archive_heatmap_long_square",
        "sliding_boundaries_heatmap_long_square",
    ],
    remove_text=False,
    extensions=["png"],
)
def test_heatmap_long_square(_long_grid_archive, _long_cvt_archive,
                             _long_sliding_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(_long_grid_archive, square=True)

    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(_long_cvt_archive, square=True)

    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(_long_sliding_archive, square=True)


@image_comparison(
    baseline_images=[
        "grid_archive_heatmap_long_transpose",
        "cvt_archive_heatmap_long_transpose",
        "sliding_boundaries_heatmap_long_transpose",
    ],
    remove_text=False,
    extensions=["png"],
)
def test_heatmap_long_transpose(_long_grid_archive, _long_cvt_archive,
                                _long_sliding_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(_long_grid_archive, transpose_bcs=True)

    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(_long_cvt_archive, transpose_bcs=True)

    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(_long_sliding_archive,
                                       transpose_bcs=True)


@image_comparison(
    baseline_images=[
        "grid_archive_heatmap_with_limits",
        "cvt_archive_heatmap_with_limits",
        "sliding_boundaries_heatmap_with_limits",
    ],
    remove_text=False,
    extensions=["png"],
)
def test_heatmap_with_limits(_grid_archive, _cvt_archive, _sliding_archive):
    # Negative sphere function should have range (-2, 0). These limits should
    # give a more uniform-looking archive.
    kwargs = {"vmin": -1.0, "vmax": -0.5}

    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(_grid_archive, **kwargs)

    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(_cvt_archive, **kwargs)

    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(_sliding_archive, **kwargs)


@image_comparison(
    baseline_images=[
        "grid_archive_heatmap_with_listed_cmap",
        "cvt_archive_heatmap_with_listed_cmap",
        "sliding_boundaries_heatmap_with_listed_cmap",
    ],
    remove_text=False,
    extensions=["png"],
)
def test_heatmap_listed_cmap(_grid_archive, _cvt_archive, _sliding_archive):
    cmap = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Primary red, green, blue.

    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(_grid_archive, cmap=cmap)

    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(_cvt_archive, cmap=cmap)

    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(_sliding_archive, cmap=cmap)


@image_comparison(
    baseline_images=[
        "grid_archive_heatmap_with_coolwarm_cmap",
        "cvt_archive_heatmap_with_coolwarm_cmap",
        "sliding_boundaries_heatmap_with_coolwarm_cmap",
    ],
    remove_text=False,
    extensions=["png"],
)
def test_heatmap_coolwarm_cmap(_grid_archive, _cvt_archive, _sliding_archive):
    cmap = matplotlib.cm.get_cmap("coolwarm")

    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(_grid_archive, cmap=cmap)

    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(_cvt_archive, cmap=cmap)

    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(_sliding_archive, cmap=cmap)


#
# Miscellaneous heatmap tests.
#


@image_comparison(baseline_images=["grid_archive_heatmap_with_boundaries"],
                  remove_text=False,
                  extensions=["png"])
def test_grid_archive_with_boundaries(_grid_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(_grid_archive,
                         pcm_kwargs={
                             "edgecolor": "black",
                             "linewidth": 0.1
                         })


@image_comparison(
    baseline_images=["sliding_boundaries_heatmap_with_boundaries"],
    remove_text=False,
    extensions=["png"])
def test_sliding_archive_with_boundaries(_sliding_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(_sliding_archive, boundary_lw=0.5)


@image_comparison(baseline_images=["cvt_archive_heatmap_with_samples"],
                  remove_text=False,
                  extensions=["png"])
def test_cvt_archive_heatmap_with_samples(_cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(_cvt_archive, plot_samples=True)


#
# Parallel coordinate plot test
#


@image_comparison(baseline_images=["parallel_axes_2d"],
                  remove_text=False,
                  extensions=["png"])
def test_parallel_axes_2d(_grid_archive):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(_grid_archive)


@image_comparison(baseline_images=["parallel_axes_3d"],
                  remove_text=False,
                  extensions=["png"])
def test_parallel_axes_3d(_3d_grid_archive):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(_3d_grid_archive)
