"""Tests for ribs.visualize.

For image comparison tests, read these instructions:
https://matplotlib.org/3.3.1/devel/testing.html#writing-an-image-comparison-test.
Essentially, after running a new test for the first time in the _root_ directory
of this repo, copy the image output from result_images/visualize_test to
tests/extras/baseline_images/visualize_test. For instance, for
``test_cvt_archive_heatmap_with_samples``, run::

    cp result_images/visualize_test/cvt_archive_heatmap_with_samples.png \
        tests/baseline_images/visualize_test/

Assuming the output is as expected (and assuming the code is deterministic), the
test should now pass when it is re-run.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.testing.decorators import image_comparison

from ribs.archives import CVTArchive, GridArchive, SlidingBoundariesArchive
from ribs.visualize import (cvt_archive_heatmap, grid_archive_heatmap,
                            parallel_axes_plot,
                            sliding_boundaries_archive_heatmap)

# pylint: disable = redefined-outer-name


@pytest.fixture(autouse=True)
def clean_matplotlib():
    """Cleans up matplotlib figures before and after each test."""
    # Before the test.
    plt.close("all")

    yield

    # After the test.
    plt.close("all")


def add_uniform_sphere_1d(archive, x_range):
    """Adds points from the negative sphere function in a 1D grid w/ 100 elites.

    The solutions are the same as the measures

    x_range is a tuple of (lower_bound, upper_bound).
    """
    x = np.linspace(x_range[0], x_range[1], 100)
    archive.add(
        solution_batch=x[:, None],
        objective_batch=-x**2,
        measures_batch=x[:, None],
    )


def add_uniform_sphere(archive, x_range, y_range):
    """Adds points from the negative sphere function in a 100x100 grid.

    The solutions are the same as the measures (the (x,y) coordinates).

    x_range and y_range are tuples of (lower_bound, upper_bound).
    """
    xxs, yys = np.meshgrid(
        np.linspace(x_range[0], x_range[1], 100),
        np.linspace(y_range[0], y_range[1], 100),
    )
    xxs, yys = xxs.ravel(), yys.ravel()
    coords = np.stack((xxs, yys), axis=1)
    archive.add(
        solution_batch=coords,
        objective_batch=-(xxs**2 + yys**2),  # Negative sphere.
        measures_batch=coords,
    )


def add_uniform_3d_sphere(archive, x_range, y_range, z_range):
    """Adds points from the negative sphere function in a 100x100x100 grid.

    The solutions are the same as the measures (the (x,y,z) coordinates).

    x_range, y_range, and z_range are tuples of (lower_bound, upper_bound).
    """
    xxs, yys, zzs = np.meshgrid(
        np.linspace(x_range[0], x_range[1], 40),
        np.linspace(y_range[0], y_range[1], 40),
        np.linspace(z_range[0], z_range[1], 40),
    )
    xxs, yys, zzs = xxs.ravel(), yys.ravel(), zzs.ravel()
    coords = np.stack((xxs, yys, zzs), axis=1)
    archive.add(
        solution_batch=coords,
        objective_batch=-(xxs**2 + yys**2 + zzs**2),  # Negative sphere.
        measures_batch=coords,
    )


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
    archive.add(
        solution_batch=solutions,
        objective_batch=-np.sum(np.square(solutions), axis=1),
        measures_batch=solutions,
    )


#
# Archive fixtures.
#
@pytest.fixture(scope="module")
def grid_archive_1d():
    """Deterministically created GridArchive with 1 measure."""
    # The archive must be low-res enough that we can tell if the number of cells
    # is correct, yet high-res enough that we can see different colors.
    archive = GridArchive(solution_dim=1, dims=[10], ranges=[(-1, 1)], seed=42)
    add_uniform_sphere_1d(archive, (-1, 1))
    return archive


@pytest.fixture(scope="module")
def grid_archive():
    """Deterministically created GridArchive."""
    # The archive must be low-res enough that we can tell if the number of cells
    # is correct, yet high-res enough that we can see different colors.
    archive = GridArchive(solution_dim=2,
                          dims=[10, 10],
                          ranges=[(-1, 1), (-1, 1)],
                          seed=42)
    add_uniform_sphere(archive, (-1, 1), (-1, 1))
    return archive


@pytest.fixture(scope="module")
def long_grid_archive():
    """Same as above, but the measure space is longer in one direction."""
    archive = GridArchive(solution_dim=2,
                          dims=[10, 10],
                          ranges=[(-2, 2), (-1, 1)],
                          seed=42)
    add_uniform_sphere(archive, (-2, 2), (-1, 1))
    return archive


@pytest.fixture(scope="module")
def three_d_grid_archive():
    """Deterministic archive, but there are three measure axes of different
    sizes, and some of the axes are not totally filled."""
    archive = GridArchive(solution_dim=3,
                          dims=[10, 10, 10],
                          ranges=[(-2, 2), (-1, 1), (-2, 1)],
                          seed=42)
    add_uniform_3d_sphere(archive, (0, 2), (-1, 1), (-1, 0))
    return archive


@pytest.fixture(scope="module")
def cvt_archive():
    """Deterministically created CVTArchive."""
    archive = CVTArchive(solution_dim=2,
                         cells=100,
                         ranges=[(-1, 1), (-1, 1)],
                         samples=1000,
                         use_kd_tree=True,
                         seed=42)
    add_uniform_sphere(archive, (-1, 1), (-1, 1))
    return archive


@pytest.fixture(scope="module")
def long_cvt_archive():
    """Same as above, but the measure space is longer in one direction."""
    archive = CVTArchive(solution_dim=2,
                         cells=100,
                         ranges=[(-2, 2), (-1, 1)],
                         samples=1000,
                         use_kd_tree=True,
                         seed=42)
    add_uniform_sphere(archive, (-2, 2), (-1, 1))
    return archive


@pytest.fixture(scope="module")
def sliding_archive():
    """Deterministically created SlidingBoundariesArchive."""
    archive = SlidingBoundariesArchive(solution_dim=2,
                                       dims=[10, 20],
                                       ranges=[(-1, 1), (-1, 1)],
                                       seed=42)
    add_random_sphere(archive, (-1, 1), (-1, 1))
    return archive


@pytest.fixture(scope="module")
def long_sliding_archive():
    """Same as above, but the measure space is longer in one direction."""
    archive = SlidingBoundariesArchive(solution_dim=2,
                                       dims=[10, 20],
                                       ranges=[(-2, 2), (-1, 1)],
                                       seed=42)
    add_random_sphere(archive, (-2, 2), (-1, 1))
    return archive


#
# Tests for all heatmap functions. Unfortunately, these tests are hard to
# parametrize because the image_comparison decorator needs the filename, and
# pytest does not seem to pass params to decorators. It is important to keep
# these tests separate so that if the test fails, we can immediately retrieve
# the correct result image. For instance, if we have 3 tests in a row and the
# first one fails, no result images will be generated for the remaining two
# tests.
#


@pytest.mark.parametrize("archive_type", ["grid", "cvt", "sliding"])
def test_heatmap_fails_on_unsupported_dims(archive_type):
    archive = {
        "grid":
            lambda: GridArchive(
                solution_dim=2, dims=[20, 20, 20], ranges=[(-1, 1)] * 3),
        "cvt":
            lambda: CVTArchive(
                solution_dim=2,
                cells=100,
                ranges=[(-1, 1)] * 3,
                samples=100,
            ),
        "sliding":
            lambda: SlidingBoundariesArchive(
                solution_dim=2, dims=[20, 20, 20], ranges=[(-1, 1)] * 3),
    }[archive_type]()

    with pytest.raises(ValueError):
        {
            "grid": grid_archive_heatmap,
            "cvt": cvt_archive_heatmap,
            "sliding": sliding_boundaries_archive_heatmap,
        }[archive_type](archive)


@pytest.mark.parametrize("archive_type", ["grid", "cvt", "sliding"])
@pytest.mark.parametrize("invalid_arg_cbar",
                         ["None", 3.2, True, (3.2, None), [3.2, None]]
                        )  # some random but invalid inputs
def test_heatmap_fails_on_invalid_cbar_option(archive_type, invalid_arg_cbar):
    archive = {
        "grid":
            lambda: GridArchive(
                solution_dim=2, dims=[20, 20, 20], ranges=[(-1, 1)] * 3),
        "cvt":
            lambda: CVTArchive(
                solution_dim=2,
                cells=100,
                ranges=[(-1, 1)] * 3,
                samples=100,
            ),
        "sliding":
            lambda: SlidingBoundariesArchive(
                solution_dim=2,
                dims=[20, 20, 20],
                ranges=[(-1, 1)] * 3,
            ),
    }[archive_type]()

    with pytest.raises(ValueError):
        {
            "grid": grid_archive_heatmap,
            "cvt": cvt_archive_heatmap,
            "sliding": sliding_boundaries_archive_heatmap,
        }[archive_type](archive=archive, cbar=invalid_arg_cbar)


@pytest.mark.parametrize("archive_type", ["grid", "cvt", "sliding"])
@pytest.mark.parametrize("invalid_arg_aspect",
                         ["None", True, (3.2, None), [3.2, None]]
                        )  # some random but invalid inputs
def test_heatmap_fails_on_invalid_aspect_option(archive_type,
                                                invalid_arg_aspect):
    archive = {
        "grid":
            lambda: GridArchive(
                solution_dim=2, dims=[20, 20, 20], ranges=[(-1, 1)] * 3),
        "cvt":
            lambda: CVTArchive(
                solution_dim=2, cells=100, ranges=[(-1, 1)] * 3, samples=100),
        "sliding":
            lambda: SlidingBoundariesArchive(
                solution_dim=2, dims=[20, 20, 20], ranges=[(-1, 1)] * 3),
    }[archive_type]()

    with pytest.raises(ValueError):
        {
            "grid": grid_archive_heatmap,
            "cvt": cvt_archive_heatmap,
            "sliding": sliding_boundaries_archive_heatmap,
        }[archive_type](archive=archive, aspect=invalid_arg_aspect)


@image_comparison(baseline_images=["grid_archive_heatmap"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_archive__grid(grid_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive)


@image_comparison(baseline_images=["grid_archive_heatmap_equal_aspect"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_archive__grid_equal_aspect(grid_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive, aspect="equal")


@image_comparison(baseline_images=["grid_archive_heatmap_aspect_gt_1"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_archive__grid_aspect_gt_1(grid_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive, aspect=2.5)


@image_comparison(baseline_images=["grid_archive_heatmap_aspect_lt_1"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_archive__grid_aspect_lt_1(grid_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive, aspect=0.5)


@image_comparison(baseline_images=["grid_archive_heatmap_1d"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_archive__grid_1d(grid_archive_1d):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive_1d)


@image_comparison(baseline_images=["grid_archive_heatmap_1d_aspect_gt_1"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_archive__grid_1d_aspect_gt_1(grid_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive, aspect=2.5)


@image_comparison(baseline_images=["grid_archive_heatmap_1d_aspect_lt_1"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_archive__grid_1d_aspect_lt_1(grid_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive, aspect=0.5)


@image_comparison(baseline_images=["grid_archive_heatmap_no_cbar"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_archive__grid_no_cbar(grid_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive, cbar=None)


@image_comparison(baseline_images=["grid_archive_heatmap_custom_cbar_axis"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_archive__grid_custom_cbar_axis(grid_archive):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
    grid_archive_heatmap(grid_archive, ax=ax1, cbar=ax2)


@image_comparison(baseline_images=["cvt_archive_heatmap"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_archive__cvt(cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive)


@image_comparison(baseline_images=["sliding_boundaries_heatmap"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_archive__sliding(sliding_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(sliding_archive)


@image_comparison(baseline_images=["grid_archive_heatmap"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_with_custom_axis__grid(grid_archive):
    _, ax = plt.subplots(figsize=(8, 6))
    grid_archive_heatmap(grid_archive, ax=ax)


@image_comparison(baseline_images=["cvt_archive_heatmap"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_with_custom_axis__cvt(cvt_archive):
    _, ax = plt.subplots(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive, ax=ax)


@image_comparison(baseline_images=["sliding_boundaries_heatmap"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_with_custom_axis__sliding(sliding_archive):
    _, ax = plt.subplots(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(sliding_archive, ax=ax)


@image_comparison(baseline_images=["grid_archive_heatmap_long"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_long__grid(long_grid_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(long_grid_archive)


@image_comparison(baseline_images=["cvt_archive_heatmap_long"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_long__cvt(long_cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(long_cvt_archive)


@image_comparison(baseline_images=["sliding_boundaries_heatmap_long"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_long__sliding(long_sliding_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(long_sliding_archive)


@image_comparison(baseline_images=["grid_archive_heatmap_long_square"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_long_square__grid(long_grid_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(long_grid_archive, aspect="equal")


@image_comparison(baseline_images=["cvt_archive_heatmap_long_square"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_long_square__cvt(long_cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(long_cvt_archive, aspect="equal")


@image_comparison(baseline_images=["sliding_boundaries_heatmap_long_square"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_long_square__sliding(long_sliding_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(long_sliding_archive, aspect="equal")


@image_comparison(baseline_images=["grid_archive_heatmap_long_transpose"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_long_transpose__grid(long_grid_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(long_grid_archive, transpose_measures=True)


@image_comparison(baseline_images=["cvt_archive_heatmap_long_transpose"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_long_transpose__cvt(long_cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(long_cvt_archive, transpose_measures=True)


@image_comparison(baseline_images=["sliding_boundaries_heatmap_long_transpose"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_long_transpose__sliding(long_sliding_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(long_sliding_archive,
                                       transpose_measures=True)


@image_comparison(baseline_images=["grid_archive_heatmap_with_limits"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_with_limits__grid(grid_archive):
    # Negative sphere function should have range (-2, 0). These limits should
    # give a more uniform-looking archive.
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive, vmin=-1.0, vmax=-0.5)


@image_comparison(baseline_images=["cvt_archive_heatmap_with_limits"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_with_limits__cvt(cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive, vmin=-1.0, vmax=-0.5)


@image_comparison(baseline_images=["sliding_boundaries_heatmap_with_limits"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_with_limits__sliding(sliding_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(sliding_archive, vmin=-1.0, vmax=-0.5)


@image_comparison(baseline_images=["grid_archive_heatmap_with_listed_cmap"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_listed_cmap__grid(grid_archive):
    # cmap consists of primary red, green, and blue.
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive, cmap=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])


@image_comparison(baseline_images=["cvt_archive_heatmap_with_listed_cmap"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_listed_cmap__cvt(cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive, cmap=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])


@image_comparison(
    baseline_images=["sliding_boundaries_heatmap_with_listed_cmap"],
    remove_text=False,
    extensions=["png"])
def test_heatmap_listed_cmap__sliding(sliding_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(sliding_archive,
                                       cmap=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])


@image_comparison(baseline_images=["grid_archive_heatmap_with_coolwarm_cmap"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_coolwarm_cmap__grid(grid_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive, cmap="coolwarm")


@image_comparison(baseline_images=["cvt_archive_heatmap_with_coolwarm_cmap"],
                  remove_text=False,
                  extensions=["png"])
def test_heatmap_coolwarm_cmap__cvt(cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive, cmap="coolwarm")


@image_comparison(
    baseline_images=["sliding_boundaries_heatmap_with_coolwarm_cmap"],
    remove_text=False,
    extensions=["png"])
def test_heatmap_coolwarm_cmap__sliding(sliding_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(sliding_archive, cmap="coolwarm")


#
# Miscellaneous heatmap tests.
#


@image_comparison(baseline_images=["grid_archive_heatmap_with_boundaries"],
                  remove_text=False,
                  extensions=["png"])
def test_grid_archive_with_boundaries(grid_archive):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(grid_archive,
                         pcm_kwargs={
                             "edgecolor": "black",
                             "linewidth": 0.1
                         })


@image_comparison(
    baseline_images=["sliding_boundaries_heatmap_with_boundaries"],
    remove_text=False,
    extensions=["png"])
def test_sliding_archive_with_boundaries(sliding_archive):
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(sliding_archive, boundary_lw=0.5)


@image_comparison(
    baseline_images=["sliding_boundaries_heatmap_mismatch_xy_with_boundaries"],
    remove_text=False,
    extensions=["png"])
def test_sliding_archive_mismatch_xy_with_boundaries():
    """There was a bug caused by the boundary lines being assigned incorrectly.

    https://github.com/icaros-usc/pyribs/issues/270
    """
    archive = SlidingBoundariesArchive(solution_dim=2,
                                       dims=[10, 20],
                                       ranges=[(-1, 1), (-2, 2)],
                                       seed=42)
    add_random_sphere(archive, (-1, 1), (-2, 2))
    plt.figure(figsize=(8, 6))
    sliding_boundaries_archive_heatmap(archive, boundary_lw=0.5)


@image_comparison(baseline_images=["cvt_archive_heatmap_with_samples"],
                  remove_text=False,
                  extensions=["png"])
def test_cvt_archive_heatmap_with_samples(cvt_archive):
    plt.figure(figsize=(8, 6))
    cvt_archive_heatmap(cvt_archive, plot_samples=True)


#
# Parallel coordinate plot test
#


@image_comparison(baseline_images=["parallel_axes_2d"],
                  remove_text=False,
                  extensions=["png"])
def test_parallel_axes_2d(grid_archive):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(grid_archive)


@image_comparison(baseline_images=["parallel_axes_3d"],
                  remove_text=False,
                  extensions=["png"])
def test_parallel_axes_3d(three_d_grid_archive):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(three_d_grid_archive)


@image_comparison(baseline_images=["parallel_axes_3d"],
                  remove_text=False,
                  extensions=["png"])
def test_parallel_axes_3d_custom_ax(three_d_grid_archive):
    _, ax = plt.subplots(figsize=(8, 6))
    parallel_axes_plot(three_d_grid_archive, ax=ax)


@image_comparison(baseline_images=["parallel_axes_3d_custom_order"],
                  remove_text=False,
                  extensions=["png"])
def test_parallel_axes_3d_custom_order(three_d_grid_archive):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(three_d_grid_archive, measure_order=[1, 2, 0])


@image_comparison(baseline_images=["parallel_axes_3d_custom_names"],
                  remove_text=False,
                  extensions=["png"])
def test_parallel_axes_3d_custom_names(three_d_grid_archive):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(three_d_grid_archive,
                       measure_order=[(1, 'One'), (2, 'Two'), (0, 'Zero')])


@image_comparison(baseline_images=["parallel_axes_3d_coolwarm"],
                  remove_text=False,
                  extensions=["png"])
def test_parallel_axes_3d_coolwarm_cmap(three_d_grid_archive):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(three_d_grid_archive, cmap='coolwarm')


@image_comparison(baseline_images=["parallel_axes_3d_width2_alpha2"],
                  remove_text=False,
                  extensions=["png"])
def test_parallel_axes_3d_width2_alpha2(three_d_grid_archive):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(three_d_grid_archive, linewidth=2.0, alpha=0.2)


@image_comparison(baseline_images=["parallel_axes_3d_custom_objective_limits"],
                  remove_text=False,
                  extensions=["png"])
def test_parallel_axes_3d_custom_objective_limits(three_d_grid_archive):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(three_d_grid_archive, vmin=-2.0, vmax=-1.0)


@image_comparison(baseline_images=["parallel_axes_3d_sorted"],
                  remove_text=False,
                  extensions=["png"])
def test_parallel_axes_3d_sorted(three_d_grid_archive):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(three_d_grid_archive, sort_archive=True)


@image_comparison(baseline_images=["parallel_axes_3d_vertical_cbar"],
                  remove_text=False,
                  extensions=["png"])
def test_parallel_axes_3d_vertical_cbar(three_d_grid_archive):
    plt.figure(figsize=(8, 6))
    parallel_axes_plot(three_d_grid_archive, cbar_orientation='vertical')
