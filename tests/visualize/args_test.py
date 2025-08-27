"""Tests to check argument validation in various visualization functions.

Many of the functions share parameters like cbar, so here we check if passing
invalid arguments to these functions results in an error.
"""

import pytest

from ribs.archives import CVTArchive, GridArchive, SlidingBoundariesArchive
from ribs.visualize import (
    cvt_archive_3d_plot,
    cvt_archive_heatmap,
    grid_archive_heatmap,
    sliding_boundaries_archive_heatmap,
)


@pytest.mark.parametrize("archive_type", ["grid", "cvt", "sliding", "cvt_3d"])
def test_fails_on_unsupported_dims(archive_type):
    archive = {
        "grid": lambda: GridArchive(
            solution_dim=2, dims=[20, 20, 20], ranges=[(-1, 1)] * 3
        ),
        "cvt": lambda: CVTArchive(
            solution_dim=2,
            cells=100,
            ranges=[(-1, 1)] * 3,
            samples=100,
        ),
        "sliding": lambda: SlidingBoundariesArchive(
            solution_dim=2, dims=[20, 20, 20], ranges=[(-1, 1)] * 3
        ),
        "cvt_3d": lambda: CVTArchive(
            solution_dim=2,
            cells=100,
            ranges=[(-1, 1)] * 4,
            samples=100,
        ),
    }[archive_type]()

    with pytest.raises(ValueError, match=r".* can only be .* for .*"):
        {
            "grid": grid_archive_heatmap,
            "cvt": cvt_archive_heatmap,
            "sliding": sliding_boundaries_archive_heatmap,
            "cvt_3d": cvt_archive_3d_plot,
        }[archive_type](archive)


@pytest.mark.parametrize("archive_type", ["grid", "cvt", "sliding", "cvt_3d"])
@pytest.mark.parametrize(
    "invalid_arg_cbar", ["None", 3.2, True, (3.2, None), [3.2, None]]
)  # some random but invalid inputs
def test_heatmap_fails_on_invalid_cbar_option(archive_type, invalid_arg_cbar):
    archive = {
        "grid": lambda: GridArchive(
            solution_dim=2, dims=[20, 20], ranges=[(-1, 1)] * 2
        ),
        "cvt": lambda: CVTArchive(
            solution_dim=2,
            cells=100,
            ranges=[(-1, 1)] * 2,
            samples=100,
        ),
        "sliding": lambda: SlidingBoundariesArchive(
            solution_dim=2,
            dims=[20, 20],
            ranges=[(-1, 1)] * 2,
        ),
        "cvt_3d": lambda: CVTArchive(
            solution_dim=2,
            cells=100,
            ranges=[(-1, 1)] * 3,
            samples=100,
        ),
    }[archive_type]()

    with pytest.raises(ValueError, match=r"Invalid arg cbar=.*"):
        {
            "grid": grid_archive_heatmap,
            "cvt": cvt_archive_heatmap,
            "sliding": sliding_boundaries_archive_heatmap,
            "cvt_3d": cvt_archive_3d_plot,
        }[archive_type](archive=archive, cbar=invalid_arg_cbar)


# Note: cvt_3d is not included because cvt_archive_3d_plot does not have an
# aspect parameter.
@pytest.mark.parametrize("archive_type", ["grid", "cvt", "sliding"])
@pytest.mark.parametrize(
    "invalid_arg_aspect", ["None", True, (3.2, None), [3.2, None]]
)  # some random but invalid inputs
def test_heatmap_fails_on_invalid_aspect_option(archive_type, invalid_arg_aspect):
    archive = {
        "grid": lambda: GridArchive(
            solution_dim=2, dims=[20, 20], ranges=[(-1, 1)] * 2
        ),
        "cvt": lambda: CVTArchive(
            solution_dim=2, cells=100, ranges=[(-1, 1)] * 2, samples=100
        ),
        "sliding": lambda: SlidingBoundariesArchive(
            solution_dim=2, dims=[20, 20], ranges=[(-1, 1)] * 2
        ),
    }[archive_type]()

    with pytest.raises(ValueError, match=r"Invalid arg aspect=.*"):
        {
            "grid": grid_archive_heatmap,
            "cvt": cvt_archive_heatmap,
            "sliding": sliding_boundaries_archive_heatmap,
        }[archive_type](archive=archive, aspect=invalid_arg_aspect)
