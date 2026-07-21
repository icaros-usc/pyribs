"""Tests for aggregate_cdf.

See README.md for instructions on writing tests.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.testing.decorators import image_comparison

from ribs.archives import GridArchive
from ribs.visualize import aggregate_cdf

# pylint: disable=redefined-outer-name


#
# Fixtures
#


@pytest.fixture
def simple_archives():
    """Three archives intended to have three bins of objective values."""
    rng = np.random.default_rng(42)

    archives = []
    for obj_count in [
        # Number of items in [0, 1), [1, 2), and [2, 3).
        #
        # Keep in mind that when plotting a CDF, the CDF does a cumulative sum over the
        # number of items in the bins. Hence, there should be a high std in the count in
        # [0, 1), then [1, 2) will have lower std (sums are 55, 60, 65), and finally,
        # [2, 3) will have zero std (all sums are 100 at that point).
        [20, 35, 45],
        [30, 30, 40],
        [40, 25, 35],
    ]:
        # Populate the archive with the negative sphere function.
        archive = GridArchive(solution_dim=2, dims=[100], ranges=[(0, 1)])

        objectives = np.concatenate(
            (
                0.0 + rng.uniform(0, 1, size=obj_count[0]),
                1.0 + rng.uniform(0, 1, size=obj_count[1]),
                2.0 + rng.uniform(0, 1, size=obj_count[2]),
            )
        )

        archive.add(
            solution=np.zeros((100, 2)),
            objective=objectives,
            measures=np.arange(0, 1, 0.01)[:, None],
        )
        archives.append(archive)

    return archives


#
# Tests
#


@image_comparison(
    baseline_images=["basic_histogram"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_basic_histogram(simple_archives):
    plt.figure(figsize=(8, 6))
    aggregate_cdf(simple_archives, bins=3)


@image_comparison(
    baseline_images=["basic_cdf"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_basic_cdf(simple_archives):
    plt.figure(figsize=(8, 6))
    aggregate_cdf(simple_archives, bins=3, cumulative=True)


# TODO: vmin and vmax
# TODO: cumulative
# TODO: estimator
# TODO: errorbar
# TODO: show_edges
