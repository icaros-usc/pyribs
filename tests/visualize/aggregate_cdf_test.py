"""Tests for aggregate_cdf.

See README.md for instructions on writing tests.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.testing.decorators import image_comparison

from ribs.archives import DensityArchive, GridArchive
from ribs.visualize import aggregate_cdf

# pylint: disable=redefined-outer-name


#
# Fixtures
#


@pytest.fixture(scope="module")
def three_archives():
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


@pytest.fixture(scope="module")
def three_archives_skewed():
    """Same as above, but the distribution for each range is skewed."""
    rng = np.random.default_rng(42)

    archives = []
    for obj_count in [
        # Number of items in [0, 1), [1, 2), and [2, 3).
        [30, 20, 50],
        [30, 20, 50],
        [50, 35, 15],
        # Skewed -- e.g., [30, 30, 50] for [0, 1) has mean of 36.67 but median of 30.
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


def test_no_data_method_available():
    with pytest.raises(
        AttributeError,
        match=r"To use aggregate_cdf, each archive must have the data\(\) method\.",
    ):
        aggregate_cdf([DensityArchive(measure_dim=2) for _ in range(3)])


@image_comparison(
    baseline_images=["basic_cdf"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_basic_cdf(three_archives):
    """Basic CDF.

    The line should be the mean, which is 30, 60, 100.

    The errorbar should be the std, which is std(20, 30, 40) ~= 8.16, std(55, 60, 65) ~=
    4.08, std(100, 100, 100) = 0.
    """
    plt.figure(figsize=(8, 6))
    aggregate_cdf(three_archives, bins=3, cumulative=True)


@image_comparison(
    baseline_images=["basic_cdf"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_basic_cdf_with_dfs(three_archives):
    plt.figure(figsize=(8, 6))
    aggregate_cdf(
        three_archives,
        dfs=[
            archive.data("objective", return_type="pandas")
            for archive in three_archives
        ],
        bins=3,
        cumulative=True,
    )


@image_comparison(
    baseline_images=["cdf_with_labels"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_cdf_with_labels(three_archives):
    """Take advantage of the patches returned from the function."""
    plt.figure(figsize=(8, 6))
    line, errorbar = aggregate_cdf(
        three_archives,
        bins=3,
        cumulative=True,
    )
    line.set_label("Mean")
    errorbar.set_label("Error Bar")
    plt.legend()


def test_wrong_num_dfs(three_archives):
    plt.figure(figsize=(8, 6))
    with pytest.raises(
        ValueError,
        match=r"If passed in, the number of dfs must equal the number of archives\.",
    ):
        aggregate_cdf(
            three_archives,
            # Only provide dfs for two archives.
            dfs=[
                archive.data("objective", return_type="pandas")
                for archive in three_archives[:2]
            ],
        )


@image_comparison(
    baseline_images=["basic_ccdf"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_basic_ccdf(three_archives):
    """Basic CCDF.

    The line should be the mean, which is 100, 70, 40.

    The errorbar should be the std, which is std(100, 100, 100) = 0,  std(80, 70, 60) ~=
    8.16, std(45, 40, 35) ~= 4.08.
    """
    plt.figure(figsize=(8, 6))
    aggregate_cdf(three_archives, bins=3, cumulative=-1)


@image_comparison(
    baseline_images=["basic_histogram"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_basic_histogram(three_archives):
    """Basic histogram.

    The line should be the mean, which is 30, 30, 40.

    The errorbar should be the std, which is std(20, 30, 40) = 8.16,  std(35, 30, 25) ~=
    4.08, std(45, 40, 35) ~= 4.08.
    """
    plt.figure(figsize=(8, 6))
    aggregate_cdf(three_archives, bins=3, cumulative=False)


@image_comparison(
    baseline_images=["vmin_vmax"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_vmin_vmax(three_archives):
    """CDF with vmin and vmax.

    On average, the total number of items in the range [1, 2) for the archives is 30, so
    the bin on the right should have 30 items.
    """
    plt.figure(figsize=(8, 6))
    aggregate_cdf(three_archives, bins=2, vmin=1.0, vmax=2.0)


@image_comparison(
    baseline_images=["errorbar_se"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_errorbar_se(three_archives):
    """errorbar should be a bit smaller than in test_basic_cdf."""
    plt.figure(figsize=(8, 6))
    aggregate_cdf(three_archives, bins=3, cumulative=True, errorbar="se")


@image_comparison(
    baseline_images=["errorbar_none"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_errorbar_none(three_archives):
    """errorbar should be a lot smaller than in test_basic_cdf."""
    plt.figure(figsize=(8, 6))
    aggregate_cdf(three_archives, bins=3, cumulative=True, errorbar=None)


@image_comparison(
    baseline_images=["median_with_iqr_cdf"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_median_with_iqr_cdf(three_archives_skewed):
    plt.figure(figsize=(8, 6))
    aggregate_cdf(
        three_archives_skewed,
        bins=3,
        estimator="median",
        errorbar="iqr",
        cumulative=True,
    )


@image_comparison(
    baseline_images=["median_with_iqr_hist"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_median_with_iqr_hist(three_archives_skewed):
    plt.figure(figsize=(8, 6))
    aggregate_cdf(
        three_archives_skewed,
        bins=3,
        estimator="median",
        errorbar="iqr",
        cumulative=False,
    )


@image_comparison(
    baseline_images=["no_edges"],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_no_edges(three_archives):
    plt.figure(figsize=(8, 6))
    aggregate_cdf(three_archives, bins=3, cumulative=True, show_edges=False)


@image_comparison(
    baseline_images=[
        "full_scale_single_hist",
        "full_scale_hist",
        "full_scale_cdf_mean",
        "full_scale_cdf_median",
    ],
    remove_text=False,
    extensions=["png"],
    style="mpl20",
)
def test_full_scale():
    """Larger-scale test."""
    archives = []
    for i in range(5):
        archive = GridArchive(
            solution_dim=2, ranges=[(-1, 1), (-1, 1)], dims=[100, 100]
        )
        xxs, yys = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        xxs, yys = xxs.ravel(), yys.ravel()
        coords = np.stack((xxs, yys), axis=1)
        archive.add(
            solution=coords,
            objective=-(xxs**2 + yys**2) + i,  # Negative sphere, offset by i.
            measures=coords,
        )
        archives.append(archive)

    plt.figure(figsize=(8, 6))
    aggregate_cdf(archives[:1], cumulative=False)

    plt.figure(figsize=(8, 6))
    aggregate_cdf(archives, cumulative=False)

    plt.figure(figsize=(8, 6))
    aggregate_cdf(archives, cumulative=True)

    plt.figure(figsize=(8, 6))
    aggregate_cdf(archives, cumulative=True, estimator="median", errorbar="iqr")
