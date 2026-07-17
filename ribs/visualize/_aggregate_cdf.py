"""Provides aggregate_cdf."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from pandas import DataFrame

from ribs.archives import ArchiveBase, ArchiveDataFrame
from ribs.visualize._utils import compute_vmin_vmax


def aggregate_cdf(
    archives: Collection[ArchiveBase],
    ax: Axes | None = None,
    dfs: Collection[DataFrame] | Collection[ArchiveDataFrame] | None = None,
    bins: int | Sequence[float] | str | None = 100,
    vmin: float | None = None,
    vmax: float | None = None,
    # estimator: Literal["mean", "median"] = "mean",
    # TODO: complementary
) -> None:
    """Plots a CDF aggregated over multiple archives.

    .. info::

        The idea of using a CCDF to evaluate QD algorithms was introduced and formalized
        in `Vassiliades
        2018 <https://arxiv.org/abs/1610.05729>`_.

    Args:
        archives: Archives to aggregate for the CDF.
        ax: Axes on which to plot the CDF. If ``None``, the current axis will be used.
        dfs: If provided, we will plot data from this sequence of dataframes instead of
            the data currently in the archives. This data can be obtained by, for
            instance, calling :meth:`ribs.archives.ArchiveBase.data` with
            ``return_type="pandas"`` and modifying the resulting
            :class:`~ribs.archives.ArchiveDataFrame`. Note that, at a minimum, each
            dataframe must contain a column for "objective".
        bins: Bins for the CDF. The default of 100 indicates that the histogram will
            consist of 100 equally-sized bins. See :func:`~matplotlib.pyplot.hist` for
            more info.
        vmin: Minimum objective value to use in the plot. If ``None``, the minimum
            objective value in the archives is used.
        vmax: Maximum objective value to use in the plot. If ``None``, the maximum
            objective value in the archives is used.
    """
    # TODO: Proper data retrieval.
    objectives = [archive.data("objective") for archive in archives]

    vmin, vmax = compute_vmin_vmax(vmin, vmax, np.concatenate(objectives))

    # Initialize axis.
    ax = plt.gca() if ax is None else ax

    histograms = []
    for objs in objectives:
        hist, bin_edges = np.histogram(objs, bins, range=(vmin, vmax))
        histograms.append(hist)
    histograms = np.stack(histograms, axis=0)

    print(histograms)

    # TODO: Complementary
    cdf = np.cumsum(histograms, axis=1)

    # TODO: Turn into arg.
    estimator = "mean"
    if estimator == "mean":
        agg_cdf = cdf.mean(axis=0)
    elif estimator == "median":
        agg_cdf = cdf.median(axis=0)
    else:
        raise ValueError(f"Unknown estimator {estimator}")

    # TODO: spread parameter.
    spread_cdf = cdf.std(axis=0)

    print(histograms.shape)
    print(agg_cdf.shape)
    print(spread_cdf.shape)

    print(agg_cdf[40:45])
    print(spread_cdf[40:45])

    patch = ax.stairs(values=agg_cdf, edges=bin_edges)
    ax.stairs(
        values=agg_cdf + spread_cdf,
        edges=bin_edges,
        baseline=agg_cdf - spread_cdf,
        fill=True,
        alpha=0.2,
        color=patch.get_edgecolor(),
    )
