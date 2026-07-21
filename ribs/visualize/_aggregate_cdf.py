"""Provides aggregate_cdf."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib.axes import Axes
from pandas import DataFrame

from ribs.archives import ArchiveBase, ArchiveDataFrame
from ribs.visualize._utils import compute_vmin_vmax, validate_df


# TODO: Support more errorbars.
# TODO: None default for estimator and errorbar?
def aggregate_cdf(
    archives: Collection[ArchiveBase],
    ax: Axes | None = None,
    dfs: Collection[DataFrame] | Collection[ArchiveDataFrame] | None = None,
    bins: int | Sequence[float] | str | None = 100,
    vmin: float | None = None,
    vmax: float | None = None,
    cumulative: bool | Literal[-1] = False,
    estimator: Literal["mean", "median"] = "mean",
    errorbar: Literal["se", "sd"] = "sd",
    show_edges: bool = True,
) -> None:
    """Plots a CDF/CCDF aggregated over multiple archives.

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
            dataframe must contain a column for "objective". The number of dataframes
            must be the same as the number of archives.
        bins: Bins for the CDF. The default of 100 indicates that the histogram will
            consist of 100 equally-sized bins. See :func:`~matplotlib.pyplot.hist` for
            more info.
        vmin: Minimum objective value to use in the plot. If ``None``, the minimum
            objective value across all the archives is used.
        vmax: Maximum objective value to use in the plot. If ``None``, the maximum
            objective value across all the archives is used.
        cumulative: Pass False to plot a regular histogram, True to plot a CDF, and -1
            to pass a CCDF.
        estimator: Method for aggregating the histogram or CDF/CCDF across the multiple
            archives. For example, if "mean" is passed, we count the number of entries
            in each histogram bin for each archive, and the final plot shows the mean
            number of entries in each bin.
        errorbar: Method for computing the errorbar for the histogram or CDF/CCDF. For
            example, if "sd" is passed, we display an errorbar showing the standard
            deviation of the number of entries in each histogram bin.
        show_edges: Whether to show the edges of the histogram or CDF/CCDF.
    """
    if dfs is None:
        objectives = []
        for archive in archives:
            try:
                objectives.append(archive.data("objective"))
            except NotImplementedError as e:
                raise AttributeError(
                    "To use aggregate_cdf, each archive must have the data() method."
                ) from e
    else:
        if len(dfs) != len(archives):
            raise ValueError(
                "If passed in, the number of dfs must equal the number of archives."
            )
        objectives = []
        for df in dfs:
            df = validate_df(df)
            objectives.append(np.asarray(df["objective"]))

    vmin, vmax = compute_vmin_vmax(vmin, vmax, np.concatenate(objectives))

    # Initialize axis.
    ax = plt.gca() if ax is None else ax

    # Compute histogram for each archive.
    histograms = []
    for objs in objectives:
        hist, bin_edges = np.histogram(objs, bins, range=(vmin, vmax))
        histograms.append(hist)
    histograms = np.stack(histograms, axis=0)

    # Apply the cumulative parameter if needed.
    if cumulative > 0:
        histograms = np.cumsum(histograms, axis=1)
    elif cumulative < 0:
        # TODO: Complementary
        histograms = ...
    else:
        # `cumulative` is False or 0, meaning we leave the histogram as is.
        ...

    # Aggregate values with `estimator`.
    if estimator == "mean":
        agg_hist = histograms.mean(axis=0)
    elif estimator == "median":
        agg_hist = histograms.median(axis=0)
    else:
        raise ValueError(f"Unknown estimator {estimator}")

    # Compute errors/spread with `errorbar`.
    if errorbar == "sd":
        err_hist = histograms.std(axis=0)
    elif errorbar == "se":
        err_hist = scipy.stats.sem(histograms, axis=0)
    else:
        raise ValueError(f"Unknown errorbar {errorbar}")

    patch = ax.stairs(
        values=agg_hist,
        edges=bin_edges,
        baseline=0 if show_edges else None,
    )
    ax.stairs(
        values=agg_hist + err_hist,
        edges=bin_edges,
        baseline=agg_hist - err_hist,
        fill=True,
        alpha=0.2,
        color=patch.get_edgecolor(),
    )
