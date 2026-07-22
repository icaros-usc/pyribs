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


# TODO: Examples.
# TODO: Support more errorbars.
def aggregate_cdf(
    archives: Collection[ArchiveBase],
    ax: Axes | None = None,
    dfs: Collection[DataFrame] | Collection[ArchiveDataFrame] | None = None,
    cumulative: bool | Literal[-1] = True,
    bins: int | Sequence[float] | str | None = 100,
    vmin: float | None = None,
    vmax: float | None = None,
    estimator: Literal["mean", "median"] = "mean",
    errorbar: None | Literal["se", "sd"] = "sd",
    show_edges: bool = True,
) -> None:
    """Plots a CDF/CCDF aggregated over multiple archives.

    Generally, the `CDF (cumulative distribution function)
    <https://en.wikipedia.org/wiki/Cumulative_distribution_function>`_ represents the
    number of observations that fall below each output value. In the case of archives, a
    CDF counts the number of elites that perform worse than or equal to each objective
    value. Conversely, a CCDF (*complementary* cumulative distribution function) counts
    the number of elites that perform better than or equal to each objective value.

    This function approximates a CDF/CCDF over multiple archives by first computing a
    histogram over the objective values in each archive. Then, it performs a cumulative
    sum on the histograms to obtain a CDF/CCDF. Finally, it aggregates the CDFs/CCDFs by
    aggregating the number of values in each histogram's bin. For example, if Archive
    0's CDF has 10 values in the bin [0, 0.1), and Archive 1's CDF has 20 values in the
    bin [0, 0.1), and we aggregate values using the "mean" estimator, then the final
    histogram shows 15 values in the bin [0, 0.1).

    This function also supports plotting histograms by setting the `cumulative`
    parameter to False.

    .. info::

        The idea of using a CDF/CCDF to evaluate QD algorithms was introduced and
        formalized in `Vassiliades
        2018 <https://arxiv.org/abs/1610.05729>`_.

    Args:
        archives: Archives to aggregate for the CDF/CCDF.
        ax: Axes on which to plot the CDF/CCDF. If ``None``, the current axis will be
            used.
        dfs: If provided, we will plot data from this sequence of dataframes instead of
            the data currently in the archives. This data can be obtained by, for
            instance, calling :meth:`ribs.archives.ArchiveBase.data` with
            ``return_type="pandas"`` and modifying the resulting
            :class:`~ribs.archives.ArchiveDataFrame`. Note that, at a minimum, each
            dataframe must contain a column for "objective". The number of dataframes
            must be the same as the number of archives.
        cumulative: Pass True to plot a CDF, -1 to plot a CCDF, and False to plot a
            histogram.
        bins: Bins for the CDF/CCDF. The default of 100 indicates that there will be 100
            equally-sized bins.
        vmin: Minimum objective value to use in the plot. If ``None``, the minimum
            objective value across all the archives is used.
        vmax: Maximum objective value to use in the plot. If ``None``, the maximum
            objective value across all the archives is used.
        estimator: Method for aggregating the CDF/CCDF or histogram across the multiple
            archives. For example, if "mean" is passed, we count the number of entries
            in each histogram bin for each archive, and the final plot shows the mean
            number of entries in each bin.
        errorbar: Method for computing the errorbar for the CDF/CCDF or histogram. For
            example, if "sd" is passed, we display an errorbar showing the standard
            deviation of the number of entries in each histogram bin. Pass None to hide
            the errorbar.
        show_edges: Whether to show the left and right edges of the CDF/CCDF or
            histogram (these show up as vertical lines).
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
        # CDF.
        histograms = np.cumsum(histograms, axis=1)
    elif cumulative < 0:
        # CCDF -- We want to do a cumsum from right to left, so we flip the histograms,
        # compute the cumsum from left to right, and then flip back.
        histograms = np.flip(np.cumsum(np.flip(histograms, axis=1), axis=1), axis=1)
    else:  # cumulative is False (i.e., 0).
        # Leave the histograms as is.
        pass

    # Aggregate values with `estimator`.
    if estimator == "mean":
        agg_hist = histograms.mean(axis=0)
    elif estimator == "median":
        agg_hist = histograms.median(axis=0)
    else:
        raise ValueError(f"Unknown estimator {estimator}")

    # Compute errors/spread with `errorbar`.
    if errorbar is None:
        err_hist = np.zeros_like(histograms)
    elif errorbar == "sd":
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
    if errorbar is not None:
        ax.stairs(
            values=agg_hist + err_hist,
            edges=bin_edges,
            baseline=agg_hist - err_hist,
            fill=True,
            alpha=0.2,
            color=patch.get_edgecolor(),
        )
