"""Provides archive_histogram."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.typing import ColorType
from pandas import DataFrame

from ribs.archives import ArchiveBase, ArchiveDataFrame
from ribs.visualize._utils import retrieve_cmap, validate_df


def archive_histogram(
    archive: ArchiveBase,
    ax: Axes | None = None,
    *,
    df: DataFrame | ArchiveDataFrame | None = None,
    bins: int | Sequence[float] | str | None = 100,
    vmin: float | None = None,
    vmax: float | None = None,
    ylim: float | None = None,
    color: ColorType = "#7e57c2",
    cmap: str | Sequence[ColorType] | matplotlib.colors.Colormap | None = None,
    rasterized: bool = False,
    hist_kwargs: dict | None = None,
) -> None:
    r"""Plots a histogram of the objective values in an archive.

    In short, this function is a wrapper around Matplotlib's
    :func:`~matplotlib.pyplot.hist`. It calls ``hist`` with objective values retrieved
    from the archive and then applies a number of (opinionated) customizations. As such,
    many of this function's arguments are shared with ``hist``.

    Examples:
        Basic Histogram of a 2D GridArchive

        .. plot::
            :context: close-figs

            import numpy as np
            import matplotlib.pyplot as plt
            from ribs.archives import GridArchive
            from ribs.visualize import archive_histogram

            # Populate the archive with the negative sphere function.
            archive = GridArchive(solution_dim=2,
                                  dims=[100, 100],
                                  ranges=[(-1, 1), (-1, 1)])
            x = np.random.uniform(-1, 1, 10000)
            y = np.random.uniform(-1, 1, 10000)
            archive.add(solution=np.stack((x, y), axis=1),
                        objective=-(x**2 + y**2),
                        measures=np.stack((x, y), axis=1))

            # Plot a histogram of the archive.
            plt.figure(figsize=(8, 6))
            archive_histogram(archive)
            plt.xlabel("Objective")
            plt.ylabel("Num. Elites")
            plt.show()

        Histogram Where Bars Are Colored With a Colormap

        .. plot::
            :context: close-figs

            import numpy as np
            import matplotlib.pyplot as plt
            from ribs.archives import GridArchive
            from ribs.visualize import archive_histogram

            # Populate the archive with the negative sphere function.
            archive = GridArchive(solution_dim=2,
                                  dims=[100, 100],
                                  ranges=[(-1, 1), (-1, 1)])
            x = np.random.uniform(-1, 1, 10000)
            y = np.random.uniform(-1, 1, 10000)
            archive.add(solution=np.stack((x, y), axis=1),
                        objective=-(x**2 + y**2),
                        measures=np.stack((x, y), axis=1))

            # Plot a histogram of the archive.
            plt.figure(figsize=(8, 6))
            archive_histogram(archive, cmap="magma")
            plt.xlabel("Objective")
            plt.ylabel("Num. Elites")
            plt.show()

        Histogram with More Customizations

        .. plot::
            :context: close-figs

            import numpy as np
            import matplotlib.pyplot as plt
            from ribs.archives import GridArchive
            from ribs.visualize import archive_histogram

            # Populate the archive with the negative sphere function.
            archive = GridArchive(solution_dim=2,
                                  dims=[100, 100],
                                  ranges=[(-1, 1), (-1, 1)])
            x = np.random.uniform(-1, 1, 10000)
            y = np.random.uniform(-1, 1, 10000)
            archive.add(solution=np.stack((x, y), axis=1),
                        objective=-(x**2 + y**2),
                        measures=np.stack((x, y), axis=1))

            # Plot a histogram of the archive.
            plt.figure(figsize=(8, 6))
            archive_histogram(
                archive,
                bins=50,  # Only use 50 bins.
                vmin=-2.5,  # Minimum objective value.
                vmax=0.5,  # Maximum objective value.
                ylim=400,  # Set the top of the y-axis.
            )
            plt.xlabel("Objective")
            plt.ylabel("Num. Elites")
            plt.show()

    Args:
        archive: An archive that can provide its objective values via the
            :meth:`~ribs.archives.ArchiveBase.data` method.
        ax: Axes on which to plot the histogram. If ``None``, the current axis will be
            used.
        df: If provided, we will plot data from this argument instead of the data
            currently in the archive. This data can be obtained by, for instance,
            calling :meth:`ribs.archives.ArchiveBase.data` with ``return_type="pandas"``
            and modifying the resulting :class:`~ribs.archives.ArchiveDataFrame`. Note
            that, at a minimum, the data must contain a column for "objective".
        bins: Bins for the histogram. The default of 100 indicates that the histogram
            will consist of 100 equally-sized bins. See :func:`~matplotlib.pyplot.hist`
            for more info.
        vmin: Minimum objective value to use in the plot. If ``None``, the minimum
            objective value in the archive is used.
        vmax: Maximum objective value to use in the plot. If ``None``, the maximum
            objective value in the archive is used.
        ylim: If provided, this is used to set the top ylimit (i.e., the upper bound of
            the y-axis) for the histogram.
        color: Color of the histogram bars. Defaults to the pyribs theme color. Refer to
            this `Matplotlib page
            <https://matplotlib.org/stable/users/explain/colors/colors.html>`_ for more
            info on specifying colors.
        cmap: Instead of passing in ``color``, a colormap can be passed in to color the
            histogram bars according to their objective value. This parameter can either
            be the name of a :class:`~matplotlib.colors.Colormap`, a list of Matplotlib
            color specifications (e.g., an :math:`N \times 3` or :math:`N \times 4`
            array -- see :class:`~matplotlib.colors.ListedColormap`), or a
            :class:`~matplotlib.colors.Colormap` object. For example, "magma" is the
            default used in the Pyribs heatmap visualizations. If both ``color`` and
            ``cmap`` are passed in, the ``cmap`` will take precedence.
        rasterized: Whether to rasterize the bars of the histogram. This can be useful
            for saving to a vector format like PDF. Essentially, only the bars will be
            converted to a raster graphic, so that they will not have to be individually
            rendered. Meanwhile, the surrounding axes, particularly text labels, will
            remain in vector format.
        hist_kwargs: Additional kwargs to pass to :func:`~matplotlib.pyplot.hist`. Note
            that since ``hist`` calls :func:`numpy.histogram`, some of these arguments
            may ultimately go to ``histogram``. Note that we already pass the following
            parameters: ``bins``, ``range`` (via ``vmin`` and ``vmax``), ``color``, and
            ``rasterized``.

    Raises:
        AttributeError: The data() method is not implemented on the given archive, which
            prevents retrieving its objective values.
    """
    # Retrieve archive data.
    if df is None:
        try:
            objectives = archive.data("objective")
        except NotImplementedError as e:
            raise AttributeError(
                "To use archive_histogram, the archive must have the data() method."
            ) from e
    else:
        df = validate_df(df)
        objectives = df["objective"]

    # Compute vmin and vmax.
    if len(objectives) == 0:
        # Sensible defaults when there is no elite in the archive. The colorbar for the
        # heatmap functions usually defaults to -0.1 and 0.1 when no objectives exist in
        # the archive.
        vmin = -0.1 if vmin is None else vmin
        vmax = 0.1 if vmax is None else vmax
    else:
        vmin = np.min(objectives) if vmin is None else vmin
        vmax = np.max(objectives) if vmax is None else vmax

    # Initialize axis.
    ax = plt.gca() if ax is None else ax

    # Set axis grid to show up behind the histogram.
    # https://stackoverflow.com/questions/1726391/matplotlib-draw-grid-lines-behind-other-graph-elements
    ax.set_axisbelow(True)

    # Hide spines.
    for pos in ["right", "top", "left"]:
        ax.spines[pos].set_visible(False)

    # Set up grid lines on y-axis.
    ax.grid(color="0.9", linestyle="-", axis="y")

    # Set the top of the y-axis.
    if ylim is not None:
        ax.set_ylim(top=ylim)

    # Plot the histogram.
    hist_kwargs = {} if hist_kwargs is None else hist_kwargs.copy()
    hist_kwargs.update(
        # If these keys are already in hist_kwargs, they will simply be replaced.
        {
            "bins": bins,
            "range": (vmin, vmax),
            "color": color,
            # `hist` passes this to Matplotlib's Patch as kwargs.
            "rasterized": rasterized,
        }
    )
    _bin_counts, bin_edges, patches = ax.hist(objectives, **hist_kwargs)

    # Color the histogram with a cmap. Inspiration from:
    # https://stackoverflow.com/questions/51347451/how-to-fill-histogram-with-color-gradient-where-a-fixed-point-represents-the-mid
    if cmap is not None:
        cmap = retrieve_cmap(cmap)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        norm = plt.Normalize(vmin, vmax)
        for c, p in zip(bin_centers, patches, strict=True):
            p.set(facecolor=cmap(norm(c)))
