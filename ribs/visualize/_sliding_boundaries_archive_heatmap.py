"""Provides sliding_boundaries_archive_heatmap."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.typing import ColorType
from pandas import DataFrame

from ribs.archives import ArchiveDataFrame, SlidingBoundariesArchive
from ribs.visualize._utils import (
    retrieve_cmap,
    set_cbar,
    validate_df,
    validate_heatmap_visual_args,
)


def sliding_boundaries_archive_heatmap(
    archive: SlidingBoundariesArchive,
    ax: Axes | None = None,
    *,
    df: DataFrame | ArchiveDataFrame | None = None,
    transpose_measures: bool = False,
    cmap: str | Sequence[ColorType] | matplotlib.colors.Colormap = "magma",
    aspect: Literal["auto", "equal"] | float | None = None,
    ms: float | None = None,
    boundary_lw: float = 0,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar: Literal["auto"] | None | Axes = "auto",
    cbar_kwargs: dict | None = None,
    rasterized: bool = False,
) -> None:
    r"""Plots heatmap of a :class:`~ribs.archives.SlidingBoundariesArchive` with 2D measure space.

    Since the boundaries of :class:`ribs.archives.SlidingBoundariesArchive` are dynamic,
    we plot the heatmap as a scatter plot, in which each marker is an elite and its
    color represents the objective value. Boundaries can optionally be drawn by setting
    ``boundary_lw`` to a positive value.

    Examples:
        .. plot::
            :context: close-figs

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from ribs.archives import SlidingBoundariesArchive
            >>> from ribs.visualize import sliding_boundaries_archive_heatmap
            >>> archive = SlidingBoundariesArchive(solution_dim=2,
            ...                                    dims=[10, 20],
            ...                                    ranges=[(-1, 1), (-1, 1)],
            ...                                    seed=42)
            >>> # Populate the archive with the negative sphere function.
            >>> xy = np.clip(np.random.standard_normal((1000, 2)), -1.5, 1.5)
            >>> archive.add(solution=xy,
            ...             objective=-np.sum(xy**2, axis=1),
            ...             measures=xy)
            >>> # Plot heatmaps of the archive.
            >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
            >>> fig.suptitle("Negative sphere function")
            >>> sliding_boundaries_archive_heatmap(archive, ax=ax1,
            ...                                    boundary_lw=0.5)
            >>> sliding_boundaries_archive_heatmap(archive, ax=ax2)
            >>> ax1.set_title("With boundaries")
            >>> ax2.set_title("Without boundaries")
            >>> ax1.set(xlabel='x coords', ylabel='y coords')
            >>> ax2.set(xlabel='x coords', ylabel='y coords')
            >>> plt.show()

    Args:
        archive: A 2D :class:`~ribs.archives.SlidingBoundariesArchive`.
        ax: Axes on which to plot the heatmap.  If ``None``, the current axis will be
            used.
        df: If provided, we will plot data from this argument instead of the data
            currently in the archive. This data can be obtained by, for instance,
            calling :meth:`ribs.archives.ArchiveBase.data` with ``return_type="pandas"``
            and modifying the resulting :class:`~ribs.archives.ArchiveDataFrame`. Note
            that, at a minimum, the data must contain columns for index, objective, and
            measures. To display a custom metric, replace the "objective" column.
        transpose_measures: By default, the first measure in the archive will appear
            along the x-axis, and the second will be along the y-axis. To switch this
            behavior (i.e. to transpose the axes), set this to ``True``.
        cmap: The colormap to use when plotting intensity. Either the name of a
            :class:`~matplotlib.colors.Colormap`, a list of Matplotlib color
            specifications (e.g., an :math:`N \times 3` or :math:`N \times 4` array --
            see :class:`~matplotlib.colors.ListedColormap`), or a
            :class:`~matplotlib.colors.Colormap` object.
        aspect: The aspect ratio of the heatmap (i.e. height/width). Defaults to
            ``'auto'`` for 2D and ``0.5`` for 1D. ``'equal'`` is the same as
            ``aspect=1``. See :meth:`matplotlib.axes.Axes.set_aspect` for more info.
        ms: Marker size for the solutions.
        boundary_lw: Line width when plotting the boundaries. Set to ``0`` to have no
            boundaries.
        vmin: Minimum objective value to use in the plot. If ``None``, the minimum
            objective value in the archive is used.
        vmax: Maximum objective value to use in the plot. If ``None``, the maximum
            objective value in the archive is used.
        cbar: By default, this is set to ``'auto'`` which displays the colorbar on the
            archive's current :class:`~matplotlib.axes.Axes`. If ``None``, then colorbar
            is not displayed. If this is an :class:`~matplotlib.axes.Axes`, displays the
            colorbar on the specified Axes.
        cbar_kwargs: Additional kwargs to pass to :func:`~matplotlib.pyplot.colorbar`.
        rasterized: Whether to rasterize the heatmap. This can be useful for saving to a
            vector format like PDF. Essentially, only the heatmap will be converted to a
            raster graphic so that the archive cells will not have to be individually
            rendered. Meanwhile, the surrounding axes, particularly text labels, will
            remain in vector format.

    Raises:
        ValueError: The archive is not 2D.
    """
    validate_heatmap_visual_args(
        aspect,
        cbar,
        archive.measure_dim,
        [2],
        "Heatmap can only be plotted for a 2D SlidingBoundariesArchive",
    )

    if aspect is None:
        aspect = "auto"

    # Try getting the colormap early in case it fails.
    cmap = retrieve_cmap(cmap)

    # Retrieve archive data.
    if df is None:
        measures_batch = archive.data("measures")
        objective_batch = archive.data("objective")
    else:
        df = validate_df(df)
        measures_batch = df.get_field("measures")
        objective_batch = df.get_field("objective")
    x = measures_batch[:, 0]
    y = measures_batch[:, 1]
    x_boundary = archive.boundaries[0]
    y_boundary = archive.boundaries[1]
    lower_bounds = archive.lower_bounds
    upper_bounds = archive.upper_bounds

    if transpose_measures:
        # Since the archive is 2D, transpose by swapping the x and y measures and
        # boundaries and by flipping the bounds (the bounds are arrays of length 2).
        x, y = y, x
        x_boundary, y_boundary = y_boundary, x_boundary
        lower_bounds = np.flip(lower_bounds)
        upper_bounds = np.flip(upper_bounds)

    # Initialize the axis.
    ax = plt.gca() if ax is None else ax
    ax.set_xlim(lower_bounds[0], upper_bounds[0])  # ty: ignore[invalid-argument-type]
    ax.set_ylim(lower_bounds[1], upper_bounds[1])  # ty: ignore[invalid-argument-type]
    ax.set_aspect(aspect)

    # Create the plot.
    vmin = (
        np.min(objective_batch) if vmin is None and len(objective_batch) > 0 else vmin
    )
    vmax = (
        np.max(objective_batch) if vmax is None and len(objective_batch) > 0 else vmax
    )
    t = ax.scatter(
        x,
        y,
        s=ms,
        c=objective_batch,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=rasterized,
    )
    if boundary_lw > 0.0:
        # Careful with bounds here. Lines drawn along the x axis should extend between
        # the y bounds and vice versa -- see
        # https://github.com/icaros-usc/pyribs/issues/270
        ax.vlines(
            x_boundary,
            lower_bounds[1],
            upper_bounds[1],
            color="k",
            linewidth=boundary_lw,
            rasterized=rasterized,
        )
        ax.hlines(
            y_boundary,
            lower_bounds[0],
            upper_bounds[0],
            color="k",
            linewidth=boundary_lw,
            rasterized=rasterized,
        )

    # Create color bar.
    set_cbar(t, ax, cbar, cbar_kwargs)
