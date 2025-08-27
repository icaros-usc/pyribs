"""Provides proximity_archive_plot."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.typing import ColorType
from pandas import DataFrame

from ribs.archives import ArchiveDataFrame, ProximityArchive
from ribs.typing import Float
from ribs.visualize._utils import (
    retrieve_cmap,
    set_cbar,
    validate_df,
    validate_heatmap_visual_args,
)


def proximity_archive_plot(
    archive: ProximityArchive,
    ax: Axes | None = None,
    *,
    df: DataFrame | ArchiveDataFrame | None = None,
    transpose_measures: bool = False,
    cmap: str | Sequence[ColorType] | matplotlib.colors.Colormap = "magma",
    aspect: Literal["auto", "equal"] | Float | None = None,
    ms: Float | None = None,
    lower_bounds: Sequence[Float] | None = None,
    upper_bounds: Sequence[Float] | None = None,
    vmin: Float | None = None,
    vmax: Float | None = None,
    cbar: Literal["auto"] | None | Axes = "auto",
    cbar_kwargs: dict | None = None,
    rasterized: bool = False,
) -> None:
    r"""Plots scatterplot of a :class:`~ribs.archives.ProximityArchive` with 2D measure space.

    Each marker in the scatterplot is an elite, and its color represents the objective
    value (objective values default to 0 in the ``ProximityArchive``).

    Examples:
        .. plot::
            :context: close-figs

            Single color plot for diversity optimization settings.

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from ribs.archives import ProximityArchive
            >>> from ribs.visualize import proximity_archive_plot
            >>> archive = ProximityArchive(solution_dim=2,
            ...                            measure_dim=2,
            ...                            k_neighbors=5,
            ...                            novelty_threshold=0.1,
            ...                            seed=42)
            >>> for _ in range(10):
            ...     x = np.random.uniform(-1, 1, 100)
            ...     y = np.random.uniform(-1, 1, 100)
            ...     archive.add(solution=np.stack((x, y), axis=1),
            ...                 # Objectives default to 0 in this case.
            ...                 objective=None,
            ...                 measures=np.stack((x, y), axis=1))
            >>> # Plot the archive.
            >>> plt.figure(figsize=(6, 6))
            >>> # Notice that the colormap is just a single RGB array, and the
            ... # colorbar is removed since it is just one color.
            >>> proximity_archive_plot(archive, cmap=[[0.5, 0.5, 0.5]],
            ...                        cbar=None)
            >>> plt.xlabel("x coords")
            >>> plt.ylabel("y coords")
            >>> plt.show()

        .. plot::
            :context: close-figs

            Plot where the objective has been recorded while using the archive.

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from ribs.archives import ProximityArchive
            >>> from ribs.visualize import proximity_archive_plot
            >>> archive = ProximityArchive(solution_dim=2,
            ...                            measure_dim=2,
            ...                            k_neighbors=5,
            ...                            novelty_threshold=0.1,
            ...                            seed=42)
            >>> for _ in range(10):
            ...     x = np.random.uniform(-1, 1, 100)
            ...     y = np.random.uniform(-1, 1, 100)
            ...     archive.add(solution=np.stack((x, y), axis=1),
            ...                 objective=-(x**2 + y**2),
            ...                 measures=np.stack((x, y), axis=1))
            >>> # Plot the archive.
            >>> plt.figure(figsize=(8, 6))
            >>> proximity_archive_plot(archive)
            >>> plt.title("Negative sphere function")
            >>> plt.xlabel("x coords")
            >>> plt.ylabel("y coords")
            >>> plt.show()

    Args:
        archive: A 2D :class:`~ribs.archives.ProximityArchive`.
        ax: Axes on which to make the plot. If ``None``, the current axis will be used.
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
        lower_bounds: Lower bounds of the measure space for the plot. Defaults to the
            minimum measure value along each dimension of the archive, minus 0.01.
        upper_bounds: Upper bounds of the measure space for the plot. Defaults to the
            maximum measure value along each dimension of the archive, plus 0.01.
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
        "Plot can only be made for a 2D ProximityArchive",
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

    if lower_bounds is None:
        if len(measures_batch) > 0:
            lower_bounds = np.min(measures_batch, axis=0) - 0.01
        else:
            # Sensible defaults when the archive is empty.
            lower_bounds = np.full(archive.measure_dim, -0.01)
    if upper_bounds is None:
        if len(measures_batch) > 0:
            upper_bounds = np.max(measures_batch, axis=0) + 0.01
        else:
            upper_bounds = np.full(archive.measure_dim, 0.01)

    if transpose_measures:
        # Since the archive is 2D, transpose by swapping the x and y measures and
        # boundaries and by flipping the bounds (the bounds are arrays of length 2).
        x, y = y, x
        lower_bounds = np.flip(lower_bounds)
        upper_bounds = np.flip(upper_bounds)

    # Initialize the axis.
    ax = plt.gca() if ax is None else ax
    ax.set_xlim(lower_bounds[0], upper_bounds[0])
    ax.set_ylim(lower_bounds[1], upper_bounds[1])
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

    # Create color bar.
    set_cbar(t, ax, cbar, cbar_kwargs)
