"""Provides sliding_boundaries_archive_heatmap."""
import matplotlib.pyplot as plt
import numpy as np

from ribs.visualize._utils import (retrieve_cmap, set_cbar,
                                   validate_heatmap_visual_args)

# Matplotlib functions tend to have a ton of args.
# pylint: disable = too-many-arguments


def sliding_boundaries_archive_heatmap(archive,
                                       ax=None,
                                       *,
                                       transpose_measures=False,
                                       cmap="magma",
                                       aspect="auto",
                                       ms=None,
                                       boundary_lw=0,
                                       vmin=None,
                                       vmax=None,
                                       cbar="auto",
                                       cbar_kwargs=None):
    """Plots heatmap of a :class:`~ribs.archives.SlidingBoundariesArchive` with
    2D measure space.

    Since the boundaries of :class:`ribs.archives.SlidingBoundariesArchive` are
    dynamic, we plot the heatmap as a scatter plot, in which each marker is an
    elite and its color represents the objective value. Boundaries can
    optionally be drawn by setting ``boundary_lw`` to a positive value.

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
            >>> rng = np.random.default_rng(seed=10)
            >>> coords = np.clip(rng.standard_normal((1000, 2)), -1.5, 1.5)
            >>> archive.add(solution_batch=coords,
            ...             objective_batch=-np.sum(coords**2, axis=1),
            ...             measures_batch=coords)
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
        archive (SlidingBoundariesArchive): A 2D
            :class:`~ribs.archives.SlidingBoundariesArchive`.
        ax (matplotlib.axes.Axes): Axes on which to plot the heatmap.
            If ``None``, the current axis will be used.
        transpose_measures (bool): By default, the first measure in the archive
            will appear along the x-axis, and the second will be along the
            y-axis. To switch this behavior (i.e. to transpose the axes), set
            this to ``True``.
        cmap (str, list, matplotlib.colors.Colormap): Colormap to use when
            plotting intensity. Either the name of a
            :class:`~matplotlib.colors.Colormap`, a list of RGB or RGBA colors
            (i.e. an :math:`N \\times 3` or :math:`N \\times 4` array), or a
            :class:`~matplotlib.colors.Colormap` object.
        aspect ('auto', 'equal', float): The aspect ratio of the heatmap (i.e.
            height/width). Defaults to ``'auto'``. ``'equal'`` is the same as
            ``aspect=1``.
        ms (float): Marker size for the solutions.
        boundary_lw (float): Line width when plotting the boundaries.
            Set to ``0`` to have no boundaries.
        vmin (float): Minimum objective value to use in the plot. If ``None``,
            the minimum objective value in the archive is used.
        vmax (float): Maximum objective value to use in the plot. If ``None``,
            the maximum objective value in the archive is used.
        cbar ('auto', None, matplotlib.axes.Axes): By default, this is set to
            ``'auto'`` which displays the colorbar on the archive's current
            :class:`~matplotlib.axes.Axes`. If ``None``, then colorbar is not
            displayed. If this is an :class:`~matplotlib.axes.Axes`, displays
            the colorbar on the specified Axes.
        cbar_kwargs (dict): Additional kwargs to pass to
            :func:`~matplotlib.pyplot.colorbar`.
    Raises:
        ValueError: The archive is not 2D.
    """
    validate_heatmap_visual_args(
        aspect, cbar, archive.measure_dim, [2],
        "Heatmaps can only be plotted for 2D SlidingBoundariesArchive")

    if aspect is None:
        aspect = "auto"

    # Try getting the colormap early in case it fails.
    cmap = retrieve_cmap(cmap)

    # Retrieve data from archive.
    df = archive.as_pandas()
    measures_batch = df.measures_batch()
    x = measures_batch[:, 0]
    y = measures_batch[:, 1]
    x_boundary = archive.boundaries[0]
    y_boundary = archive.boundaries[1]
    lower_bounds = archive.lower_bounds
    upper_bounds = archive.upper_bounds

    if transpose_measures:
        # Since the archive is 2D, transpose by swapping the x and y measures
        # and boundaries and by flipping the bounds (the bounds are arrays of
        # length 2).
        x, y = y, x
        x_boundary, y_boundary = y_boundary, x_boundary
        lower_bounds = np.flip(lower_bounds)
        upper_bounds = np.flip(upper_bounds)

    # Initialize the axis.
    ax = plt.gca() if ax is None else ax
    ax.set_xlim(lower_bounds[0], upper_bounds[0])
    ax.set_ylim(lower_bounds[1], upper_bounds[1])
    ax.set_aspect(aspect)

    # Create the plot.
    objective_batch = df.objective_batch()
    vmin = np.min(objective_batch) if vmin is None else vmin
    vmax = np.max(objective_batch) if vmax is None else vmax
    t = ax.scatter(x,
                   y,
                   s=ms,
                   c=objective_batch,
                   cmap=cmap,
                   vmin=vmin,
                   vmax=vmax)
    if boundary_lw > 0.0:
        # Careful with bounds here. Lines drawn along the x axis should extend
        # between the y bounds and vice versa -- see
        # https://github.com/icaros-usc/pyribs/issues/270
        ax.vlines(x_boundary,
                  lower_bounds[1],
                  upper_bounds[1],
                  color='k',
                  linewidth=boundary_lw)
        ax.hlines(y_boundary,
                  lower_bounds[0],
                  upper_bounds[0],
                  color='k',
                  linewidth=boundary_lw)

    # Create color bar.
    set_cbar(t, ax, cbar, cbar_kwargs)