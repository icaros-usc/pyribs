"""Provides grid_archive_heatmap."""
import matplotlib.pyplot as plt
import numpy as np

from ribs.visualize._utils import (archive_heatmap_1d, retrieve_cmap, set_cbar,
                                   validate_df, validate_heatmap_visual_args)

# Matplotlib functions tend to have a ton of args.
# pylint: disable = too-many-arguments


def grid_archive_heatmap(archive,
                         ax=None,
                         *,
                         df=None,
                         transpose_measures=False,
                         cmap="magma",
                         aspect=None,
                         vmin=None,
                         vmax=None,
                         cbar="auto",
                         cbar_kwargs=None,
                         rasterized=False,
                         pcm_kwargs=None):
    """Plots heatmap of a :class:`~ribs.archives.GridArchive` with 1D or 2D
    measure space.

    This function creates a grid of cells and shades each cell with a color
    corresponding to the objective value of that cell's elite. This function
    uses :func:`~matplotlib.pyplot.pcolormesh` to generate the grid. For further
    customization, pass extra kwargs to :func:`~matplotlib.pyplot.pcolormesh`
    through the ``pcm_kwargs`` parameter. For instance, to create black
    boundaries of width 0.1, pass in ``pcm_kwargs={"edgecolor": "black",
    "linewidth": 0.1}``.

    Examples:
        .. plot::
            :context: close-figs

            Heatmap of a 2D GridArchive

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from ribs.archives import GridArchive
            >>> from ribs.visualize import grid_archive_heatmap
            >>> # Populate the archive with the negative sphere function.
            >>> archive = GridArchive(solution_dim=2,
            ...                       dims=[20, 20],
            ...                       ranges=[(-1, 1), (-1, 1)])
            >>> x = np.random.uniform(-1, 1, 10000)
            >>> y = np.random.uniform(-1, 1, 10000)
            >>> archive.add(solution_batch=np.stack((x, y), axis=1),
            ...             objective_batch=-(x**2 + y**2),
            ...             measures_batch=np.stack((x, y), axis=1))
            >>> # Plot a heatmap of the archive.
            >>> plt.figure(figsize=(8, 6))
            >>> grid_archive_heatmap(archive)
            >>> plt.title("Negative sphere function")
            >>> plt.xlabel("x coords")
            >>> plt.ylabel("y coords")
            >>> plt.show()

        .. plot::
            :context: close-figs

            Heatmap of a 1D GridArchive

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from ribs.archives import GridArchive
            >>> from ribs.visualize import grid_archive_heatmap
            >>> # Populate the archive with the negative sphere function.
            >>> archive = GridArchive(solution_dim=2,
            ...                       dims=[20], ranges=[(-1, 1)])
            >>> x = np.random.uniform(-1, 1, 1000)
            >>> archive.add(solution_batch=np.stack((x, x), axis=1),
            ...             objective_batch=-x**2,
            ...             measures_batch=x[:, None])
            >>> # Plot a heatmap of the archive.
            >>> plt.figure(figsize=(8, 6))
            >>> grid_archive_heatmap(archive)
            >>> plt.title("Negative sphere function with 1D measures")
            >>> plt.xlabel("x coords")
            >>> plt.show()

    Args:
        archive (GridArchive): A 1D or 2D :class:`~ribs.archives.GridArchive`.
        ax (matplotlib.axes.Axes): Axes on which to plot the heatmap.
            If ``None``, the current axis will be used.
        df (ribs.archives.ArchiveDataFrame): If provided, we will plot data from
            this argument instead of the data currently in the archive. This
            data can be obtained by, for instance, calling
            :meth:`ribs.archives.ArchiveBase.as_pandas()` and modifying the
            resulting :class:`ArchiveDataFrame`. Note that, at a minimum, the
            data must contain columns for index, objective, and measures. To
            display a custom metric, replace the "objective" column.
        transpose_measures (bool): By default, the first measure in the archive
            will appear along the x-axis, and the second will be along the
            y-axis. To switch this behavior (i.e. to transpose the axes), set
            this to ``True``. Does not apply for 1D archives.
        cmap (str, list, matplotlib.colors.Colormap): The colormap to use when
            plotting intensity. Either the name of a
            :class:`~matplotlib.colors.Colormap`, a list of RGB or RGBA colors
            (i.e. an :math:`N \\times 3` or :math:`N \\times 4` array), or a
            :class:`~matplotlib.colors.Colormap` object.
        aspect ('auto', 'equal', float): The aspect ratio of the heatmap (i.e.
            height/width). Defaults to ``'auto'`` for 2D and ``0.5`` for 1D.
            ``'equal'`` is the same as ``aspect=1``. See
            :meth:`matplotlib.axes.Axes.set_aspect` for more info.
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
        rasterized (bool): Whether to rasterize the heatmap. This can be useful
            for saving to a vector format like PDF. Essentially, only the
            heatmap will be converted to a raster graphic so that the archive
            cells will not have to be individually rendered. Meanwhile, the
            surrounding axes, particularly text labels, will remain in vector
            format. This is implemented by passing ``rasterized`` to
            :func:`~matplotlib.pyplot.pcolormesh`, so passing ``"rasterized"``
            in the ``pcm_kwargs`` below will raise an error.
        pcm_kwargs (dict): Additional kwargs to pass to
            :func:`~matplotlib.pyplot.pcolormesh`.

    Raises:
        ValueError: The archive's measure dimension must be 1D or 2D.
    """
    validate_heatmap_visual_args(
        aspect, cbar, archive.measure_dim, [1, 2],
        "Heatmap can only be plotted for a 1D or 2D GridArchive")

    if aspect is None:
        # Handles default aspects for different dims.
        if archive.measure_dim == 1:
            aspect = 0.5
        else:
            aspect = "auto"

    # Try getting the colormap early in case it fails.
    cmap = retrieve_cmap(cmap)

    # Retrieve archive data.
    df = archive.as_pandas() if df is None else validate_df(df)

    if archive.measure_dim == 1:
        cell_objectives = np.full(archive.cells, np.nan)
        cell_idx = archive.int_to_grid_index(df.index_batch()).squeeze()
        cell_objectives[cell_idx] = df.objective_batch()

        archive_heatmap_1d(
            archive,
            archive.boundaries[0],
            cell_objectives,
            ax,
            cmap,
            aspect,
            vmin,
            vmax,
            cbar,
            cbar_kwargs,
            rasterized,
            pcm_kwargs,
        )

    elif archive.measure_dim == 2:
        # Retrieve data from archive.
        objective_batch = df.objective_batch()
        lower_bounds = archive.lower_bounds
        upper_bounds = archive.upper_bounds
        x_dim, y_dim = archive.dims
        x_bounds = archive.boundaries[0]
        y_bounds = archive.boundaries[1]

        # Color for each cell in the heatmap.
        colors = np.full((y_dim, x_dim), np.nan)
        grid_index_batch = archive.int_to_grid_index(df.index_batch())
        colors[grid_index_batch[:, 1], grid_index_batch[:, 0]] = objective_batch

        if transpose_measures:
            # Since the archive is 2D, transpose by swapping the x and y
            # boundaries and by flipping the bounds (the bounds are arrays of
            # length 2).
            x_bounds, y_bounds = y_bounds, x_bounds
            lower_bounds = np.flip(lower_bounds)
            upper_bounds = np.flip(upper_bounds)
            colors = colors.T

        # Initialize the axis.
        ax = plt.gca() if ax is None else ax
        ax.set_xlim(lower_bounds[0], upper_bounds[0])
        ax.set_ylim(lower_bounds[1], upper_bounds[1])

        ax.set_aspect(aspect)

        # Create the plot.
        pcm_kwargs = {} if pcm_kwargs is None else pcm_kwargs
        vmin = np.min(objective_batch) if vmin is None else vmin
        vmax = np.max(objective_batch) if vmax is None else vmax
        t = ax.pcolormesh(x_bounds,
                          y_bounds,
                          colors,
                          cmap=cmap,
                          vmin=vmin,
                          vmax=vmax,
                          rasterized=rasterized,
                          **pcm_kwargs)

        # Create color bar.
        set_cbar(t, ax, cbar, cbar_kwargs)
