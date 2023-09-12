"""Utilities shared by the visualize module."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def retrieve_cmap(cmap):
    """Retrieves colormap from Matplotlib."""
    if isinstance(cmap, str):
        return plt.get_cmap(cmap)
    if isinstance(cmap, list):
        return matplotlib.colors.ListedColormap(cmap)
    return cmap


def validate_heatmap_visual_args(aspect, cbar, measure_dim, valid_dims,
                                 error_msg_measure_dim):
    """Helper function to validate arguments passed to `*_archive_heatmap`
    plotting functions.

    Args:
        valid_dims (list[int]): All specified valid archive dimensions that may
            be plotted into heatmaps.
        error_msg_measure_dim (str): Error message in ValueError if archive
            dimension plotting is not supported.

    Raises:
        ValueError: if validity checks for heatmap args fail
    """
    if aspect is not None and not (isinstance(aspect, float) or
                                   aspect in ["equal", "auto"]):
        raise ValueError(
            f"Invalid arg aspect='{aspect}'; must be 'auto', 'equal', or float")
    if measure_dim not in valid_dims:
        raise ValueError(error_msg_measure_dim)
    if not (cbar == "auto" or isinstance(cbar, matplotlib.axes.Axes) or
            cbar is None):
        raise ValueError(f"Invalid arg cbar={cbar}; must be 'auto', None, "
                         "or matplotlib.axes.Axes")


def set_cbar(t, ax, cbar, cbar_kwargs):
    """Sets cbar on the Axes given cbar arg."""
    cbar_kwargs = {} if cbar_kwargs is None else cbar_kwargs
    if cbar == "auto":
        ax.figure.colorbar(t, ax=ax, **cbar_kwargs)
    elif isinstance(cbar, matplotlib.axes.Axes):
        cbar.figure.colorbar(t, ax=cbar, **cbar_kwargs)


def archive_heatmap_1d(
    archive,
    boundaries,
    ax,
    cmap,
    aspect,
    vmin,
    vmax,
    cbar,
    cbar_kwargs,
    rasterized,
    pcm_kwargs,
):
    """Plots a heatmap of a 1D archive.

    Args:
        archive (ribs.archives.ArchiveBase): A 1D archive to plot.
        boundaries (np.ndarray): 1D array with the cell boundaries of the
            heatmap.
        ax (matplotlib.axes.Axes): See heatmap methods, e.g.,
            grid_archive_heatmap.
        cmap (matplotlib.colors.Colormap): The colormap to use when
            plotting intensity. Unlike in user-facing functions, we expect that
            this arg was already through retrieve_cmap to get a colormap object.
        aspect ('auto', 'equal', float): The aspect ratio of the heatmap. No
            default value for this function, unlike in user-facing functions.
        vmin (float): See heatmap methods, e.g., grid_archive_heatmap.
        vmax (float): See heatmap methods, e.g., grid_archive_heatmap.
        cbar ('auto', None, matplotlib.axes.Axes): See heatmap methods, e.g.,
            grid_archive_heatmap.
        cbar_kwargs (dict): See heatmap methods, e.g., grid_archive_heatmap.
        rasterized (bool): See heatmap methods, e.g., grid_archive_heatmap.
        pcm_kwargs (dict): Additional kwargs to pass to
            :func:`~matplotlib.pyplot.pcolormesh`.
    """

    # Retrieve data from archive. There should be only 2 bounds; upper and
    # lower, since it is 1D.
    df = archive.as_pandas()
    objective_batch = df.objective_batch()
    lower_bounds = archive.lower_bounds
    upper_bounds = archive.upper_bounds
    x_dim = archive.dims[0]
    x_bounds = boundaries
    y_bounds = np.array([0, 1])  # To facilitate default x-y aspect ratio.

    # Color for each cell in the heatmap.
    colors = np.full((1, x_dim), np.nan)
    grid_index_batch = archive.int_to_grid_index(df.index_batch())
    colors[0, grid_index_batch[:, 0]] = objective_batch

    # Initialize the axis.
    ax = plt.gca() if ax is None else ax
    ax.set_xlim(lower_bounds[0], upper_bounds[0])

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
