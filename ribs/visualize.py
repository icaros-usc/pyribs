"""Miscellaneous visualization tools.

Note that this module only works when you install ``ribs[all]``. As such, we do
not import it when you run ``import ribs``, and you will need to explicitly use
``import ribs.visualize``.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from scipy.spatial import Voronoi  # pylint: disable=no-name-in-module

__all__ = [
    "cvt_archive_heatmap",
    "DimensionalityError",
]


class DimensionalityError(Exception):
    """Indicates an archive has the wrong dimensionality for a visualization."""


def _get_pt_to_obj(cvt_archive):
    """Creates a dict from point index to objective value from a CVTArchive."""

    # Hopefully as_pandas() is okay in terms of efficiency since there are only
    # 7 columns (1 index, 2 centroid, 2 behavior, 1 objective, 1 solution).
    data = cvt_archive.as_pandas()

    pt_to_obj = {}
    for _, row in data.iterrows():
        pt_to_obj[row["index"]] = row["objective"]
    return pt_to_obj


def cvt_archive_heatmap(archive,
                        plot_samples=False,
                        ax=None,
                        figsize=(8, 6),
                        filename=None,
                        colormap=None):
    """Plots heatmap of a 2D :class:`ribs.archives.CVTArchive`.

    Essentially, we create a Voronoi diagram and shade in each cell with a
    color corresponding to the value of that cell's elite.

    Args:
        archive (CVTArchive): A 2D CVTArchive.
        plot_samples (bool): Whether to plot the samples used when generating
            the clusters.
        colormap (matplotlib.colors.Colormap): A colormap to use when plotting
            intensity. If None, will default to matplotlib's "magma" colormap.
        ax (matplotlib.axes.Axes): Axes on which to plot the heatmap. If None,
            a new figure and axis will be created with ``plt.subplots()``.
        figsize (tuple of (float, float)): Size of figure to create if ``ax`` is
            not passed in.
        filename (str): File to save the figure to. Can be used even when
            passing in an axis. Leave as None to avoid saving any figure.
    Returns:
        tuple: 2-element tuple containing:

            **fig** (*matplotlib figure*): Figure containing the heatmap and
            its colorbar.

            **ax** (*matplotlib axes*): Axis with the heatmap.
    Raises:
        DimensionalityError: The archive is not 2D.
    """
    if not archive.is_2d():
        raise DimensionalityError("Cannot plot heatmap for non-2D archive.")

    # Try getting the colormap early in case it fails.
    if colormap is None:
        colormap = plt.get_cmap("magma")

    # Retrieve and initialize the axis.
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    ax.set_aspect("equal")
    ax.set_xlim(archive.lower_bounds[0], archive.upper_bounds[0])
    ax.set_ylim(archive.lower_bounds[1], archive.upper_bounds[1])

    # Add faraway points so that the edge regions of the Voronoi diagram are
    # filled in. Refer to
    # https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
    # for more info.
    interval = archive.upper_bounds - archive.lower_bounds
    scale = 1000
    faraway_pts = [
        archive.upper_bounds + interval * scale,  # Far upper right.
        archive.upper_bounds + interval * [-1, 1] * scale,  # Far upper left.
        archive.lower_bounds + interval * [-1, -1] * scale,  # Far bottom left.
        archive.lower_bounds + interval * [1, -1] * scale,  # Far bottom right.
    ]
    vor = Voronoi(np.append(archive.centroids, faraway_pts, axis=0))

    # Calculate objective value for each region. `vor.point_region` contains
    # the region index of each point.
    region_obj = [None] * len(vor.regions)
    min_obj, max_obj = np.inf, np.NINF
    pt_to_obj = _get_pt_to_obj(archive)
    for pt_idx, region_idx in enumerate(
            vor.point_region[:-4]):  # Exclude faraway_pts.
        if region_idx != -1 and pt_idx in pt_to_obj:
            obj = pt_to_obj[pt_idx]
            min_obj = min(min_obj, obj)
            max_obj = max(max_obj, obj)
            region_obj[region_idx] = obj

    # Shade the regions.
    for region, objective in zip(vor.regions, region_obj):
        # This check is O(n), but n is typically small, and creating
        # `polygon` is also O(n) anyway.
        if -1 not in region:
            if objective is None:
                color = "white"
            else:
                normalized_obj = (objective - min_obj) / (max_obj - min_obj)
                color = colormap(normalized_obj)
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), color=color, ec="k", lw=0.5)

    # Create a colorbar.
    mappable = ScalarMappable(cmap=colormap)
    mappable.set_clim(min_obj, max_obj)
    fig.colorbar(mappable, ax=ax, pad=0.1)

    # Plot the sample points and centroids.
    if plot_samples:
        ax.plot(archive.samples[:, 0],
                archive.samples[:, 1],
                "o",
                c="gray",
                ms=1)
    ax.plot(archive.centroids[:, 0], archive.centroids[:, 1], "ko", ms=3)

    # Save figure if necessary.
    if filename is not None:
        fig.savefig(filename)

    return fig, ax
