"""Miscellaneous visualization tools.

Note that this module only works when you install ``ribs[all]``. As such, we do
not import it when you run ``import ribs``, and you will need to use ``import
ribs.visualize``.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from scipy.spatial import Voronoi  # pylint: disable=no-name-in-module

__all__ = [
    "cvt_archive_heatmap",
]


# TODO: figure out how to pass in ax parameter.
def cvt_archive_heatmap(archive,
                        filename=None,
                        plot_samples=False,
                        figsize=(8, 6),
                        colormap="magma"):
    """Plots heatmap of a 2D :class:`ribs.archives.CVTArchive`.

    Essentially, we create a Voronoi diagram and shade in each cell with a
    color corresponding to the value of that cell's elite.

    Raises:
        RuntimeError: The archive is not 2D.
    """
    # TODO: stop accessing this.
    if archive._n_dims != 2:
        raise RuntimeError("Cannot plot heatmap for non-2D archive.")

    colormap = plt.get_cmap(colormap)
    fig, ax = plt.subplots(figsize=figsize)
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
    for index, region_idx in enumerate(
            vor.point_region[:-4]):  # Exclude faraway_pts.
        # TODO: Stop accessing private members -- get solutions from as_pandas()
        if region_idx != -1 and archive._solutions[index] is not None:
            obj = archive._objective_values[index]
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
    ax.plot(archive.centroids[:, 0], archive.centroids[:, 1], "ko")

    if filename is not None:
        fig.savefig(filename)

    return fig, ax
