"""Miscellaneous visualization tools.

Note that this module only works when you install ``ribs[all]``. As such, we do
not import it when you run ``import ribs``, and you will need to explicitly use
``import ribs.visualize``.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from scipy.spatial import Voronoi  # pylint: disable=no-name-in-module

__all__ = [
    "cvt_archive_heatmap",
]


def _retrieve_cmap(cmap):
    """Retrieves colormap from matplotlib."""
    if isinstance(cmap, str):
        return matplotlib.cm.get_cmap(cmap)
    if isinstance(cmap, list):
        return matplotlib.colors.ListedColormap(cmap)
    return cmap


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
                        ax=None,
                        plot_centroids=True,
                        plot_samples=False,
                        cmap="magma",
                        square=True,
                        ms=1,
                        lw=0.5,
                        vmin=None,
                        vmax=None):
    """Plots heatmap of a 2D :class:`ribs.archives.CVTArchive`.

    Essentially, we create a Voronoi diagram and shade in each cell with a
    color corresponding to the value of that cell's elite.

    When you create a figure for plotting this heatmap, we recommend having an
    aspect ratio of 4:3. Depending on how many bins you have in your archive,
    you may need to tune the values of ``ms`` and ``lw``. If there are too many
    bins, the Voronoi diagram and centroid markers will make the entire image
    appear black. In that case, you can turn off the centroids with
    ``plot_centroids=False`` and even remove the lines completely with
    ``lw=0.0``.

    Examples:

        .. plot::
            :context: close-figs

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from ribs.archives import CVTArchive
            >>> from ribs.visualize import cvt_archive_heatmap
            >>> # Populate the archive with the negative sphere function.
            >>> archive = CVTArchive([(-1, 1), (-1, 1)], 100)
            >>> archive.initialize(solution_dim=2)
            >>> for x in np.linspace(-1, 1, 100):
            ...     for y in np.linspace(-1, 1, 100):
            ...         archive.add(solution=np.array([x,y]),
            ...                     objective_value=-(x**2 + y**2),
            ...                     behavior_values=np.array([x,y]))
            >>> # Plot a heatmap of the archive.
            >>> plt.figure(figsize=(8, 6))
            >>> cvt_archive_heatmap(archive)
            >>> plt.title("Negative sphere function")
            >>> plt.xlabel("x coords")
            >>> plt.ylabel("y coords")
            >>> plt.show()

    Args:
        archive (CVTArchive): A 2D CVTArchive.
        ax (matplotlib.axes.Axes): Axes on which to plot the heatmap. If None,
            the current axis will be used.
        plot_centroids (bool): Whether to plot the cluster centroids.
        plot_samples (bool): Whether to plot the samples used when generating
            the clusters.
        cmap (str, list, matplotlib.colors.Colormap): Colormap to use when
            plotting intensity. Either the name of a colormap, a list of RGB or
            RGBA colors (i.e. an Nx3 or Nx4 array), or a colormap object.
        square (bool): If True, set the axes aspect raio to be "equal".
        ms (float): Marker size for both centroids and samples.
        lw (float): Line width when plotting the voronoi diagram.
        vmin (float): Minimum objective value to use in the plot. If None, the
            minimum objective value in the archive is used.
        vmax (float): Maximum objective value to use in the plot. If None, the
            maximum objective value in the archive is used.
    Raises:
        ValueError: The archive is not 2D.
    """
    if not archive.is_2d:
        raise ValueError("Cannot plot heatmap for non-2D archive.")

    # Try getting the colormap early in case it fails.
    cmap = _retrieve_cmap(cmap)

    # Retrieve and initialize the axis.
    if ax is None:
        ax = plt.gca()
    if square:
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

    # Override objective value range.
    min_obj = min_obj if vmin is None else vmin
    max_obj = max_obj if vmax is None else vmax

    # Shade the regions.
    for region, objective in zip(vor.regions, region_obj):
        # This check is O(n), but n is typically small, and creating
        # `polygon` is also O(n) anyway.
        if -1 not in region:
            if objective is None:
                color = "white"
            else:
                normalized_obj = np.clip(
                    (objective - min_obj) / (max_obj - min_obj), 0.0, 1.0)
                color = cmap(normalized_obj)
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), color=color, ec="k", lw=lw)

    # Create a colorbar.
    mappable = ScalarMappable(cmap=cmap)
    mappable.set_clim(min_obj, max_obj)
    ax.figure.colorbar(mappable, ax=ax, pad=0.1)

    # Plot the sample points and centroids.
    if plot_samples:
        ax.plot(archive.samples[:, 0],
                archive.samples[:, 1],
                "o",
                c="gray",
                ms=ms)
    if plot_centroids:
        ax.plot(archive.centroids[:, 0], archive.centroids[:, 1], "ko", ms=ms)
