"""Miscellaneous visualization tools.

Note that this module only works when you install ``ribs[all]``. As such, we do
not import it when you run ``import ribs``, and you will need to explicitly use
``import ribs.visualize``.

.. autosummary::
    :toctree:

    ribs.visualize.cvt_archive_heatmap
    ribs.visualize.sliding_boundary_archive_heatmap
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from scipy.spatial import Voronoi  # pylint: disable=no-name-in-module

# Matplotlib functions tend to have a ton of args.
# pylint: disable = too-many-arguments
# pylint: disable = invalid-name

__all__ = [
    "cvt_archive_heatmap",
    "sliding_boundary_archive_heatmap",
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
    # 5 columns (1 index, 2 behavior, 1 objective, 1 solution).
    data = cvt_archive.as_pandas()

    pt_to_obj = {}
    for _, row in data.iterrows():
        pt_to_obj[row["index"]] = row["objective"]
    return pt_to_obj


def cvt_archive_heatmap(archive,
                        ax=None,
                        plot_centroids=True,
                        plot_samples=False,
                        transpose_bcs=False,
                        cmap="magma",
                        square=False,
                        ms=1,
                        lw=0.5,
                        vmin=None,
                        vmax=None):
    """Plots heatmap of a 2D :class:`ribs.archives.CVTArchive`.

    Essentially, we create a Voronoi diagram and shade in each cell with a
    color corresponding to the value of that cell's elite.

    Depending on how many bins you have in your archive, you may need to tune
    the values of ``ms`` and ``lw``. If there are too many bins, the Voronoi
    diagram and centroid markers will make the entire image appear black. In
    that case, you can turn off the centroids with ``plot_centroids=False`` and
    even remove the lines completely with ``lw=0.0``.

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
        transpose_bcs (bool): By default, the first BC in the archive will
            appear along the x-axis, and the second will be along the y-axis. To
            switch this (i.e. to transpose the axes), set this to True.
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
    # pylint: disable = too-many-locals

    if not archive.is_2d:
        raise ValueError("Cannot plot heatmap for non-2D archive.")

    # Try getting the colormap early in case it fails.
    cmap = _retrieve_cmap(cmap)

    # Retrieve data from archive.
    lower_bounds = archive.lower_bounds
    upper_bounds = archive.upper_bounds
    centroids = archive.centroids
    samples = archive.samples
    if transpose_bcs:
        lower_bounds = np.flip(lower_bounds)
        upper_bounds = np.flip(upper_bounds)
        centroids = np.flip(centroids, axis=1)
        samples = np.flip(samples, axis=1)

    # Retrieve and initialize the axis.
    ax = plt.gca() if ax is None else ax
    ax.set_xlim(lower_bounds[0], upper_bounds[0])
    ax.set_ylim(lower_bounds[1], upper_bounds[1])
    if square:
        ax.set_aspect("equal")

    # Add faraway points so that the edge regions of the Voronoi diagram are
    # filled in. Refer to
    # https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
    # for more info.
    interval = upper_bounds - lower_bounds
    scale = 1000
    faraway_pts = [
        upper_bounds + interval * scale,  # Far upper right.
        upper_bounds + interval * [-1, 1] * scale,  # Far upper left.
        lower_bounds + interval * [-1, -1] * scale,  # Far bottom left.
        lower_bounds + interval * [1, -1] * scale,  # Far bottom right.
    ]
    vor = Voronoi(np.append(centroids, faraway_pts, axis=0))

    # Calculate objective value for each region. `vor.point_region` contains
    # the region index of each point.
    region_obj = [None] * len(vor.regions)
    min_obj, max_obj = np.inf, -np.inf
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
        ax.plot(samples[:, 0], samples[:, 1], "o", c="gray", ms=ms)
    if plot_centroids:
        ax.plot(centroids[:, 0], centroids[:, 1], "ko", ms=ms)


def sliding_boundary_archive_heatmap(
    archive,
    ax=None,
    transpose_bcs=False,
    cmap="magma",
    square=False,
    ms=None,
    vmin=None,
    vmax=None,
):
    """Plots heatmap of a 2D :class:`ribs.archives.SlidingBoundaryArchive`.

    Since the boundaries of :class:`ribs.archives.SlidingBoundaryArchive` is
    dynamic, we plot the heatmap as a scatter plot, in which each marker is a
    solution and its color represents the fitness value.

    Examples:
        .. plot::
            :context: close-figs

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from ribs.archives import SlidingBoundaryArchive
            >>> from ribs.visualize import sliding_boundary_archive_heatmap
            >>> archive = SlidingBoundaryArchive([10, 20],
            >>>                                  [(-1, 1), (-1, 1)],
            >>>                                  seed=42)
            >>> archive.initialize(solution_dim=2)
            >>> # Populate the archive with the negative sphere function.
            >>> rng = np.random.default_rng(10)
            >>> for _ in range(1000):
            >>>     x, y = rng.uniform((-1, -1), (1, 1))
            >>>     archive.add(
            >>>         solution=rng.random(2),
            >>>         objective_value=-(x**2 + y**2),
            >>>         behavior_values=np.array([x, y]),
            >>>     )
            >>> # Plot a heatmap of the archive.
            >>> plt.figure(figsize=(8, 6))
            >>> sliding_boundary_archive_heatmap(archive)
            >>> plt.title("Negative sphere function")
            >>> plt.xlabel("x coords")
            >>> plt.ylabel("y coords")
            >>> plt.show()


    Args:
        archive (SlidingBoundaryArchive): A 2D SlidingBoundaryArchive.
        ax (matplotlib.axes.Axes): Axes on which to plot the heatmap. If None,
            the current axis will be used.
        transpose_bcs (bool): By default, the first BC in the archive will
            appear along the x-axis, and the second will be along the y-axis. To
            switch this (i.e. to transpose the axes), set this to True.
        cmap (str, list, matplotlib.colors.Colormap): Colormap to use when
            plotting intensity. Either the name of a colormap, a list of RGB or
            RGBA colors (i.e. an Nx3 or Nx4 array), or a colormap object.
        square (bool): If True, set the axes aspect raio to be "equal".
        ms (float): Marker size for the solutions.
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

    # Retrieve data from archive.
    archive_data = archive.as_pandas()
    x = archive_data['behavior-0'].to_list()
    y = archive_data['behavior-1'].to_list()
    lower_bounds = archive.lower_bounds
    upper_bounds = archive.upper_bounds
    objective_values = archive_data['objective'].to_list()

    if transpose_bcs:
        y = archive_data['behavior-0'].to_list()
        x = archive_data['behavior-1'].to_list()
        lower_bounds = np.flip(lower_bounds)
        upper_bounds = np.flip(upper_bounds)

    # Initialize the axis.
    ax = plt.gca() if ax is None else ax
    ax.set_xlim(lower_bounds[0], upper_bounds[0])
    ax.set_ylim(lower_bounds[1], upper_bounds[1])

    if square:
        ax.set_aspect("equal")

    # Create the plot.
    ax.scatter(x, y, s=ms, c=objective_values, cmap=cmap)

    # Create the colorbar.
    min_obj = np.min(objective_values) if vmin is None else vmin
    max_obj = np.max(objective_values) if vmax is None else vmax

    mappable = ScalarMappable(cmap=cmap)
    mappable.set_clim(min_obj, max_obj)
    ax.figure.colorbar(mappable, ax=ax, pad=0.1)
