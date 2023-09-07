"""Provides cvt_archive_heatmap."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.cm import ScalarMappable
from scipy.spatial import Voronoi  # pylint: disable=no-name-in-module

from ribs.visualize._utils import (retrieve_cmap, set_cbar,
                                   validate_heatmap_visual_args)

# Matplotlib functions tend to have a ton of args.
# pylint: disable = too-many-arguments


def cvt_archive_heatmap(archive,
                        ax=None,
                        *,
                        transpose_measures=False,
                        cmap="magma",
                        aspect="auto",
                        lw=0.5,
                        ec="black",
                        vmin=None,
                        vmax=None,
                        cbar="auto",
                        cbar_kwargs=None,
                        clip=False,
                        plot_centroids=False,
                        plot_samples=False,
                        ms=1):
    """Plots heatmap of a :class:`~ribs.archives.CVTArchive` with 2D measure
    space.

    Essentially, we create a Voronoi diagram and shade in each cell with a
    color corresponding to the objective value of that cell's elite.

    Depending on how many cells are in the archive, ``ms`` and ``lw`` may need
    to be tuned. If there are too many cells, the Voronoi diagram and centroid
    markers will make the entire image appear black. In that case, try turning
    off the centroids with ``plot_centroids=False`` or even removing the lines
    completely with ``lw=0``.

    Examples:

        .. plot::
            :context: close-figs

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from ribs.archives import CVTArchive
            >>> from ribs.visualize import cvt_archive_heatmap
            >>> # Populate the archive with the negative sphere function.
            >>> archive = CVTArchive(solution_dim=2,
            ...                      cells=100, ranges=[(-1, 1), (-1, 1)])
            >>> x = y = np.linspace(-1, 1, 100)
            >>> xxs, yys = np.meshgrid(x, y)
            >>> xxs, yys = xxs.flatten(), yys.flatten()
            >>> archive.add(solution_batch=np.stack((xxs, yys), axis=1),
            ...             objective_batch=-(xxs**2 + yys**2),
            ...             measures_batch=np.stack((xxs, yys), axis=1))
            >>> # Plot a heatmap of the archive.
            >>> plt.figure(figsize=(8, 6))
            >>> cvt_archive_heatmap(archive)
            >>> plt.title("Negative sphere function")
            >>> plt.xlabel("x coords")
            >>> plt.ylabel("y coords")
            >>> plt.show()

    Args:
        archive (CVTArchive): A 2D :class:`~ribs.archives.CVTArchive`.
        ax (matplotlib.axes.Axes): Axes on which to plot the heatmap.
            If ``None``, the current axis will be used.
        transpose_measures (bool): By default, the first measure in the archive
            will appear along the x-axis, and the second will be along the
            y-axis. To switch this behavior (i.e. to transpose the axes), set
            this to ``True``.
        cmap (str, list, matplotlib.colors.Colormap): The colormap to use when
            plotting intensity. Either the name of a
            :class:`~matplotlib.colors.Colormap`, a list of RGB or RGBA colors
            (i.e. an :math:`N \\times 3` or :math:`N \\times 4` array), or a
            :class:`~matplotlib.colors.Colormap` object.
        aspect ('auto', 'equal', float): The aspect ratio of the heatmap (i.e.
            height/width). Defaults to ``'auto'``. ``'equal'`` is the same as
            ``aspect=1``.
        lw (float): Line width when plotting the Voronoi diagram.
        ec (matplotlib color): Edge color of the cells in the Voronoi diagram.
            See `here
            <https://matplotlib.org/stable/tutorials/colors/colors.html>`_ for
            more info on specifying colors in Matplotlib.
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
        clip (bool, shapely.Polygon): Clip the heatmap cells to a given polygon.
            By default, we draw the cells along the outer edges of the heatmap
            as polygons that extend beyond the archive bounds, but these
            polygons are hidden because we set the axis limits to be the archive
            bounds. Passing `clip=True` will clip the heatmap such that these
            "outer edge" polygons are within the archive bounds. An arbitrary
            polygon can also be passed in to clip the heatmap to a custom shape.
            See `#356 <https://github.com/icaros-usc/pyribs/pull/356>`_ for more
            info.
        plot_centroids (bool): Whether to plot the cluster centroids.
        plot_samples (bool): Whether to plot the samples used when generating
            the clusters.
        ms (float): Marker size for both centroids and samples.

    Raises:
        ValueError: The archive is not 2D.
        ValueError: ``plot_samples`` is passed in but the archive does not have
            samples (e.g., due to using custom centroids during construction).
    """
    validate_heatmap_visual_args(
        aspect, cbar, archive.measure_dim, [2],
        "Heatmaps can only be plotted for 2D CVTArchive")

    if plot_samples and archive.samples is None:
        raise ValueError("Samples are not available for this archive, but "
                         "`plot_samples` was passed in.")

    if aspect is None:
        aspect = "auto"

    # Try getting the colormap early in case it fails.
    cmap = retrieve_cmap(cmap)

    # Retrieve data from archive.
    lower_bounds = archive.lower_bounds
    upper_bounds = archive.upper_bounds
    centroids = archive.centroids
    if transpose_measures:
        lower_bounds = np.flip(lower_bounds)
        upper_bounds = np.flip(upper_bounds)
        centroids = np.flip(centroids, axis=1)

    # If clip is on, make it default to an archive bounding box.
    if clip and not isinstance(clip, shapely.Polygon):
        clip = shapely.box(lower_bounds[0], lower_bounds[1], upper_bounds[0],
                           upper_bounds[1])

    if plot_samples:
        samples = archive.samples
        if transpose_measures:
            samples = np.flip(samples, axis=1)

    # Retrieve and initialize the axis.
    ax = plt.gca() if ax is None else ax
    ax.set_xlim(lower_bounds[0], upper_bounds[0])
    ax.set_ylim(lower_bounds[1], upper_bounds[1])
    ax.set_aspect(aspect)

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
    pt_to_obj = {elite.index: elite.objective for elite in archive}
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

    # If the min and max are the same, we set a sensible default range.
    if min_obj == max_obj:
        min_obj, max_obj = min_obj - 0.01, max_obj + 0.01

    # Vertices of all cells.
    vertices = []
    # The facecolor of each cell. Shape (n_regions, 4) for RGBA format, but we
    # do not know n_regions in advance.
    facecolors = []
    # Boolean array indicating which of the facecolors needs to be computed with
    # the cmap. The other colors correspond to empty cells. Shape (n_regions,)
    facecolor_cmap_mask = []
    # The objective corresponding to the regions which must be passed through
    # the cmap. Shape (sum(facecolor_cmap_mask),)
    facecolor_objs = []

    # Cycle through the regions to set up polygon vertices and facecolors.
    for region, objective in zip(vor.regions, region_obj):
        # Checking for -1 is O(n), but n is typically small.
        #
        # We check length since the first region is an empty list by default:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html
        if -1 in region or len(region) == 0:
            continue

        if clip:
            # Clip the cell vertices to the polygon. Clipping may cause some
            # cells to split into two or more polygons, especially if the clip
            # polygon has holes.
            polygon = shapely.Polygon(vor.vertices[region])
            intersection = polygon.intersection(clip)
            if isinstance(intersection, shapely.MultiPolygon):
                for polygon in intersection.geoms:
                    vertices.append(polygon.exterior.coords)
                n_splits = len(intersection.geoms)
            else:
                # The intersection is a single Polygon.
                vertices.append(intersection.exterior.coords)
                n_splits = 1
        else:
            vertices.append(vor.vertices[region])
            n_splits = 1

        # Repeat values for each split.
        for _ in range(n_splits):
            if objective is None:
                # Transparent white (RGBA format) -- this ensures that if a
                # figure is saved with a transparent background, the empty cells
                # will also be transparent.
                facecolors.append(np.array([1.0, 1.0, 1.0, 0.0]))
                facecolor_cmap_mask.append(False)
            else:
                facecolors.append(np.empty(4))
                facecolor_cmap_mask.append(True)
                facecolor_objs.append(objective)

    # Compute facecolors from the cmap. We first normalize the objectives and
    # clip them to [0, 1].
    normalized_objs = np.clip(
        (np.asarray(facecolor_objs) - min_obj) / (max_obj - min_obj), 0.0, 1.0)
    facecolors = np.asarray(facecolors)
    facecolors[facecolor_cmap_mask] = cmap(normalized_objs)

    # Plot the collection on the axes. Note that this is faster than plotting
    # each polygon individually with ax.fill().
    ax.add_collection(
        matplotlib.collections.PolyCollection(
            vertices,
            edgecolors=ec,
            facecolors=facecolors,
            linewidths=lw,
        ))

    # Create a colorbar.
    mappable = ScalarMappable(cmap=cmap)
    mappable.set_clim(min_obj, max_obj)

    # Plot the sample points and centroids.
    if plot_samples:
        ax.plot(samples[:, 0], samples[:, 1], "o", c="grey", ms=ms)
    if plot_centroids:
        ax.plot(centroids[:, 0], centroids[:, 1], "o", c="black", ms=ms)

    # Create color bar.
    set_cbar(mappable, ax, cbar, cbar_kwargs)