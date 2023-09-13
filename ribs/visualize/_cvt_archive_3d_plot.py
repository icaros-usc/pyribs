"""Provides cvt_archive_3d_plot."""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Voronoi  # pylint: disable=no-name-in-module

from ribs.visualize._utils import retrieve_cmap, validate_heatmap_visual_args


def cvt_archive_3d_plot(archive,
                        ax=None,
                        *,
                        plot_centroids=False,
                        plot_samples=False,
                        transpose_measures=False,
                        cmap="magma",
                        aspect="auto",
                        ms=1,
                        lw=0.5,
                        vmin=None,
                        vmax=None,
                        cbar="auto",
                        cbar_kwargs=None):
    """Plots heatmap of a :class:`~ribs.archives.CVTArchive` with 3D measure
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
        plot_centroids (bool): Whether to plot the cluster centroids.
        plot_samples (bool): Whether to plot the samples used when generating
            the clusters.
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
        ms (float): Marker size for both centroids and samples.
        lw (float): Line width when plotting the voronoi diagram.
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
        ValueError: The archive is not 3D.
    """
    validate_heatmap_visual_args(
        aspect, cbar, archive.measure_dim, [3],
        "This plot can only be made for a 3D CVTArchive")

    if aspect is None:
        aspect = "auto"

    # Try getting the colormap early in case it fails.
    cmap = retrieve_cmap(cmap)

    # Retrieve data from archive.
    lower_bounds = archive.lower_bounds
    upper_bounds = archive.upper_bounds
    centroids = archive.centroids
    samples = archive.samples

    # TODO: Measure order
    if transpose_measures:
        lower_bounds = np.flip(lower_bounds)
        upper_bounds = np.flip(upper_bounds)
        centroids = np.flip(centroids, axis=1)
        samples = np.flip(samples, axis=1)

    # Retrieve and initialize the axis.
    # TODO: Is this the best way to get the 3D axes? Maybe we can get gca() and
    # check if it is 3D before creating our own, or even just throw an error if
    # current axis is not 3D.
    ax = plt.axes(projection="3d") if ax is None else ax
    #  ax.set_xlim(-10, 10)
    #  ax.set_ylim(-10, 10)
    #  ax.set_zlim(-10, 10)
    ax.set_xlim(lower_bounds[0], upper_bounds[0])
    ax.set_ylim(lower_bounds[1], upper_bounds[1])
    ax.set_zlim(lower_bounds[2], upper_bounds[2])
    ax.set_aspect(aspect)

    # TODO: Figure out what points should be added here, if any.

    # Add faraway points so that the edge regions of the Voronoi diagram are
    # filled in. Refer to
    # https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
    # for more info.
    #  interval = upper_bounds - lower_bounds
    #  scale = 1000
    #  faraway_pts = [
    #      upper_bounds + interval * [1, 1, 1] * scale,
    #      upper_bounds + interval * [1, 1, -1] * scale,
    #      lower_bounds + interval * [1, -1, 1] * scale,
    #      lower_bounds + interval * [1, -1, -1] * scale,
    #      upper_bounds + interval * [-1, 1, 1] * scale,
    #      upper_bounds + interval * [-1, 1, -1] * scale,
    #      lower_bounds + interval * [-1, -1, 1] * scale,
    #      lower_bounds + interval * [-1, -1, -1] * scale,
    #  ]
    #  vor = Voronoi(np.append(centroids, faraway_pts, axis=0))

    # TODO: Try point reflections as is done here:
    # https://stackoverflow.com/questions/28665491/getting-a-bounded-polygon-coordinates-from-voronoi-cells

    # Point reflections
    xmin, ymin, zmin = lower_bounds
    xmax, ymax, zmax = upper_bounds
    (
        xmin_reflec,
        ymin_reflec,
        zmin_reflec,
        xmax_reflec,
        ymax_reflec,
        zmax_reflec,
    ) = [centroids.copy() for _ in range(6)]

    xmin_reflec[:, 0] = xmin - (centroids[:, 0] - xmin)
    ymin_reflec[:, 1] = ymin - (centroids[:, 1] - ymin)
    zmin_reflec[:, 2] = zmin - (centroids[:, 2] - zmin)
    xmax_reflec[:, 0] = xmax + (xmax - centroids[:, 0])
    ymax_reflec[:, 1] = ymax + (ymax - centroids[:, 1])
    zmax_reflec[:, 2] = zmax + (zmax - centroids[:, 2])

    centroids = np.concatenate(
        (centroids, xmin_reflec, ymin_reflec, zmin_reflec, xmax_reflec,
         ymax_reflec, zmax_reflec))
    vor = Voronoi(centroids)

    #  print("Centroids:", len(centroids))
    #  print("Vertices:", vor.vertices)

    eps = 1e-6  # TODO: Parameter

    vertices = []
    for ridge in vor.ridge_vertices:
        if -1 in ridge:
            continue
        p = vor.vertices[ridge]
        # Using epsilon makes us more tolerant to polygons at the edge of the
        # plot -- due to numerical errors, surfaces at the edges of the plot
        # tended to get clipped.
        if np.any((p < (lower_bounds - eps)) | (p > (upper_bounds + eps))):
            continue
        vertices.append(p)
    ax.add_collection(
        Poly3DCollection(
            vertices,
            edgecolor=[(0.0, 0.0, 0.0, 0.2) for _ in vertices],
            facecolor=["none" for _ in vertices],
        ))

    #  # Calculate objective value for each region. `vor.point_region` contains
    #  # the region index of each point.
    #  region_obj = [None] * len(vor.regions)
    #  min_obj, max_obj = np.inf, -np.inf
    #  pt_to_obj = {elite.index: elite.objective for elite in archive}
    #  for pt_idx, region_idx in enumerate(
    #          vor.point_region[:-4]):  # Exclude faraway_pts.
    #      if region_idx != -1 and pt_idx in pt_to_obj:
    #          obj = pt_to_obj[pt_idx]
    #          min_obj = min(min_obj, obj)
    #          max_obj = max(max_obj, obj)
    #          region_obj[region_idx] = obj

    #  # Override objective value range.
    #  min_obj = min_obj if vmin is None else vmin
    #  max_obj = max_obj if vmax is None else vmax

    #  # Shade the regions.
    #  #
    #  # Note: by default, the first region will be an empty list -- see:
    #  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html
    #  # However, this empty region is ignored by ax.fill since `polygon` is also
    #  # an empty list in this case.
    #  for region, objective in zip(vor.regions, region_obj):
    #      # This check is O(n), but n is typically small, and creating
    #      # `polygon` is also O(n) anyway.
    #      if -1 not in region:
    #          if objective is None:
    #              color = "white"
    #          else:
    #              normalized_obj = np.clip(
    #                  (objective - min_obj) / (max_obj - min_obj), 0.0, 1.0)
    #              color = cmap(normalized_obj)
    #          polygon = vor.vertices[region]
    #          ax.fill(*zip(*polygon), color=color, ec="k", lw=lw)

    #  # Create a colorbar.
    #  mappable = ScalarMappable(cmap=cmap)
    #  mappable.set_clim(min_obj, max_obj)

    #  # Plot the sample points and centroids.
    #  if plot_samples:
    #      ax.plot(samples[:, 0], samples[:, 1], "o", c="gray", ms=ms)
    if plot_centroids:
        ax.plot(centroids[:, 0], centroids[:, 1], centroids[:, 2], "ko", ms=ms)

    #  # Create color bar.
    #  _set_cbar(mappable, ax, cbar, cbar_kwargs)

    # TODO: Clean up this point plotting code.
    # TODO: Account for different axes orders.

    # Retrieve data from archive.
    df = archive.as_pandas()
    measures_batch = df.measures_batch()
    x = measures_batch[:, 0]
    y = measures_batch[:, 1]
    z = measures_batch[:, 2]

    objective_batch = df.objective_batch()
    vmin = np.min(objective_batch) if vmin is None else vmin
    vmax = np.max(objective_batch) if vmax is None else vmax
    #  t = ax.scatter(x,
    #                 y,
    #                 z,
    #                 s=ms,
    #                 c=objective_batch,
    #                 cmap=cmap,
    #                 vmin=vmin,
    #                 vmax=vmax)
