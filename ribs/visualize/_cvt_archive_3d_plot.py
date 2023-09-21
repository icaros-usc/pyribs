"""Provides cvt_archive_3d_plot."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Voronoi  # pylint: disable=no-name-in-module

from ribs.visualize._utils import (retrieve_cmap, set_cbar, validate_df,
                                   validate_heatmap_visual_args)


def cvt_archive_3d_plot(
    archive,
    ax=None,
    *,
    df=None,
    measure_order=None,
    cmap="magma",
    lw=0.5,
    ec=(0.0, 0.0, 0.0, 0.1),
    cell_alpha=1.0,
    vmin=None,
    vmax=None,
    cbar="auto",
    cbar_kwargs=None,
    plot_elites=False,
    elite_ms=100,
    elite_alpha=0.5,
    plot_centroids=False,
    plot_samples=False,
    ms=1,
):
    """Plots a :class:`~ribs.archives.CVTArchive` with 3D measure space.

    This function relies on Matplotlib's `mplot3d
    <https://matplotlib.org/stable/tutorials/toolkits/mplot3d.html>`_ toolkit.
    By default, this function plots a 3D Voronoi diagram of the cells in the
    archive and shades each cell based on its objective value. It is also
    possible to plot a "wireframe" with only the cells' boundaries, along with a
    dot inside each cell indicating its objective value.

    Depending on how many cells are in the archive, ``ms`` and ``lw`` may need
    to be tuned. If there are too many cells, the Voronoi diagram and centroid
    markers will make the entire image appear black. In that case, try turning
    off the centroids with ``plot_centroids=False`` or even removing the lines
    completely with ``lw=0``.

    Examples:

        .. plot::
            :context: close-figs

            3D Plot with Solid Cells

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from ribs.archives import CVTArchive
            >>> from ribs.visualize import cvt_archive_3d_plot
            >>> # Populate the archive with the negative sphere function.
            >>> archive = CVTArchive(solution_dim=2,
            ...                      cells=500,
            ...                      ranges=[(-2, 0), (-2, 0), (-2, 0)])
            >>> x = np.random.uniform(-2, 0, 5000)
            >>> y = np.random.uniform(-2, 0, 5000)
            >>> z = np.random.uniform(-2, 0, 5000)
            >>> archive.add(solution_batch=np.stack((x, y), axis=1),
            ...             objective_batch=-(x**2 + y**2 + z**2),
            ...             measures_batch=np.stack((x, y, z), axis=1))
            >>> # Plot the archive.
            >>> plt.figure(figsize=(8, 6))
            >>> cvt_archive_3d_plot(archive)
            >>> plt.title("Negative sphere function with 3D measures")
            >>> plt.show()

        .. plot::
            :context: close-figs

            3D Plot with Translucent Cells

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from ribs.archives import CVTArchive
            >>> from ribs.visualize import cvt_archive_3d_plot
            >>> # Populate the archive with the negative sphere function.
            >>> archive = CVTArchive(solution_dim=2,
            ...                      cells=500,
            ...                      ranges=[(-2, 0), (-2, 0), (-2, 0)])
            >>> x = np.random.uniform(-2, 0, 5000)
            >>> y = np.random.uniform(-2, 0, 5000)
            >>> z = np.random.uniform(-2, 0, 5000)
            >>> archive.add(solution_batch=np.stack((x, y), axis=1),
            ...             objective_batch=-(x**2 + y**2 + z**2),
            ...             measures_batch=np.stack((x, y, z), axis=1))
            >>> # Plot the archive.
            >>> plt.figure(figsize=(8, 6))
            >>> cvt_archive_3d_plot(archive, cell_alpha=0.1)
            >>> plt.title("Negative sphere function with 3D measures")
            >>> plt.show()

        .. plot::
            :context: close-figs

            3D "Wireframe" (Shading Turned Off)

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from ribs.archives import CVTArchive
            >>> from ribs.visualize import cvt_archive_3d_plot
            >>> # Populate the archive with the negative sphere function.
            >>> archive = CVTArchive(solution_dim=2,
            ...                      cells=100,
            ...                      ranges=[(-2, 0), (-2, 0), (-2, 0)])
            >>> x = np.random.uniform(-2, 0, 1000)
            >>> y = np.random.uniform(-2, 0, 1000)
            >>> z = np.random.uniform(-2, 0, 1000)
            >>> archive.add(solution_batch=np.stack((x, y), axis=1),
            ...             objective_batch=-(x**2 + y**2 + z**2),
            ...             measures_batch=np.stack((x, y, z), axis=1))
            >>> # Plot the archive.
            >>> plt.figure(figsize=(8, 6))
            >>> cvt_archive_3d_plot(archive, cell_alpha=0.0)
            >>> plt.title("Negative sphere function with 3D measures")
            >>> plt.show()

        .. plot::
            :context: close-figs

            3D Wireframe with Elites as Scatter Plot

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from ribs.archives import CVTArchive
            >>> from ribs.visualize import cvt_archive_3d_plot
            >>> # Populate the archive with the negative sphere function.
            >>> archive = CVTArchive(solution_dim=2,
            ...                      cells=100,
            ...                      ranges=[(-2, 0), (-2, 0), (-2, 0)])
            >>> x = np.random.uniform(-2, 0, 1000)
            >>> y = np.random.uniform(-2, 0, 1000)
            >>> z = np.random.uniform(-2, 0, 1000)
            >>> archive.add(solution_batch=np.stack((x, y), axis=1),
            ...             objective_batch=-(x**2 + y**2 + z**2),
            ...             measures_batch=np.stack((x, y, z), axis=1))
            >>> # Plot the archive.
            >>> plt.figure(figsize=(8, 6))
            >>> cvt_archive_3d_plot(archive, cell_alpha=0.0, plot_elites=True)
            >>> plt.title("Negative sphere function with 3D measures")
            >>> plt.show()

    Args:
        archive (CVTArchive): A 3D :class:`~ribs.archives.CVTArchive`.
        ax (matplotlib.axes.Axes): Axes on which to plot the heatmap.
            If ``None``, we will create a new 3D axis.
        df (ribs.archives.ArchiveDataFrame): If provided, we will plot data from
            this argument instead of the data currently in the archive. This
            data can be obtained by, for instance, calling
            :meth:`ribs.archives.ArchiveBase.as_pandas()` and modifying the
            resulting :class:`ArchiveDataFrame`. Note that, at a minimum, the
            data must contain columns for index, objective, and measures. To
            display a custom metric, replace the "objective" column.
        measure_order (array-like of int): Specifies the axes order for plotting
            the measures. By default, the first measure (measure 0) in the
            archive appears on the x-axis, the second (measure 1) on y-axis, and
            third (measure 2) on z-axis. This argument is an array of length 3
            that specifies which measure should appear on the x, y, and z axes.
            For instance, [2, 1, 0] will put measure 2 on the x-axis, measure 1
            on the y-axis, and measure 0 on the z-axis.
        cmap (str, list, matplotlib.colors.Colormap): The colormap to use when
            plotting intensity. Either the name of a
            :class:`~matplotlib.colors.Colormap`, a list of RGB or RGBA colors
            (i.e. an :math:`N \\times 3` or :math:`N \\times 4` array), or a
            :class:`~matplotlib.colors.Colormap` object.
        lw (float): Line width when plotting the Voronoi diagram.
        ec (matplotlib color): Edge color of the cells in the Voronoi diagram.
            See `here
            <https://matplotlib.org/stable/tutorials/colors/colors.html>`_ for
            more info on specifying colors in Matplotlib.
        cell_alpha: Alpha value for the cell colors. Set to 1.0 for opaque
            cells; set to 0.0 for fully transparent cells.
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
        plot_elites (bool): If True, we will plot a scatter plot of the elites
            in the archive. The elites will be colored according to their
            objective value.
        elite_ms (float): Marker size for plotting elites.
        elite_alpha (float): Alpha value for plotting elites.
        plot_centroids (bool): Whether to plot the cluster centroids.
        plot_samples (bool): Whether to plot the samples used when generating
            the clusters.
        ms (float): Marker size for both centroids and samples.

    Raises:
        ValueError: The archive's measure dimension must be 1D or 2D.
        ValueError: ``measure_order`` is not a permutation of ``[0, 1, 2]``.
        ValueError: ``plot_samples`` is passed in but the archive does not have
            samples (e.g., due to using custom centroids during construction).
    """
    # We don't have an aspect arg here so we just pass None.
    validate_heatmap_visual_args(
        None, cbar, archive.measure_dim, [3],
        "This plot can only be made for a 3D CVTArchive")

    if plot_samples and archive.samples is None:
        raise ValueError("Samples are not available for this archive, but "
                         "`plot_samples` was passed in.")

    # Try getting the colormap early in case it fails.
    cmap = retrieve_cmap(cmap)

    # Retrieve archive data.
    df = archive.as_pandas() if df is None else validate_df(df)
    objective_batch = df.objective_batch()
    measures_batch = df.measures_batch()
    lower_bounds = archive.lower_bounds
    upper_bounds = archive.upper_bounds
    centroids = archive.centroids
    samples = archive.samples

    if measure_order is not None:
        if sorted(measure_order) != [0, 1, 2]:
            raise ValueError(
                "measure_order should be a permutation of [0, 1, 2] but "
                f"received {measure_order}")
        measures_batch = measures_batch[:, measure_order]
        lower_bounds = lower_bounds[measure_order]
        upper_bounds = upper_bounds[measure_order]
        centroids = centroids[:, measure_order]
        samples = samples[:, measure_order]

    # Compute objective value range.
    min_obj = np.min(objective_batch) if vmin is None else vmin
    max_obj = np.max(objective_batch) if vmax is None else vmax

    # If the min and max are the same, we set a sensible default range.
    if min_obj == max_obj:
        min_obj, max_obj = min_obj - 0.01, max_obj + 0.01

    # Default ax behavior.
    if ax is None:
        ax = plt.axes(projection="3d")

    ax.set_xlim(lower_bounds[0], upper_bounds[0])
    ax.set_ylim(lower_bounds[1], upper_bounds[1])
    ax.set_zlim(lower_bounds[2], upper_bounds[2])

    # Create reflections of the points so that we can easily find the polygons
    # at the edge of the Voronoi diagram. See here for the basic idea in 2D:
    # https://stackoverflow.com/questions/28665491/getting-a-bounded-polygon-coordinates-from-voronoi-cells
    #
    # Note that this indeed results in us creating a Voronoi diagram with 7
    # times the cells we need. However, the Voronoi creation is still pretty
    # fast.
    #
    # Note that the above StackOverflow approach proposes filtering the points
    # after creating the Voronoi diagram by checking if they are outside the
    # upper or lower bounds. We found that this approach works fine, but it
    # requires subtracting an epsilon from the lower bounds and adding an
    # epsilon to the upper bounds, to allow for some margin of error due to
    # numerical stability. Otherwise, some of the edge polygons will be clipped.
    # Below, we do not filter with this method; instead, we just check whether
    # the point on each side of the ridge is one of the original centroids.
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

    vor = Voronoi(
        np.concatenate((centroids, xmin_reflec, ymin_reflec, zmin_reflec,
                        xmax_reflec, ymax_reflec, zmax_reflec)))

    # Collect the vertices of the ridges of each cell -- the boundary between
    # two points in a Voronoi diagram is referred to as a ridge; in 3D, the
    # ridge is a planar polygon; in 2D, the ridge is a line.
    vertices = []
    objs = []  # Also record objective for each ridge so we can color it.

    # Map from centroid index to objective.
    pt_to_obj = dict(zip(df.index_batch(), objective_batch))

    # The points in the Voronoi diagram are indexed by their placement in the
    # input list. Above, when we called Voronoi, `centroids` were placed first,
    # so the centroid points all have indices less than len(centroids).
    max_centroid_idx = len(centroids)

    for ridge_points, ridge_vertices in zip(vor.ridge_points,
                                            vor.ridge_vertices):
        a, b = ridge_points
        # Record the ridge. We are only interested in a ridge if it involves one
        # of our centroid points, hence the check for max_idx.
        #
        # Note that we record the ridge twice if a and b are both valid points,
        # so we end up plotting the same polygon twice. Unclear how to resolve
        # this, but it seems to show up fine as is.
        if a < max_centroid_idx:
            vertices.append(vor.vertices[ridge_vertices])
            # NaN indicates the cell was not filled and thus had no objective.
            objs.append(pt_to_obj.get(a, np.nan))
        if b < max_centroid_idx:
            vertices.append(vor.vertices[ridge_vertices])
            objs.append(pt_to_obj.get(b, np.nan))

    # Collect and normalize objs that need to be passed through cmap.
    objs = np.asarray(objs)
    cmap_idx = ~np.isnan(objs)
    cmap_objs = objs[cmap_idx]
    normalized_objs = np.clip(
        (np.asarray(cmap_objs) - min_obj) / (max_obj - min_obj), 0.0, 1.0)

    # Create an array of facecolors in RGBA format that defaults to transparent
    # white.
    facecolors = np.full((len(objs), 4), [1.0, 1.0, 1.0, 0.0])

    # Set colors based on objectives. Also set alpha, which is located in index
    # 3 since this is RGBA format.
    facecolors[cmap_idx] = cmap(normalized_objs)
    facecolors[cmap_idx, 3] = cell_alpha

    ax.add_collection(
        Poly3DCollection(
            vertices,
            edgecolor=[ec for _ in vertices],
            facecolor=facecolors,
            lw=lw,
        ))

    if plot_elites:
        ax.scatter(measures_batch[:, 0],
                   measures_batch[:, 1],
                   measures_batch[:, 2],
                   s=elite_ms,
                   c=objective_batch,
                   cmap=cmap,
                   vmin=vmin,
                   vmax=vmax,
                   lw=0.0,
                   alpha=elite_alpha)
    if plot_samples:
        ax.plot(samples[:, 0],
                samples[:, 1],
                samples[:, 2],
                "o",
                c="grey",
                ms=ms)
    if plot_centroids:
        ax.plot(centroids[:, 0],
                centroids[:, 1],
                centroids[:, 2],
                "o",
                c="black",
                ms=ms)

    # Create color bar.
    mappable = ScalarMappable(cmap=cmap)
    mappable.set_clim(min_obj, max_obj)
    set_cbar(mappable, ax, cbar, cbar_kwargs)
