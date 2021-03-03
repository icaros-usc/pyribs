"""Miscellaneous visualization tools.

These functions are similar to matplotlib functions like
:func:`~matplotlib.pyplot.scatter` and :func:`~matplotlib.pyplot.pcolormesh`.
When called, these functions default to creating plots on the current axis.
After plotting, functions like :func:`~matplotlib.pyplot.xlabel` and
:func:`~matplotlib.pyplot.title` may be used to further modify the axis.
Alternatively, if using maplotlib's object-oriented API, pass the `ax` parameter
to these functions.

.. note:: This module only works with ``ribs[all]`` installed. As such, it is
    not imported with ``import ribs``, and it must be explicitly imported with
    ``import ribs.visualize``.

.. autosummary::
    :toctree:

    ribs.visualize.grid_archive_heatmap
    ribs.visualize.cvt_archive_heatmap
    ribs.visualize.sliding_boundaries_archive_heatmap
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
from scipy.spatial import Voronoi  # pylint: disable=no-name-in-module

# Matplotlib functions tend to have a ton of args.
# pylint: disable = too-many-arguments

__all__ = [
    "grid_archive_heatmap",
    "cvt_archive_heatmap",
    "sliding_boundaries_archive_heatmap",
]


def _retrieve_cmap(cmap):
    """Retrieves colormap from matplotlib."""
    if isinstance(cmap, str):
        return matplotlib.cm.get_cmap(cmap)
    if isinstance(cmap, list):
        return matplotlib.colors.ListedColormap(cmap)
    return cmap


def _get_pt_to_obj(cvt_archive):
    """Creates a dict from centroid index to objective value in a CVTArchive."""
    data = cvt_archive.as_pandas(include_solutions=False)
    pt_to_obj = {}
    for row in data.itertuples():
        # row.index is the centroid index. The dataframe index is row.Index.
        pt_to_obj[row.index] = row.objective
    return pt_to_obj


def grid_archive_heatmap(archive,
                         ax=None,
                         transpose_bcs=False,
                         cmap="magma",
                         square=False,
                         vmin=None,
                         vmax=None,
                         pcm_kwargs=None):
    """Plots heatmap of a :class:`~ribs.archives.GridArchive` with 2D behavior
    space.

    Essentially, we create a grid of cells and shade each cell with a color
    corresponding to the objective value of that cell's elite. This method uses
    :func:`~matplotlib.pyplot.pcolormesh` to generate the grid. For further
    customization, pass extra kwargs to :func:`~matplotlib.pyplot.pcolormesh`
    through the ``pcm_kwargs`` parameter. For instance, to create black
    boundaries of width 0.1, pass in ``pcm_kwargs={"edgecolor": "black",
    "linewidth": 0.1}``.

    Examples:
        .. plot::
            :context: close-figs

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from ribs.archives import GridArchive
            >>> from ribs.visualize import grid_archive_heatmap
            >>> # Populate the archive with the negative sphere function.
            >>> archive = GridArchive([20, 20], [(-1, 1), (-1, 1)])
            >>> archive.initialize(solution_dim=2)
            >>> for x in np.linspace(-1, 1, 100):
            ...     for y in np.linspace(-1, 1, 100):
            ...         archive.add(solution=np.array([x,y]),
            ...                     objective_value=-(x**2 + y**2),
            ...                     behavior_values=np.array([x,y]))
            >>> # Plot a heatmap of the archive.
            >>> plt.figure(figsize=(8, 6))
            >>> grid_archive_heatmap(archive)
            >>> plt.title("Negative sphere function")
            >>> plt.xlabel("x coords")
            >>> plt.ylabel("y coords")
            >>> plt.show()


    Args:
        archive (GridArchive): A 2D GridArchive.
        ax (matplotlib.axes.Axes): Axes on which to plot the heatmap. If None,
            the current axis will be used.
        transpose_bcs (bool): By default, the first BC in the archive will
            appear along the x-axis, and the second will be along the y-axis. To
            switch this (i.e. to transpose the axes), set this to True.
        cmap (str, list, matplotlib.colors.Colormap): Colormap to use when
            plotting intensity. Either the name of a colormap, a list of RGB or
            RGBA colors (i.e. an Nx3 or Nx4 array), or a colormap object.
        square (bool): If True, set the axes aspect ratio to be "equal".
        vmin (float): Minimum objective value to use in the plot. If None, the
            minimum objective value in the archive is used.
        vmax (float): Maximum objective value to use in the plot. If None, the
            maximum objective value in the archive is used.
        pcm_kwargs (dict): Additional kwargs to pass to
            :func:`~matplotlib.pyplot.pcolormesh`.
    Raises:
        ValueError: The archive is not 2D.
    """
    if archive.behavior_dim != 2:
        raise ValueError("Cannot plot heatmap for non-2D archive.")

    # Try getting the colormap early in case it fails.
    cmap = _retrieve_cmap(cmap)

    # Retrieve data from archive.
    lower_bounds = archive.lower_bounds
    upper_bounds = archive.upper_bounds
    x_dim, y_dim = archive.dims
    x_bounds = np.linspace(lower_bounds[0], upper_bounds[0], x_dim + 1)
    y_bounds = np.linspace(lower_bounds[1], upper_bounds[1], y_dim + 1)

    # Color for each cell in the heatmap.
    archive_data = archive.as_pandas(include_solutions=False)
    colors = np.full((y_dim, x_dim), np.nan)
    for row in archive_data.itertuples():
        colors[row.index_1, row.index_0] = row.objective
    objective_values = archive_data["objective"]

    if transpose_bcs:
        # Since the archive is 2D, transpose by swapping the x and y boundaries
        # and by flipping the bounds (the bounds are arrays of length 2).
        x_bounds, y_bounds = y_bounds, x_bounds
        lower_bounds = np.flip(lower_bounds)
        upper_bounds = np.flip(upper_bounds)
        colors = colors.T

    # Initialize the axis.
    ax = plt.gca() if ax is None else ax
    ax.set_xlim(lower_bounds[0], upper_bounds[0])
    ax.set_ylim(lower_bounds[1], upper_bounds[1])

    if square:
        ax.set_aspect("equal")

    # Create the plot.
    pcm_kwargs = {} if pcm_kwargs is None else pcm_kwargs
    vmin = np.min(objective_values) if vmin is None else vmin
    vmax = np.max(objective_values) if vmax is None else vmax
    t = ax.pcolormesh(x_bounds,
                      y_bounds,
                      colors,
                      cmap=cmap,
                      vmin=vmin,
                      vmax=vmax,
                      **pcm_kwargs)

    # Create the colorbar.
    ax.figure.colorbar(t, ax=ax, pad=0.1)


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
    """Plots heatmap of a :class:`~ribs.archives.CVTArchive` with 2D behavior
    space.

    Essentially, we create a Voronoi diagram and shade in each cell with a
    color corresponding to the objective value of that cell's elite.

    Depending on how many bins are in the archive, ``ms`` and ``lw`` may need to
    be tuned. If there are too many bins, the Voronoi diagram and centroid
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
            >>> archive = CVTArchive(100, [(-1, 1), (-1, 1)])
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
        square (bool): If True, set the axes aspect ratio to be "equal".
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

    if archive.behavior_dim != 2:
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


def sliding_boundaries_archive_heatmap(archive,
                                       ax=None,
                                       transpose_bcs=False,
                                       cmap="magma",
                                       square=False,
                                       ms=None,
                                       boundary_lw=0,
                                       vmin=None,
                                       vmax=None):
    """Plots heatmap of a :class:`~ribs.archives.SlidingBoundariesArchive` with
    2D behavior space.

    Since the boundaries of :class:`ribs.archives.SlidingBoundariesArchive` are
    dynamic, we plot the heatmap as a scatter plot, in which each marker is a
    solution and its color represents the objective value. Boundaries can
    optionally be drawn by setting ``boundary_lw`` to a positive value.

    Examples:
        .. plot::
            :context: close-figs

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from ribs.archives import SlidingBoundariesArchive
            >>> from ribs.visualize import sliding_boundaries_archive_heatmap
            >>> archive = SlidingBoundariesArchive([10, 20],
            ...                                  [(-1, 1), (-1, 1)],
            ...                                  seed=42)
            >>> archive.initialize(solution_dim=2)
            >>> # Populate the archive with the negative sphere function.
            >>> rng = np.random.default_rng(10)
            >>> for _ in range(1000):
            ...     x, y = rng.uniform((-1, -1), (1, 1))
            ...     archive.add(
            ...         solution=np.array([x,y]),
            ...         objective_value=-(x**2 + y**2),
            ...         behavior_values=np.array([x, y]),
            ...     )
            >>> # Plot heatmaps of the archive.
            >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
            >>> fig.suptitle("Negative sphere function")
            >>> sliding_boundaries_archive_heatmap(archive, ax=ax1,
            ...                                  boundary_lw=0.5)
            >>> sliding_boundaries_archive_heatmap(archive, ax=ax2)
            >>> ax1.set_title("With boundaries")
            >>> ax2.set_title("Without boundaries")
            >>> ax1.set(xlabel='x coords', ylabel='y coords')
            >>> ax2.set(xlabel='x coords', ylabel='y coords')
            >>> plt.show()


    Args:
        archive (SlidingBoundariesArchive): A 2D SlidingBoundariesArchive.
        ax (matplotlib.axes.Axes): Axes on which to plot the heatmap. If None,
            the current axis will be used.
        transpose_bcs (bool): By default, the first BC in the archive will
            appear along the x-axis, and the second will be along the y-axis. To
            switch this (i.e. to transpose the axes), set this to True.
        cmap (str, list, matplotlib.colors.Colormap): Colormap to use when
            plotting intensity. Either the name of a colormap, a list of RGB or
            RGBA colors (i.e. an Nx3 or Nx4 array), or a colormap object.
        square (bool): If True, set the axes aspect ratio to be "equal".
        ms (float): Marker size for the solutions.
        boundary_lw (float): Line width when plotting the boundaries. Set to 0
            to have no boundaries.
        vmin (float): Minimum objective value to use in the plot. If None, the
            minimum objective value in the archive is used.
        vmax (float): Maximum objective value to use in the plot. If None, the
            maximum objective value in the archive is used.
    Raises:
        ValueError: The archive is not 2D.
    """
    if archive.behavior_dim != 2:
        raise ValueError("Cannot plot heatmap for non-2D archive.")

    # Try getting the colormap early in case it fails.
    cmap = _retrieve_cmap(cmap)

    # Retrieve data from archive.
    archive_data = archive.as_pandas(include_solutions=False)
    x = archive_data["behavior_0"]
    y = archive_data["behavior_1"]
    x_boundary = archive.boundaries[0]
    y_boundary = archive.boundaries[1]
    lower_bounds = archive.lower_bounds
    upper_bounds = archive.upper_bounds
    objective_values = archive_data["objective"]

    if transpose_bcs:
        # Since the archive is 2D, transpose by swapping the x and y behavior
        # values and boundaries and by flipping the bounds (the bounds are
        # arrays of length 2).
        x, y = y, x
        x_boundary, y_boundary = y_boundary, x_boundary
        lower_bounds = np.flip(lower_bounds)
        upper_bounds = np.flip(upper_bounds)

    # Initialize the axis.
    ax = plt.gca() if ax is None else ax
    ax.set_xlim(lower_bounds[0], upper_bounds[0])
    ax.set_ylim(lower_bounds[1], upper_bounds[1])

    if square:
        ax.set_aspect("equal")

    # Create the plot.
    vmin = np.min(objective_values) if vmin is None else vmin
    vmax = np.max(objective_values) if vmax is None else vmax
    t = ax.scatter(x,
                   y,
                   s=ms,
                   c=objective_values,
                   cmap=cmap,
                   vmin=vmin,
                   vmax=vmax)
    if boundary_lw > 0.0:
        ax.vlines(x_boundary,
                  lower_bounds[0],
                  upper_bounds[0],
                  color='k',
                  linewidth=boundary_lw)
        ax.hlines(y_boundary,
                  lower_bounds[1],
                  upper_bounds[1],
                  color='k',
                  linewidth=boundary_lw)

    # Create the colorbar.
    ax.figure.colorbar(t, ax=ax, pad=0.1)


def parallel_axes_plot(archive,
                       ax=None,
                       bc_order=None,
                       axis_labels=None,
                       cmap="magma",
                       linewidth=1.5,
                       alpha=0.8,
                       vmin=None,
                       vmax=None):
    """Plots a parallel axes plot of an archive with an N-Dimensional behavior
    space.

    This visualization is meant to see the coverage of the behavior space at a
    glance. It is possible to glean the correlations between consecutive axes,
    but the results should be interpreted carefully.

    Examples:
        .. plot::
            :context: close-figs

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from ribs.archives import GridArchive
            >>> from ribs.visualize import grid_archive_heatmap
            >>> # Populate the archive with the negative sphere function.
            >>> archive = GridArchive([20, 20, 20], [(-1, 1), (-1, 1), (-1, 1)])
            >>> archive.initialize(solution_dim=2)
            >>> for x in np.linspace(-1, 1, 100):
            ...     for y in np.linspace(-1, 1, 100):
            ...         for z in np.linspace(-1, 1, 100):
            ...             archive.add(solution=np.array([x,y,z]),
            ...                         objective_value=-(x**2 + y**2 + z**2),
            ...                         behavior_values=np.array([x,y,z]))
            >>> # Plot a heatmap of the archive.
            >>> plt.figure(figsize=(8, 6))
            >>> parallel_axes_plot(archive)
            >>> plt.title("Negative sphere function")
            >>> plt.ylabel("axis values")
            >>> plt.show()

    Args:
        archive (SlidingBoundariesArchive): A 2D SlidingBoundariesArchive.
        ax (matplotlib.axes.Axes): Axes on which to plot the heatmap. If None,
            the current axis will be used.
        bc_order (list<str>): By default, the BCs of the parallel axes will
            appear in-order. To switch the axes around, supply a list of the
            behavior axes (e.g., ['behavior_1', 'behavior_0', 'behavior_2']).
        axis_labels(list<str>): By default, the BCs of the parallel axes will be
            titled with the label 'behavior_x'. If the behavior dimensions have
            a meaningful label, they can be supplied in order as the labels.
        cmap (str, list, matplotlib.colors.Colormap): Colormap to use when
            plotting intensity. Either the name of a colormap, a list of RGB or
            RGBA colors (i.e. an Nx3 or Nx4 array), or a colormap object.
        linewidth (float): Line width for solutions on the plot.
        alpha (float): Opacity of solutions on graph (you may want to turn this
            down if there are many solutions that are found to see them)
        vmin (float): Minimum objective value to use in the plot. If None, the
            minimum objective value in the archive is used.
        vmax (float): Maximum objective value to use in the plot. If None, the
            maximum objective value in the archive is used.

    Raises:
        ValueError: The bcs provided do not exist in the archive.
        ValueError: The incorrect number of labels are provided.
    """
    # Try getting the colormap early in case it fails.
    cmap = _retrieve_cmap(cmap)

    #retrieve data from archive
    archive_data = archive.as_pandas(include_solutions=False)

    #if there is no order specified, plot in increasing numerical order
    if bc_order is None:
        cols = [key for key in archive_data.keys() if 'behavior' in key]
        lower_bounds = archive.lower_bounds
        upper_bounds = archive.upper_bounds

    #use the requested behaviors (may be less than the original number of bcs)
    else:
        #check for errors in specification
        for bc in bc_order:
            if bc not in archive_data.keys():
                raise ValueError(f'Behavior {bc} is not in the archive.')
        cols = bc_order
        #find the indices of the requested order
        bc_indices = [int(x.split('_')[1]) for x in bc_order]
        lower_bounds = archive.lower_bounds[bc_indices]
        upper_bounds = archive.upper_bounds[bc_indices]

    if axis_labels is not None and len(axis_labels) != len(cols):
        raise ValueError(f'Label Mismatch: You have {len(cols)} axes'
                         'and {len(axis_labels)} labels.')

    host = plt.gca() if ax is None else ax  #which axis to plot on
    df = archive.as_pandas(include_solutions=False)
    vmin = np.min(df['objective']) if vmin is None else vmin
    vmax = np.max(df['objective']) if vmax is None else vmax
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    objectives = df['objective'].to_numpy()
    ys = df[cols].to_numpy()
    y_ranges = upper_bounds - lower_bounds

    # transform all data to be in the first axis coordinates
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - lower_bounds[1:]) / \
                y_ranges[1:] * y_ranges[0] + lower_bounds[0]

    #copy the axis for the other bcs
    axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
    for i, axis in enumerate(axes):
        axis.set_ylim(lower_bounds[i], upper_bounds[i])
        axis.spines['top'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        if axis != host:
            axis.spines['left'].set_visible(False)
            axis.yaxis.set_ticks_position('right')
            axis.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))

    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    labels = cols if axis_labels is None else axis_labels
    host.set_xticklabels(labels)
    host.tick_params(axis='x', which='major', pad=7)
    host.spines['right'].set_visible(False)
    host.xaxis.tick_top()

    for j in range(len(archive_data)):
        #draw straight lines between the axes in the appropriate color
        color = matplotlib.colors.rgb2hex(cmap(norm(objectives[j])))
        host.plot(range(ys.shape[1]),
                  zs[j, :],
                  c=color,
                  alpha=alpha,
                  linewidth=linewidth)

    # Create a colorbar.
    mappable = ScalarMappable(cmap=cmap)
    mappable.set_clim(vmin, vmax)
    host.figure.colorbar(mappable, pad=0.05, orientation='horizontal')
