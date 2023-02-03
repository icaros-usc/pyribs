"""Miscellaneous visualization tools.

These functions are similar to matplotlib functions like
:func:`~matplotlib.pyplot.scatter` and :func:`~matplotlib.pyplot.pcolormesh`.
When called, these functions default to creating plots on the current axis.
After plotting, functions like :func:`~matplotlib.pyplot.xlabel` and
:func:`~matplotlib.pyplot.title` may be used to further modify the axis.
Alternatively, if using maplotlib's object-oriented API, pass the `ax` parameter
to these functions.

.. note:: This module only works with ``ribs[visualize]`` installed. As such, it
    is not imported with ``import ribs``, and it must be explicitly imported
    with ``import ribs.visualize``.

.. autosummary::
    :toctree:

    ribs.visualize.grid_archive_heatmap
    ribs.visualize.cvt_archive_heatmap
    ribs.visualize.sliding_boundaries_archive_heatmap
    ribs.visualize.parallel_axes_plot
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import axes
from matplotlib.cm import ScalarMappable
from scipy.spatial import Voronoi  # pylint: disable=no-name-in-module

# Matplotlib functions tend to have a ton of args.
# pylint: disable = too-many-arguments

__all__ = [
    "grid_archive_heatmap",
    "cvt_archive_heatmap",
    "sliding_boundaries_archive_heatmap",
    "parallel_axes_plot",
]


def _retrieve_cmap(cmap):
    """Retrieves colormap from matplotlib."""
    if isinstance(cmap, str):
        return matplotlib.cm.get_cmap(cmap)
    if isinstance(cmap, list):
        return matplotlib.colors.ListedColormap(cmap)
    return cmap


def _validate_heatmap_visual_args(aspect, cbar, measure_dim, valid_dims,
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
    if not (cbar == "auto" or isinstance(cbar, axes.Axes) or cbar is None):
        raise ValueError(f"Invalid arg cbar={cbar}; must be 'auto', None, "
                         "or matplotlib.axes.Axes")


def _set_cbar(t, ax, cbar, cbar_kwargs):
    """Sets cbar on the Axes given cbar arg."""
    cbar_kwargs = {} if cbar_kwargs is None else cbar_kwargs
    if cbar == "auto":
        ax.figure.colorbar(t, ax=ax, **cbar_kwargs)
    elif isinstance(cbar, axes.Axes):
        cbar.figure.colorbar(t, ax=cbar, **cbar_kwargs)


def grid_archive_heatmap(archive,
                         ax=None,
                         *,
                         transpose_measures=False,
                         cmap="magma",
                         aspect=None,
                         vmin=None,
                         vmax=None,
                         cbar="auto",
                         pcm_kwargs=None,
                         cbar_kwargs=None):
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

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from ribs.archives import GridArchive
            >>> from ribs.visualize import grid_archive_heatmap
            >>> # Populate the archive with the negative sphere function.
            >>> archive = GridArchive(solution_dim=2,
            ...                       dims=[20, 20],
            ...                       ranges=[(-1, 1), (-1, 1)])
            >>> x = y = np.linspace(-1, 1, 100)
            >>> xxs, yys = np.meshgrid(x, y)
            >>> xxs, yys = xxs.flatten(), yys.flatten()
            >>> archive.add(solution_batch=np.stack((xxs, yys), axis=1),
            ...             objective_batch=-(xxs**2 + yys**2),
            ...             measures_batch=np.stack((xxs, yys), axis=1))
            >>> # Plot a heatmap of the archive.
            >>> plt.figure(figsize=(8, 6))
            >>> grid_archive_heatmap(archive)
            >>> plt.title("Negative sphere function")
            >>> plt.xlabel("x coords")
            >>> plt.ylabel("y coords")
            >>> plt.show()

    Args:
        archive (GridArchive): A 2D :class:`~ribs.archives.GridArchive`.
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
            height/width). Defaults to ``'auto'`` for 2D and ``0.5`` for 1D.
            ``'equal'`` is the same as ``aspect=1``.
        vmin (float): Minimum objective value to use in the plot. If ``None``,
            the minimum objective value in the archive is used.
        vmax (float): Maximum objective value to use in the plot. If ``None``,
            the maximum objective value in the archive is used.
        cbar ('auto', None, matplotlib.axes.Axes): By default, this is set to
            ``'auto'`` which displays the colorbar on the archive's current
            :class:`~matplotlib.axes.Axes`. If ``None``, then colorbar is not
            displayed. If this is an :class:`~matplotlib.axes.Axes`, displays
            the colorbar on the specified Axes.
        pcm_kwargs (dict): Additional kwargs to pass to
            :func:`~matplotlib.pyplot.pcolormesh`.
        cbar_kwargs (dict): Additional kwargs to pass to
            :func:`~matplotlib.pyplot.colorbar`.

    Raises:
        ValueError: The archive's dimension must be 1D or 2D.
    """
    _validate_heatmap_visual_args(
        aspect, cbar, archive.measure_dim, [1, 2],
        "Heatmaps can only be plotted for 1D or 2D GridArchive")

    if aspect is None:
        # Handles default aspects for different dims.
        if archive.measure_dim == 1:
            aspect = 0.5
        else:
            aspect = "auto"

    # Try getting the colormap early in case it fails.
    cmap = _retrieve_cmap(cmap)

    # Useful to have these data available.
    df = archive.as_pandas()
    objective_batch = df.objective_batch()

    if archive.measure_dim == 1:
        # Retrieve data from archive. There should be only 2 bounds; upper and
        # lower, since it is 1D.
        lower_bounds = archive.lower_bounds
        upper_bounds = archive.upper_bounds
        x_dim = archive.dims[0]
        x_bounds = archive.boundaries[0]
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
                          **pcm_kwargs)
    elif archive.measure_dim == 2:
        # Retrieve data from archive.
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
                          **pcm_kwargs)

    # Create color bar.
    _set_cbar(t, ax, cbar, cbar_kwargs)


def cvt_archive_heatmap(archive,
                        ax=None,
                        *,
                        plot_centroids=True,
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
        ValueError: The archive is not 2D.
    """
    _validate_heatmap_visual_args(
        aspect, cbar, archive.measure_dim, [2],
        "Heatmaps can only be plotted for 2D CVTArchive")

    if aspect is None:
        aspect = "auto"

    # Try getting the colormap early in case it fails.
    cmap = _retrieve_cmap(cmap)

    # Retrieve data from archive.
    lower_bounds = archive.lower_bounds
    upper_bounds = archive.upper_bounds
    centroids = archive.centroids
    samples = archive.samples
    if transpose_measures:
        lower_bounds = np.flip(lower_bounds)
        upper_bounds = np.flip(upper_bounds)
        centroids = np.flip(centroids, axis=1)
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

    # Plot the sample points and centroids.
    if plot_samples:
        ax.plot(samples[:, 0], samples[:, 1], "o", c="gray", ms=ms)
    if plot_centroids:
        ax.plot(centroids[:, 0], centroids[:, 1], "ko", ms=ms)

    # Create color bar.
    _set_cbar(mappable, ax, cbar, cbar_kwargs)


def sliding_boundaries_archive_heatmap(archive,
                                       ax=None,
                                       *,
                                       transpose_measures=False,
                                       cmap="magma",
                                       aspect="auto",
                                       ms=None,
                                       boundary_lw=0,
                                       vmin=None,
                                       vmax=None,
                                       cbar="auto",
                                       cbar_kwargs=None):
    """Plots heatmap of a :class:`~ribs.archives.SlidingBoundariesArchive` with
    2D measure space.

    Since the boundaries of :class:`ribs.archives.SlidingBoundariesArchive` are
    dynamic, we plot the heatmap as a scatter plot, in which each marker is an
    elite and its color represents the objective value. Boundaries can
    optionally be drawn by setting ``boundary_lw`` to a positive value.

    Examples:
        .. plot::
            :context: close-figs

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from ribs.archives import SlidingBoundariesArchive
            >>> from ribs.visualize import sliding_boundaries_archive_heatmap
            >>> archive = SlidingBoundariesArchive(solution_dim=2,
            ...                                    dims=[10, 20],
            ...                                    ranges=[(-1, 1), (-1, 1)],
            ...                                    seed=42)
            >>> # Populate the archive with the negative sphere function.
            >>> rng = np.random.default_rng(seed=10)
            >>> coords = np.clip(rng.standard_normal((1000, 2)), -1.5, 1.5)
            >>> archive.add(solution_batch=coords,
            ...             objective_batch=-np.sum(coords**2, axis=1),
            ...             measures_batch=coords)
            >>> # Plot heatmaps of the archive.
            >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
            >>> fig.suptitle("Negative sphere function")
            >>> sliding_boundaries_archive_heatmap(archive, ax=ax1,
            ...                                    boundary_lw=0.5)
            >>> sliding_boundaries_archive_heatmap(archive, ax=ax2)
            >>> ax1.set_title("With boundaries")
            >>> ax2.set_title("Without boundaries")
            >>> ax1.set(xlabel='x coords', ylabel='y coords')
            >>> ax2.set(xlabel='x coords', ylabel='y coords')
            >>> plt.show()

    Args:
        archive (SlidingBoundariesArchive): A 2D
            :class:`~ribs.archives.SlidingBoundariesArchive`.
        ax (matplotlib.axes.Axes): Axes on which to plot the heatmap.
            If ``None``, the current axis will be used.
        transpose_measures (bool): By default, the first measure in the archive
            will appear along the x-axis, and the second will be along the
            y-axis. To switch this behavior (i.e. to transpose the axes), set
            this to ``True``.
        cmap (str, list, matplotlib.colors.Colormap): Colormap to use when
            plotting intensity. Either the name of a
            :class:`~matplotlib.colors.Colormap`, a list of RGB or RGBA colors
            (i.e. an :math:`N \\times 3` or :math:`N \\times 4` array), or a
            :class:`~matplotlib.colors.Colormap` object.
        aspect ('auto', 'equal', float): The aspect ratio of the heatmap (i.e.
            height/width). Defaults to ``'auto'``. ``'equal'`` is the same as
            ``aspect=1``.
        ms (float): Marker size for the solutions.
        boundary_lw (float): Line width when plotting the boundaries.
            Set to ``0`` to have no boundaries.
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
        ValueError: The archive is not 2D.
    """
    _validate_heatmap_visual_args(
        aspect, cbar, archive.measure_dim, [2],
        "Heatmaps can only be plotted for 2D SlidingBoundariesArchive")

    if aspect is None:
        aspect = "auto"

    # Try getting the colormap early in case it fails.
    cmap = _retrieve_cmap(cmap)

    # Retrieve data from archive.
    df = archive.as_pandas()
    measures_batch = df.measures_batch()
    x = measures_batch[:, 0]
    y = measures_batch[:, 1]
    x_boundary = archive.boundaries[0]
    y_boundary = archive.boundaries[1]
    lower_bounds = archive.lower_bounds
    upper_bounds = archive.upper_bounds

    if transpose_measures:
        # Since the archive is 2D, transpose by swapping the x and y measures
        # and boundaries and by flipping the bounds (the bounds are arrays of
        # length 2).
        x, y = y, x
        x_boundary, y_boundary = y_boundary, x_boundary
        lower_bounds = np.flip(lower_bounds)
        upper_bounds = np.flip(upper_bounds)

    # Initialize the axis.
    ax = plt.gca() if ax is None else ax
    ax.set_xlim(lower_bounds[0], upper_bounds[0])
    ax.set_ylim(lower_bounds[1], upper_bounds[1])
    ax.set_aspect(aspect)

    # Create the plot.
    objective_batch = df.objective_batch()
    vmin = np.min(objective_batch) if vmin is None else vmin
    vmax = np.max(objective_batch) if vmax is None else vmax
    t = ax.scatter(x,
                   y,
                   s=ms,
                   c=objective_batch,
                   cmap=cmap,
                   vmin=vmin,
                   vmax=vmax)
    if boundary_lw > 0.0:
        # Careful with bounds here. Lines drawn along the x axis should extend
        # between the y bounds and vice versa -- see
        # https://github.com/icaros-usc/pyribs/issues/270
        ax.vlines(x_boundary,
                  lower_bounds[1],
                  upper_bounds[1],
                  color='k',
                  linewidth=boundary_lw)
        ax.hlines(y_boundary,
                  lower_bounds[0],
                  upper_bounds[0],
                  color='k',
                  linewidth=boundary_lw)

    # Create color bar.
    _set_cbar(t, ax, cbar, cbar_kwargs)


def parallel_axes_plot(archive,
                       ax=None,
                       *,
                       measure_order=None,
                       cmap="magma",
                       linewidth=1.5,
                       alpha=0.8,
                       vmin=None,
                       vmax=None,
                       sort_archive=False,
                       cbar_orientation='horizontal',
                       cbar_pad=0.1):
    """Visualizes archive elites in measure space with a parallel axes plot.

    This visualization is meant to show the coverage of the measure space at a
    glance. Each axis represents one measure dimension, and each line in the
    diagram represents one elite in the archive. Three main things are evident
    from this plot:

    - **measure space coverage,** as determined by the amount of the axis that
      has lines passing through it. If the lines are passing through all parts
      of the axis, then there is likely good coverage for that measure.

    - **Correlation between neighboring measures.** In the below example, we see
      perfect correlation between ``measures_0`` and ``measures_1``, since none
      of the lines cross each other. We also see the perfect negative
      correlation between ``measures_3`` and ``measures_4``, indicated by the
      crossing of all lines at a single point.

    - **Whether certain values of the measure dimensions affect the objective
      value strongly.** In the below example, we see ``measures_2`` has many
      elites with high objective near zero. This is more visible when
      ``sort_archive`` is passed in, as elites with higher objective values
      will be plotted on top of individuals with lower objective values.

    Examples:
        .. plot::
            :context: close-figs

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from ribs.archives import GridArchive
            >>> from ribs.visualize import parallel_axes_plot
            >>> # Populate the archive with the negative sphere function.
            >>> archive = GridArchive(
            ...               solution_dim=3, dims=[20, 20, 20, 20, 20],
            ...               ranges=[(-1, 1), (-1, 1), (-1, 1),
            ...                       (-1, 1), (-1, 1)],
            ...           )
            >>> for x in np.linspace(-1, 1, 100):
            ...     for y in np.linspace(0, 1, 100):
            ...         for z in np.linspace(-1, 1, 100):
            ...             archive.add_single(
            ...                 solution=np.array([x,y,z]),
            ...                 objective=-(x**2 + y**2 + z**2),
            ...                 measures=np.array([0.5*x,x,y,z,-0.5*z]),
            ...             )
            >>> # Plot a heatmap of the archive.
            >>> plt.figure(figsize=(8, 6))
            >>> parallel_axes_plot(archive)
            >>> plt.title("Negative sphere function")
            >>> plt.ylabel("axis values")
            >>> plt.show()

    Args:
        archive (ArchiveBase): Any ribs archive.
        ax (matplotlib.axes.Axes): Axes on which to create the plot.
            If ``None``, the current axis will be used.
        measure_order (list of int or list of (int, str)): If this is a list
            of ints, it specifies the axes order for measures (e.g. ``[2, 0,
            1]``). If this is a list of tuples, each tuple takes the form
            ``(int, str)`` where the int specifies the measure index and the str
            specifies a name for the measure (e.g. ``[(1, "y-value"), (2,
            "z-value"), (0, "x-value")]``). The order specified does not need
            to have the same number of elements as the number of measures in
            the archive, e.g. ``[1, 3]`` or ``[1, 2, 3, 2]``.
        cmap (str, list, matplotlib.colors.Colormap): Colormap to use when
            plotting intensity. Either the name of a
            :class:`~matplotlib.colors.Colormap`, a list of RGB or RGBA colors
            (i.e. an :math:`N \\times 3` or :math:`N \\times 4` array), or a
            :class:`~matplotlib.colors.Colormap` object.
        linewidth (float): Line width for each elite in the plot.
        alpha (float): Opacity of the line for each elite (passing a low value
            here may be helpful if there are many archive elites, as more
            elites would be visible).
        vmin (float): Minimum objective value to use in the plot. If ``None``,
            the minimum objective value in the archive is used.
        vmax (float): Maximum objective value to use in the plot. If ``None``,
            the maximum objective value in the archive is used.
        sort_archive (boolean): If ``True``, sorts the archive so that the
            highest performing elites are plotted on top of lower performing
            elites.

            .. warning:: This may be slow for large archives.
        cbar_orientation (str): The orientation of the colorbar. Use either
            ``'vertical'`` or ``'horizontal'``
        cbar_pad (float): The amount of padding to use for the colorbar.

    Raises:
        ValueError: ``cbar_orientation`` has an invalid value.
        ValueError: The measures provided do not exist in the archive.
        TypeError: ``measure_order`` is not a list of all ints or all tuples.
    """
    # Try getting the colormap early in case it fails.
    cmap = _retrieve_cmap(cmap)

    # Check that the orientation input is correct.
    if cbar_orientation not in ['vertical', 'horizontal']:
        raise ValueError("cbar_orientation must be 'vertical' or 'horizontal' "
                         f"but is '{cbar_orientation}'")

    # If there is no order specified, plot in increasing numerical order.
    if measure_order is None:
        cols = np.arange(archive.measure_dim)
        axis_labels = [f"measure_{i}" for i in range(archive.measure_dim)]
        lower_bounds = archive.lower_bounds
        upper_bounds = archive.upper_bounds

    # Use the requested measures (may be less than the original number of
    # measures).
    else:
        # Check for errors in specification.
        if all(isinstance(measure, int) for measure in measure_order):
            cols = np.array(measure_order)
            axis_labels = [f"measure_{i}" for i in cols]
        elif all(
                len(measure) == 2 and isinstance(measure[0], int) and
                isinstance(measure[1], str) for measure in measure_order):
            cols, axis_labels = zip(*measure_order)
            cols = np.array(cols)
        else:
            raise TypeError("measure_order must be a list of ints or a list of"
                            "tuples in the form (int, str)")

        if np.max(cols) >= archive.measure_dim:
            raise ValueError(f"Invalid Measures: requested measures index "
                             f"{np.max(cols)}, but archive only has "
                             f"{archive.measure_dim} measures.")
        if any(measure < 0 for measure in cols):
            raise ValueError("Invalid Measures: requested a negative measure"
                             " index.")

        # Find the indices of the requested order.
        lower_bounds = archive.lower_bounds[cols]
        upper_bounds = archive.upper_bounds[cols]

    host_ax = plt.gca() if ax is None else ax  # Try to get current axis.
    df = archive.as_pandas(include_solutions=False)
    vmin = df["objective"].min() if vmin is None else vmin
    vmax = df["objective"].max() if vmax is None else vmax
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    if sort_archive:
        df.sort_values("objective", inplace=True)
    objectives = df.objective_batch()
    ys = df.measures_batch()[:, cols]
    y_ranges = upper_bounds - lower_bounds

    # Transform all data to be in the first axis coordinates.
    normalized_ys = np.zeros_like(ys)
    normalized_ys[:, 0] = ys[:, 0]
    normalized_ys[:, 1:] = (
        (ys[:, 1:] - lower_bounds[1:]) / y_ranges[1:] * y_ranges[0] +
        lower_bounds[0])

    # Copy the axis for the other measures.
    axs = [host_ax] + [host_ax.twinx() for i in range(len(cols) - 1)]
    for i, axis in enumerate(axs):
        axis.set_ylim(lower_bounds[i], upper_bounds[i])
        axis.spines['top'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        if axis != host_ax:
            axis.spines['left'].set_visible(False)
            axis.yaxis.set_ticks_position('right')
            axis.spines["right"].set_position(("axes", i / (len(cols) - 1)))

    host_ax.set_xlim(0, len(cols) - 1)
    host_ax.set_xticks(range(len(cols)))
    host_ax.set_xticklabels(axis_labels)
    host_ax.tick_params(axis='x', which='major', pad=7)
    host_ax.spines['right'].set_visible(False)
    host_ax.xaxis.tick_top()

    for elite_ys, objective in zip(normalized_ys, objectives):
        # Draw straight lines between the axes in the appropriate color.
        color = cmap(norm(objective))
        host_ax.plot(range(len(cols)),
                     elite_ys,
                     c=color,
                     alpha=alpha,
                     linewidth=linewidth)

    # Create a colorbar.
    mappable = ScalarMappable(cmap=cmap)
    mappable.set_clim(vmin, vmax)
    host_ax.figure.colorbar(mappable,
                            pad=cbar_pad,
                            orientation=cbar_orientation)
