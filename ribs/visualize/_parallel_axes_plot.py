"""Provides parallel_axes_plot."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable

from ribs.visualize._utils import retrieve_cmap, set_cbar, validate_df

# Matplotlib functions tend to have a ton of args.
# pylint: disable = too-many-arguments


def parallel_axes_plot(archive,
                       ax=None,
                       *,
                       df=None,
                       measure_order=None,
                       cmap="magma",
                       linewidth=1.5,
                       alpha=0.8,
                       vmin=None,
                       vmax=None,
                       sort_archive=False,
                       cbar="auto",
                       cbar_kwargs=None):
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
            >>> for x in np.linspace(-1, 1, 10):
            ...     for y in np.linspace(0, 1, 10):
            ...         for z in np.linspace(-1, 1, 10):
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
        df (ribs.archives.ArchiveDataFrame): If provided, we will plot data from
            this argument instead of the data currently in the archive. This
            data can be obtained by, for instance, calling
            :meth:`ribs.archives.ArchiveBase.as_pandas()` and modifying the
            resulting :class:`ArchiveDataFrame`. Note that, at a minimum, the
            data must contain columns for index, objective, and measures. To
            display a custom metric, replace the "objective" column.
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
        sort_archive (bool): If ``True``, sorts the archive so that the highest
            performing elites are plotted on top of lower performing elites.

            .. warning:: This may be slow for large archives.
        cbar ('auto', None, matplotlib.axes.Axes): By default, this is set to
            ``'auto'`` which displays the colorbar on the archive's current
            :class:`~matplotlib.axes.Axes`. If ``None``, then colorbar is not
            displayed. If this is an :class:`~matplotlib.axes.Axes`, displays
            the colorbar on the specified Axes.
        cbar_kwargs (dict): Additional kwargs to pass to
            :func:`~matplotlib.pyplot.colorbar`. By default, we set
            "orientation" to "horizontal" and "pad" to 0.1.

    Raises:
        ValueError: The measures provided do not exist in the archive.
        TypeError: ``measure_order`` is not a list of all ints or all tuples.
    """
    # Try getting the colormap early in case it fails.
    cmap = retrieve_cmap(cmap)

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
    df = archive.as_pandas() if df is None else validate_df(df)
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

    # Add default colorbar settings.
    cbar_kwargs = {} if cbar_kwargs is None else cbar_kwargs.copy()
    cbar_kwargs.setdefault("orientation", "horizontal")
    cbar_kwargs.setdefault("pad", 0.1)

    set_cbar(mappable, host_ax, cbar, cbar_kwargs)
