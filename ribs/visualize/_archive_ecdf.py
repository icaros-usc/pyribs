"""Provides archive_ecdf."""

from __future__ import annotations

from matplotlib.axes import Axes
from pandas import DataFrame
from seaborn import ecdfplot

from ribs.archives import ArchiveBase, ArchiveDataFrame
from ribs.visualize._utils import validate_df


def archive_ecdf(
    archive: ArchiveBase,
    df: DataFrame | ArchiveDataFrame | None = None,
    **kwargs,
) -> Axes:
    r"""Plots an ECDF or ECCDF of the objective values in an archive.

    Generally, the `ECDF (empirical cumulative distribution function)
    <https://en.wikipedia.org/wiki/Empirical_distribution_function>`_ counts the number
    of observations that fall below each output value. In the case of archives, an ECDF
    counts the number of elites that perform worse than or equal to each objective
    value. Conversely, an ECCDF (empirical *complementary* cumulative distribution
    function) counts the number of elites that perform better than or equal to each
    objective value.

    This function is a thin wrapper around Seaborn's :func:`~seaborn.ecdfplot` that
    extracts the objective values from the archive and then passes them as data. All
    kwargs are also passed directly to ``ecdfplot``. Similar to other pyribs
    visualization functions, one can also pass in the archive and a ``df`` extracted
    from the archive's :meth:`~ribs.archives.ArchiveBase.data` method.

    .. info::

        The idea of using a CCDF to evaluate QD algorithms was introduced and formalized
        in `Vassiliades
        2018 <https://arxiv.org/abs/1610.05729>`_.

    Examples:
        .. plot::
            :context: close-figs

            import numpy as np
            import matplotlib.pyplot as plt
            from ribs.archives import GridArchive
            from ribs.visualize import archive_ecdf

            # Populate the archive with the negative sphere function.
            archive = GridArchive(solution_dim=2,
                                  dims=[100, 100],
                                  ranges=[(-1, 1), (-1, 1)])
            x = np.random.uniform(-1, 1, 10000)
            y = np.random.uniform(-1, 1, 10000)
            archive.add(solution=np.stack((x, y), axis=1),
                        objective=-(x**2 + y**2),
                        measures=np.stack((x, y), axis=1))

            # Plot a CCDF of the archive.
            plt.figure(figsize=(8, 6))
            archive_ecdf(archive, complementary=True, stat="count")
            plt.xlabel("Objective")
            plt.ylabel("Num. Elites")
            plt.show()

    Args:
        archive: An archive that can provide its objective values via the
            :meth:`~ribs.archives.ArchiveBase.data` method.
        df: If provided, we will plot data from this argument instead of the data
            currently in the archive. This data can be obtained by, for instance,
            calling :meth:`ribs.archives.ArchiveBase.data` with ``return_type="pandas"``
            and modifying the resulting :class:`~ribs.archives.ArchiveDataFrame`. Note
            that, at a minimum, the data must contain a column for "objective".
        kwargs: Kwargs for :func:`~seaborn.ecdfplot`.

    Returns:
        The Matplotlib axes containing the plot.

    Raises:
        AttributeError: The data() method is not implemented on the given archive, which
            prevents retrieving its objective values.
    """
    # Retrieve archive data.
    if df is None:
        try:
            objectives = archive.data("objective")
        except NotImplementedError as e:
            raise AttributeError(
                "To use archive_ecdf, the archive must have the data() method."
            ) from e
    else:
        df = validate_df(df)
        objectives = df["objective"]

    return ecdfplot(objectives, **kwargs)
