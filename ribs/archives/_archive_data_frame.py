"""Provides ArchiveDataFrame."""
import numpy as np
import pandas as pd

from ribs.archives._elite import Elite

# Developer Notes:
# - The documentation for this class is hacked -- to add new methods, manually
#   modify the template in docs/_templates/autosummary/class.rst
# - See here for info on extending DataFrame:
#   https://pandas.pydata.org/pandas-docs/stable/development/extending.html


class ArchiveDataFrame(pd.DataFrame):
    """A modified :class:`~pandas.DataFrame` for archive data.

    As this class inherits from :class:`~pandas.DataFrame`, it has the same
    methods, attributes, and arguments (even though the arguments shown here are
    ``*args`` and ``**kwargs``). However, this class adds methods that make it
    convenient to work with elites. This documentation only lists these
    additional methods and attributes.

    Example:

        This object is created by :meth:`~ArchiveBase.as_pandas` (i.e. users
        typically do not create it on their own)::

            df = archive.as_pandas()

        To iterate through every :class:`Elite`, use::

            for elite in df.iterelites():
                elite.solution
                elite.objective
                ...

        There are also methods to access the solutions, objectives, etc. of
        all elites in the archive. For instance, the following is an array
        where entry ``i`` contains the measures of the ``i``'th elite in the
        DataFrame::

            df.measures_batch()

    .. warning::

        Accessing ``batch`` methods (e.g. :meth:`measures_batch`) always
        creates a copy, so the following will copy the measures 3 times::

            df.measures_batch()[0]
            df.measures_batch().mean()
            df.measures_batch().median()

        **Thus, if you need to use the method several times, we recommend
        storing it first, like so**::

            measures_batch = df.measures_batch()
            measures_batch[0]
            measures_batch.mean()
            measures_batch.median()

    .. note::

        After saving an ArchiveDataFrame to a CSV, loading it with
        :func:`pandas.read_csv` will load a :class:`~pandas.DataFrame`. To load
        a CSV as an ArchiveDataFrame, pass the ``DataFrame`` from ``read_csv``
        to ArchiveDataFrame::

            df = ArchiveDataFrame(pd.read_csv("file.csv"))

    .. note::

        All the ``batch`` methods "align" with each other -- i.e.
        ``measures_batch()[i]`` corresponds to ``index_batch()[i]``,
        ``metadata_batch()[i]``, ``objective_batch()[i]``, and
        ``solution_batch()[i]``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return ArchiveDataFrame

    def iterelites(self):
        """Iterator which outputs every :class:`Elite` in the ArchiveDataFrame.

        Data which is unavailable will be turned into None. For example, if
        there are no solution columns, then ``elite.solution`` will be None.
        """
        solution_batch = self.solution_batch()
        objective_batch = self.objective_batch()
        measures_batch = self.measures_batch()
        index_batch = self.index_batch()
        metadata_batch = self.metadata_batch()

        none_array = np.empty(len(self), dtype=object)

        return map(
            lambda e: Elite(e[0], e[1], e[2], e[3], e[4]),
            zip(
                none_array if solution_batch is None else solution_batch,
                none_array if objective_batch is None else objective_batch,
                none_array if measures_batch is None else measures_batch,
                none_array if index_batch is None else index_batch,
                none_array if metadata_batch is None else metadata_batch,
            ),
        )

    # Note: The slices for batch methods cannot be pre-computed because the
    # DataFrame columns might change in-place, e.g. when a column is deleted.

    def solution_batch(self):
        """Array with solutions of all elites.

        None if there are no solutions (e.g. if ``include_solutions=False`` in
        :meth:`~ArchiveBase.as_pandas`).

        Returns:
            (n, solution_dim) numpy.ndarray: See above.
        """
        cols = [c for c in self if c.startswith("solution_")]
        return self[cols].to_numpy(copy=True) if cols else None

    def objective_batch(self):
        """Array with objective values of all elites.

        None if there are no objectives in the ``ArchiveDataFrame``.

        Returns:
            (n,) numpy.ndarray: See above.
        """
        return self["objective"].to_numpy(
            copy=True) if "objective" in self else None

    def measures_batch(self):
        """Array with measures of all elites.

        None if there are no measures in the ``ArchiveDataFrame``.

        Returns:
            (n, measure_dim) numpy.ndarray: See above.
        """
        cols = [c for c in self if c.startswith("measure_")]
        return self[cols].to_numpy(copy=True) if cols else None

    def index_batch(self):
        """Array with indices of all elites.

        None if there are no indices in the ``ArchiveDataFrame``.

        Returns:
            (n,) numpy.ndarray: See above.
        """
        return self["index"].to_numpy(copy=True) if "index" in self else None

    def metadata_batch(self):
        """Array with metadata of all elites.

        None if there is no metadata (e.g. if ``include_metadata=False`` in
        :meth:`~ArchiveBase.as_pandas`).

        Returns:
            (n,) numpy.ndarray: See above.
        """
        return self["metadata"].to_numpy(
            copy=True) if "metadata" in self else None
