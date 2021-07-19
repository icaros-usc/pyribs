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
                elite.sol
                elite.obj
                ...

        There are also methods to access the solutions, objectives, etc. of
        all elites in the archive. For instance, the following is an array
        where entry ``i`` contains the behavior values of the ``i``'th elite in
        the DataFrame::

            df.batch_behaviors()

    .. warning::

        Accessing ``batch`` methods (e.g. :meth:`batch_behaviors`) always
        creates a copy, so the following will copy the behaviors 3 times::

            df.batch_behaviors()[0]
            df.batch_behaviors().mean()
            df.batch_behaviors().median()

        **Thus, if you need to use the method several times, we recommend
        storing it first, like so**::

            behaviors = df.batch_behaviors()
            behaviors[0]
            behaviors.mean()
            behaviors.median()

    .. note::

        After saving an ArchiveDataFrame to a CSV, loading it with
        :func:`pandas.read_csv` will load a :class:`~pandas.DataFrame`. To load
        a CSV as an ArchiveDataFrame, pass the ``DataFrame`` from ``read_csv``
        to ArchiveDataFrame::

            df = ArchiveDataFrame(pd.read_csv("file.csv"))

    .. note::

        All the ``batch`` methods "align" with each other -- i.e.
        ``batch_behaviors()[i]`` corresponds to ``batch_indices()[i]``,
        ``batch_metadata()[i]``, ``batch_objectives()[i]``, and
        ``batch_solutions()[i]``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return ArchiveDataFrame

    def iterelites(self):
        """Iterator which outputs every :class:`Elite` in the ArchiveDataFrame.

        Data which is unavailable will be turned into None. For example, if
        there are no solution columns, then ``elite.sol`` will be None.
        """
        batch_solutions = self.batch_solutions()
        batch_objectives = self.batch_objectives()
        batch_behaviors = self.batch_behaviors()
        batch_indices = self.batch_indices()
        batch_metadata = self.batch_metadata()

        none_array = np.empty(len(self), dtype=object)

        return map(
            lambda e: Elite(e[0], e[1], e[2], e[3], e[4]),
            zip(
                none_array if batch_solutions is None else batch_solutions,
                none_array if batch_objectives is None else batch_objectives,
                none_array if batch_behaviors is None else batch_behaviors,
                none_array if batch_indices is None else batch_indices,
                none_array if batch_metadata is None else batch_metadata,
            ),
        )

    # Note: The slices for batch methods cannot be pre-computed because the
    # DataFrame columns might change in-place, e.g. when a column is deleted.

    def batch_behaviors(self):
        """Array with behavior values of all elites.

        None if there are no behavior values in the ``ArchiveDataFrame``.

        Returns:
            (n, behavior_dim) numpy.ndarray: See above.
        """
        cols = [c for c in self if c.startswith("behavior_")]
        return self[cols].to_numpy(copy=True) if cols else None

    def batch_indices(self):
        """List of archive indices of all elites.

        This is a list because each index is a tuple, and numpy arrays are not
        designed to store tuple objects.

        None if there are no indices in the ``ArchiveDataFrame``.

        Returns:
            (n,) list: See above.
        """
        cols = [c for c in self if c.startswith("index_")]
        return ([tuple(idx[1:]) for idx in self[cols].itertuples()]
                if cols else None)

    def batch_metadata(self):
        """Array with metadata of all elites.

        None if there is no metadata (e.g. if ``include_metadata=False`` in
        :meth:`~ArchiveBase.as_pandas`).

        Returns:
            (n,) numpy.ndarray: See above.
        """
        return self["metadata"].to_numpy(
            copy=True) if "metadata" in self else None

    def batch_objectives(self):
        """Array with objective values of all elites.

        None if there are no objectives in the ``ArchiveDataFrame``.

        Returns:
            (n,) numpy.ndarray: See above.
        """
        return self["objective"].to_numpy(
            copy=True) if "objective" in self else None

    def batch_solutions(self):
        """Array with solutions of all elites.

        None if there are no solutions (e.g. if ``include_solutions=False`` in
        :meth:`~ArchiveBase.as_pandas`).

        Returns:
            (n, solution_dim) numpy.ndarray: See above.
        """
        cols = [c for c in self if c.startswith("solution_")]
        return self[cols].to_numpy(copy=True) if cols else None
