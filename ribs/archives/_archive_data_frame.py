"""Provides ArchiveDataFrame."""
import itertools

import pandas as pd

from ribs.archives._elite import Elite

# Developer Note: The documentation for this class is hacked -- to add new
# methods, manually modify the template in docs/_templates/autosummary/class.rst


class ArchiveDataFrame(pd.DataFrame):
    """A modified :class:`~pandas.DataFrame` for archive data.

    As this class inherits from :class:`~pandas.DataFrame`, it has the same
    methods, attributes, and arguments (even though the arguments are shown here
    as ``*args`` and ``**kwargs``).  However, this class adds methods that make
    it convenient to work with elites. This documentation only lists these
    additional methods and attributes.

    Example:

        This object is created by :meth:`~ArchiveBase.as_pandas` (i.e. users
        should not create it on their own)::

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

        If you save an ArchiveDataFrame to a CSV, loading it with
        :func:`pandas.read_csv` will load a :class:`~pandas.DataFrame`. To
        load a CSV as an ArchiveDataFrame, simply use::

            df = ArchiveDataFrame(pd.read_csv("file.csv"))
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Extract slices for creating the attrs. Indices, objectives, and
        # behaviors are always included, while solutions and metadata may be
        # excluded.
        last_index = None
        last_behavior = None
        last_solution = None
        self._has_metadata = False
        for col in self.columns:
            if col.startswith("index_"):
                last_index = col
            elif col.startswith("behavior_"):
                last_behavior = col
            elif col.startswith("solution_"):
                last_solution = col
            elif col == "metadata":
                self._has_metadata = True

        self._index_slice = slice("index_0", last_index)
        self._behavior_slice = slice("behavior_0", last_behavior)
        self._solution_slice = (slice("solution_0", last_solution)
                                if last_solution is not None else None)

    def iterelites(self):
        """Iterator which outputs every :class:`Elite` in the DataFrame."""
        batch_solutions = (itertools.repeat(None)
                           if self._solution_slice is None else
                           self.batch_solutions())
        batch_metadata = (itertools.repeat(None)
                          if not self._has_metadata else self.batch_metadata())
        return map(
            lambda e: Elite(e[0], e[1], e[2], e[3], e[4]),
            zip(
                batch_solutions,
                self.batch_objectives(),
                self.batch_behaviors(),
                self.batch_indices(),
                batch_metadata,
            ),
        )

    def batch_behaviors(self):
        """Array with behavior values of all elites.

        .. note::
            All the ``batch`` methods "align" with each other -- i.e.
            ``batch_behaviors()[i]`` corresponds to ``batch_indices()[i]``,
            ``batch_metadata()[i]``, ``batch_objectives()[i]``, and
            ``batch_solutions()[i]``.

        Returns:
            (n, behavior_dim) numpy.ndarray: See above.
        """
        return self.loc[:, self._behavior_slice].to_numpy(copy=True)

    def batch_indices(self):
        """List of archive indices of all elites.

        This is a list because each index is a tuple, and numpy arrays are not
        designed to store tuple objects.

        Returns:
            (n,) list: See above.
        """
        return [
            tuple(idx[1:])
            for idx in self.loc[:, self._index_slice].itertuples()
        ]

    def batch_metadata(self):
        """Array with metadata of all elites.

        None if metadata was excluded (i.e. if ``include_metadata=False`` in
        :meth:`~ArchiveBase.as_pandas`).

        Returns:
            (n,) numpy.ndarray: See above.
        """
        return self["metadata"].to_numpy(
            copy=True) if self._has_metadata else None

    def batch_objectives(self):
        """Array with objective values of all elites.

        Returns:
            (n,) numpy.ndarray: See above.
        """
        return self["objective"].to_numpy(copy=True)

    def batch_solutions(self):
        """Array with solutions of all elites.

        None if solutions were excluded (i.e. if ``include_solutions=False``
        in :meth:`~ArchiveBase.as_pandas`).

        Returns:
            (n, solution_dim) numpy.ndarray: See above.
        """
        return (None if self._solution_slice is None else
                self.loc[:, self._solution_slice].to_numpy(copy=True))
