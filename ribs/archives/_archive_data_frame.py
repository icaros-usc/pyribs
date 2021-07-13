"""Provides ArchiveDataFrame."""
import itertools

import pandas as pd

from ribs.archives._elite import Elite

# Developer Note: The documentation for this class is hacked -- to add new
# methods, manually modify the template in docs/_templates/autosummary/class.rst


class ArchiveDataFrame(pd.DataFrame):
    """A modified :class:`~pandas.DataFrame` for archive data.

    As this class inherits from :class:`~pandas.DataFrame`, it has all of the
    same methods and attributes, but it adds several more that make it
    convenient to work with elites. This documentation only lists the additional
    methods and attributes. Note that the ``__init__`` takes in the exact same
    arguments as :class:`~pandas.DataFrame`.

    Example:

        This object is created by :meth:`~ArchiveBase.as_pandas` (i.e. users
        should not create it on their own)::

            df = archive.as_pandas()

        To iterate through every :class:`Elite`, use::

            for elite in df.iterelites():
                elite.sol
                elite.obj
                ...

        There are also attributes to access the solutions, objectives, etc. of
        all elites in the archive. For instance, the following is an array
        where entry ``i`` contains the behavior values of the ``i``'th elite in
        the DataFrame::

            df.batch_behaviors

        Note that all the ``batch`` attributes "align" with each other -- i.e.
        ``batch_solutions[i]`` corresponds to ``batch_behaviors[i]``,
        ``batch_indices[i]``, ``batch_metadata[i]``, and
        ``batch_objectives[i]``.

    .. warning::

        Accessing ``batch`` attributes (e.g. ``batch_behaviors``) always creates
        a copy, so the following will copy the behaviors 3 times::

            df.batch_behaviors[0]
            df.batch_behaviors.mean()
            df.batch_behaviors.median()

        **Thus, if you need to use the attribute several times, we recommend
        storing it first, like so**::

            behaviors = df.batch_behaviors
            behaviors[0]
            behaviors.mean()
            behaviors.median()
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
                           self.batch_solutions)
        batch_metadata = (itertools.repeat(None)
                          if not self._has_metadata else self.batch_metadata)
        return map(
            lambda e: Elite(e[0], e[1], e[2], e[3], e[4]),
            zip(
                batch_solutions,
                self.batch_objectives,
                self.batch_behaviors,
                self.batch_indices,
                batch_metadata,
            ),
        )

    @property
    def batch_behaviors(self):
        """(n, behavior_dim) numpy.ndarray: Array with behavior values of all
        elites."""
        return self.loc[:, self._behavior_slice].to_numpy(copy=True)

    @property
    def batch_indices(self):
        """(n,) list: List of archive indices of all elites.

        This is a list because each index is a tuple, and numpy arrays are not
        designed to store tuple objects.
        """
        return [
            tuple(idx) for idx in self.loc[:, self._index_slice].itertuples()
        ]

    @property
    def batch_metadata(self):
        """(n,) numpy.ndarray: Array with metadata of all elites.

        None if metadata was excluded (i.e.
        ``include_metadata = False`` in :meth:`~ArchiveBase.as_pandas`).
        """
        return self["metadata"].to_numpy(
            copy=True) if self._has_metadata else None

    @property
    def batch_objectives(self):
        """(n,) numpy.ndarray: Array with objective values of all elites."""
        return self["objective"].to_numpy(copy=True)

    @property
    def batch_solutions(self):
        """(n, solution_dim) numpy.ndarray: Array with solutions of all elites.

        None if solutions were excluded (i.e.
        ``include_solutions = False`` in :meth:`~ArchiveBase.as_pandas`).
        """
        return (None if self._solution_slice is None else
                self.loc[:, self._solution_slice].to_numpy(copy=True))
