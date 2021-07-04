"""Provides EliteTable."""
import numpy as np

from ribs.archives._elite import Elite


def convert_idx(idx):
    """Converts indices to a tuple when necessary.

    We need this since EliteTable stores indices as an array. However, the
    indices may just be int's, in which case we cannot convert them to tuple.
    """
    return tuple(idx) if isinstance(idx, np.ndarray) else idx


class EliteTable:
    """Represents an ordered collection of elites from an archive.

    The table is similar to a :class:`pandas.DataFrame` -- it stores columns of
    elite data, e.g. a column of solutions, a column of objective values.

    Example:

        An :class:`EliteTable` typically comes from an archive's
        :meth:`~ArchiveBase.table` method (i.e. users do not construct it)::

            table = archive.table()

        The table provides access to data from all the elites::

            table.solutions
            table.objective_values
            ...

        The length of the table is the number of elites in it::

            len(table)

        The table can be iterated over -- :class:`Elite`'s are returned when
        doing so::

            for elite in table:
                elite.sol
                elite.obj
                ...

        Indexing is also supported, and the table behaves similarly to a 1D
        numpy array. This is particularly useful when selecting elites which
        satisfy a given condition. The following returns a new table with elites
        that have objective value at least 200::

            table[table.objective_values > 200]

        Note that while indexing usually returns a new :class:`EliteTable`, an
        integer index returns the :class:`Elite` at the given index in the
        table::

            table[0]  # Elite

        Finally, we can filter elites by providing a predicate which takes in an
        :class:`Elite` and returns a bool telling whether to keep the elite. The
        following selects elites which have objective value greater than 200 and
        metadata that is not None::

            filtered = table.filter(
                lambda elite: elite.obj > 200 and elite.meta is not None
            )

    Args:
        solutions: See :attr:`solutions`.
        objective_values: See :attr:`objective_values`.
        behavior_values: See :attr:`behavior_values`.
        indices: See :attr:`indices`.
        metadata: See :attr:`metadata`.
    Raises:
        ValueError: The arrays passed in have unequal length.
    """

    def __init__(self, solutions, objective_values, behavior_values, indices,
                 metadata):
        self._solutions = solutions
        self._objective_values = objective_values
        self._behavior_values = behavior_values
        self._indices = indices
        self._metadata = metadata

        if not (len(solutions) == len(objective_values) == len(behavior_values)
                == len(indices) == len(metadata)):
            raise ValueError(
                "All arrays must be same length, but current lengths are: "
                f"solutions ({len(solutions)}), "
                f"objective_values ({len(objective_values)}), "
                f"behavior_values ({len(behavior_values)}), "
                f"indices ({len(indices)}), "
                f"metadata ({len(metadata)})")

    @property
    def solutions(self):
        """((n, solution_dim) numpy.ndarray): Solution parameters for ``n``
        elites."""
        return self._solutions

    @property
    def objective_values(self):
        """((n,) numpy.ndarray): Objective values of the elites."""
        return self._objective_values

    @property
    def behavior_values(self):
        """((n, behavior_dim) numpy.ndarray): Behavior values of the elites."""
        return self._behavior_values

    @property
    def indices(self):
        """((n, behavior_dim) or (n,) numpy.ndarray): Indices of the elites in
        the archive. The (n,) shape is used when the indices are int.

        .. note::

            Each index should actually be an int or tuple of int, but since
            numpy arrays cannot easily store tuples, we use an array here to
            make batch manipulation easy. Nevertheless, iterating through
            this class still yields :class:`Elite` objects which have a
            tuple index.
        """
        return self._indices

    @property
    def metadata(self):
        """((n,) numpy.ndarray): Metadata of the elites.

        This array is an object array.
        """
        return self._metadata

    def __len__(self):
        """The number of elites stored in the table."""
        return len(self._solutions)

    # Note: The EliteTable itself cannot be an iterable object because if there
    # are multiple iterators over it, they will all share state.  Instead, we
    # return a new iterable object, just like Python containers do. See
    # https://stackoverflow.com/questions/46941719/how-can-i-have-multiple-iterators-over-a-single-python-iterable-at-the-same-time
    # for more info.
    def __iter__(self):
        """Creates an iterator over the elites in the table."""
        return map(
            lambda e: Elite(e[0], e[1], e[2], convert_idx(e[3]), e[4]),
            zip(
                self._solutions,
                self._objective_values,
                self._behavior_values,
                self._indices,
                self._metadata,
            ),
        )

    def __getitem__(self, key):
        """Indexing works like in a 1D numpy array, returning a new EliteTable.

        The exception is when the index is an integer -- in this case, the Elite
        at that position is returned.
        """
        if isinstance(key, (int, np.integer)):
            return Elite(
                self._solutions[key],
                self._objective_values[key],
                self._behavior_values[key],
                convert_idx(self._indices[key]),
                self._metadata[key],
            )

        return EliteTable(
            self._solutions[key],
            self._objective_values[key],
            self._behavior_values[key],
            self._indices[key],
            self._metadata[key],
        )

    def item(self):
        """If there is only one elite in the table, this method returns it.

        Similar to :meth:`numpy.ndarray.item`.

        Returns:
            The one :class:`Elite` in the table.
        Raises:
            ValueError: The number of elites in the table is not equal to one.
        """
        if len(self) != 1:
            raise ValueError(
                f"Must have one elite to call item() but there are {len(self)}")

        return Elite(
            self._solutions[0],
            self._objective_values[0],
            self._behavior_values[0],
            convert_idx(self._indices[0]),
            self._metadata[0],
        )

    def filter(self, predicate):
        """Filters the elites in the table according to ``predicate``.

        Args:
            predicate: A function which takes in a single :class:`Elite` and
                returns a bool indicating if the elite should be kept.
        Returns:
            EliteTable: New table only containing elites for which ``predicate``
            evaluates to True.
        """
        filter_indices = [bool(predicate(elite)) for elite in self]
        return EliteTable(
            self._solutions[filter_indices],
            self._objective_values[filter_indices],
            self._behavior_values[filter_indices],
            self._indices[filter_indices],
            self._metadata[filter_indices],
        )
