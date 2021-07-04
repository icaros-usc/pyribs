"""Provides EliteTable."""
import numpy as np

from ribs.archives._elite import Elite

# TODO: Usage examples


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
        :meth:`~ArchiveBase.data` method (i.e. users do not construct it)::

            data = archive.data()

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
            return Elite(self._solutions[key], self._objective_values[key],
                         self._behavior_values[key],
                         convert_idx(self._indices[key]), self._metadata[key])

        return EliteTable(self._solutions[key], self._objective_values[key],
                          self._behavior_values[key], self._indices[key],
                          self._metadata[key])

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

        return Elite(self._solutions[0],
                     self._objective_values[0], self._behavior_values[0],
                     convert_idx(self._indices[0]), self._metadata[0])

    #  def filter(self, predicate):
    #      # Returns new EliteTable where elites are filtered by predicate
    #      pass
