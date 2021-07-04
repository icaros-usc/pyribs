"""Provides EliteTable."""
from ribs.archives._elite import Elite


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
        self._iter_idx = None

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
                     tuple(self._indices[0]), self._metadata[0])

    def __iter__(self):
        """Creates an iterator over the elites in the archive."""
        self._iter_idx = 0
        return self

    def __next__(self):
        """Returns the next :class:`Elite` in the table."""
        if self._iter_idx >= len(self):
            raise StopIteration

        elite = Elite(
            self._solutions[self._iter_idx],
            self._objective_values[self._iter_idx],
            self._behavior_values[self._iter_idx],
            tuple(self._indices[self._iter_idx]),
            self._metadata[self._iter_idx],
        )
        self._iter_idx += 1
        return elite
