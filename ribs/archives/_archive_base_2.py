"""Provides ArchiveBase."""
from abc import ABC


class ArchiveBase(ABC):
    """Base class for archives.

    Archives in pyribs store solutions and their objective and measure values,
    as well as associated components like k-D trees and density estimators. The
    primary method of an archive is to :meth:`add` new solutions. There are also
    methods to read from the archive, such as :meth:`retrieve` and :meth:`data`.

    Due to the flexibility of archives and workflows available in pyribs, it is
    possible to design algorithms that use only a small subset of these methods.
    As such, none of the methods listed here are required to be implemented in
    child classes, although by default they will raise
    :class:`NotImplementedError` when called.

    Args:
        solution_dim (int): Dimensionality of the solution space.
        objective_dim (int or empty tuple): Dimensionality of the objective space.
            Typically, we consider single-objective optimization problems where
            the objective is a scalar, in which case this argument should be an
            empty tuple ``()``. In multi-objective optimization problems, this
            argument should be an integer indicating the number of objectives.
        measure_dim (int): Dimensionality of the measure space.
    """

    def __init__(self, *, solution_dim, objective_dim, measure_dim):
        self._solution_dim = solution_dim
        self._objective_dim = objective_dim
        self._measure_dim = measure_dim

    ## Properties of the archive ##

    @property
    def solution_dim(self):
        """int: Dimensionality of the solution space."""
        return self._solution_dim

    @property
    def objective_dim(self):
        """int or empty tuple: Dimensionality of the objective space.

        The empty tuple ``()`` indicates a scalar objective.
        """
        return self._objective_dim

    @property
    def measure_dim(self):
        """int: Dimensionality of the measure space."""
        return self._measure_dim

    @property
    def stats(self):
        """:class:`~ribs.archives.ArchiveStats`: Statistics about the archive.

        See :class:`~ribs.archives.ArchiveStats` for more info.
        """
        raise NotImplementedError(
            "`stats` has not been implemented in this archive")

    @property
    def empty(self):
        """bool: Whether the archive is empty."""
        raise NotImplementedError(
            "`empty` has not been implemented in this archive")

    ## Methods for writing to the archive ##

    def add(self, solution, objective, measures, **fields):
        """Inserts a batch of solutions into the archive.

        The indices of all arguments should "correspond" to each other, i.e.,
        ``solution[i]``, ``objective[i]``, and ``measures[i]`` should be the
        solution parameters, objective, and measures for solution ``i``.

        For API consistency, all child classes should take in ``solution``,
        ``objective``, and ``measures``. There may be cases where one of these
        parameters is not necessary, e.g., ``objective`` is not required in
        diversity optimization settings. In such cases, it should be possible to
        pass in ``None`` as the argument.

        Args:
            solution (array-like): (batch_size, :attr:`solution_dim`) array of
                solution parameters.
            objective (array-like): (batch_size, :attr:`objective_dim`) array
                with objective function evaluations of the solutions.
            measures (array-like): (batch_size, :attr:`measure_dim`) array with
                measure space coordinates of all the solutions.
            fields (keyword arguments): Additional data for each solution. Each
                argument should be an array with ``batch_size`` as the first
                dimension.

        Returns:
            dict: Information describing the result of the add operation. The
            content of the dict is to be determined by child classes.
        """
        raise NotImplementedError(
            "`add` has not been implemented in this archive")

    def add_single(self, solution, objective, measures, **fields):
        """Inserts a single solution into the archive.

        Child classes are not required to implement this method.

        Args:
            solution (array-like): Parameters of the solution.
            objective (scalar or array-like): Objective function evaluation of
                the solution.
            measures (array-like): Coordinates in measure space of the solution.
            fields (keyword arguments): Additional data for the solution.

        Returns:
            dict: Information describing the result of the add operation. As in
            :meth:`add`, the content of this dict is decided by child classes.
        """
        raise NotImplementedError(
            "`add_single` has not been implemented in this archive")

    def clear(self):
        """Resets the archive, e.g., by removing all elites in it.

        After calling this method, the archive will be :attr:`empty`.

        Child classes are not required to implement this method.
        """
        raise NotImplementedError(
            "`clear` has not been implemented in this archive")

    ## Methods for reading from the archive ##

    def retrieve(self, measures):
        """Queries the archive for a batch of solutions with the given measures.

        This method operates in batch, i.e., it takes in a batch of measures and
        outputs the batched data for the elites::

            occupied, elites = archive.retrieve(...)
            elites["solution"]  # Shape: (batch_size, solution_dim)
            elites["objective"]
            elites["measures"]
            elites["threshold"]
            elites["index"]

        If the cell associated with ``elites["measures"][i]`` has an elite in
        it, then ``occupied[i]`` will be True. Furthermore,
        ``elites["solution"][i]``, ``elites["objective"][i]``,
        ``elites["measures"][i]``, ``elites["threshold"][i]``, and
        ``elites["index"][i]`` will be set to the properties of the elite. Note
        that ``elites["measures"][i]`` may not be equal to the ``measures[i]``
        passed as an argument, since the measures only need to be in the same
        archive cell.

        If the cell associated with ``measures[i]`` *does not* have any elite in
        it, then ``occupied[i]`` will be set to False. Furthermore, the
        corresponding outputs will be set to empty values -- namely:

        * NaN for floating-point fields
        * -1 for the "index" field
        * 0 for integer fields
        * None for object fields

        If you need to retrieve a *single* elite associated with some measures,
        consider using :meth:`retrieve_single`.

        Args:
            measures (array-like): (batch_size, :attr:`measure_dim`) array of
                coordinates in measure space.
        Returns:
            tuple: 2-element tuple of (occupied array, dict). The occupied array
            indicates whether each of the cells indicated by the coordinates in
            ``measures`` has an elite, while the dict contains the data of those
            elites. The dict maps from field name to the corresponding array.
        Raises:
            ValueError: ``measures`` is not of shape (batch_size,
                :attr:`measure_dim`).
            ValueError: ``measures`` has non-finite values (inf or NaN).
        """
        raise NotImplementedError(
            "`retrieve` has not been implemented in this archive")

    def retrieve_single(self, measures):
        """Retrieves the elite with measures in the same cell as the measures
        specified.

        While :meth:`retrieve` takes in a *batch* of measures, this method takes
        in the measures for only *one* solution and returns a single bool and a
        dict with single entries.

        Args:
            measures (array-like): (:attr:`measure_dim`,) array of measures.
        Returns:
            tuple: If there is an elite with measures in the same cell as the
            measures specified, then this method returns a True value and a dict
            where all the fields hold the info of the elite. Otherwise, this
            method returns a False value and a dict filled with the same "empty"
            values described in :meth:`retrieve`.
        Raises:
            ValueError: ``measures`` is not of shape (:attr:`measure_dim`,).
            ValueError: ``measures`` has non-finite values (inf or NaN).
        """
        raise NotImplementedError(
            "`retrieve_single` has not been implemented in this archive")

    # TODO: I'm unclear whether we want this here, as it seems to be a very
    # specific form of selection.
    def sample_elites(self, n):
        """Randomly samples elites from the archive.

        Currently, this sampling is done uniformly at random. Furthermore, each
        sample is done independently, so elites may be repeated in the sample.
        Additional sampling methods may be supported in the future.

        Example:

            ::

                elites = archive.sample_elites(16)
                elites["solution"]  # Shape: (16, solution_dim)
                elites["objective"]
                ...

        Args:
            n (int): Number of elites to sample.
        Returns:
            dict: Holds a batch of elites randomly selected from the archive.
        Raises:
            IndexError: The archive is empty.
        """
        raise NotImplementedError(
            "`sample_elites` has not been implemented in this archive")

    # TODO: Unclear what parameters should be.
    def data(self, fields=None, return_type="dict"):
        """Returns all data in the archive.

        Args:
            fields (str or array-like of str): List of fields to include. By
                default, all fields will be included, with an additional "index"
                as the last field ("index" can also be placed anywhere in this
                list). This can also be a single str indicating a field name.
            return_type (str): Type of data to return. See below. Ignored if
                ``fields`` is a str.

        Returns:
            The data for all entries in the archive. If ``fields`` was a single
            str, this will just be an array holding data for the given field.
            Otherwise, this data can take the following forms, depending on the
            ``return_type`` argument:

            - ``return_type="dict"``: Dict mapping from the field name to the
              field data at the given indices. An example is::

                  {
                    "solution": [[1.0, 1.0, ...], ...],
                    "objective": [1.5, ...],
                    "measures": [[1.0, 2.0], ...],
                    "threshold": [0.8, ...],
                    "index": [4, ...],
                  }

              Observe that we also return the indices as an ``index`` entry in
              the dict. The keys in this dict can be modified with the
              ``fields`` arg; duplicate fields will be ignored since the dict
              stores unique keys.

            - ``return_type="tuple"``: Tuple of arrays matching the field order
              given in ``fields``. For instance, if ``fields`` was
              ``["objective", "measures"]``, we would receive a tuple of
              ``(objective_arr, measures_arr)``. In this case, the results
              from ``retrieve`` could be unpacked as::

                  objective, measures = archive.data(["objective", "measures"],
                                                     return_type="tuple")

              Unlike with the ``dict`` return type, duplicate fields will show
              up as duplicate entries in the tuple, e.g.,
              ``fields=["objective", "objective"]`` will result in two
              objective arrays being returned.

              By default, (i.e., when ``fields=None``), the fields in the tuple
              will be ordered according to the :attr:`field_list` along with
              ``index`` as the last field.

            - ``return_type="pandas"``: A
              :class:`~ribs.archives.ArchiveDataFrame` with the following
              columns:

              - For fields that are scalars, a single column with the field
                name. For example, ``objective`` would have a single column
                called ``objective``.
              - For fields that are 1D arrays, multiple columns with the name
                suffixed by its index. For instance, if we have a ``measures``
                field of length 10, we create 10 columns with names
                ``measures_0``, ``measures_1``, ..., ``measures_9``. We do not
                currently support fields with >1D data.
              - 1 column of integers (``np.int32``) for the index, named
                ``index``.

              In short, the dataframe might look like this by default:

              +------------+------+-----------+------------+------+-----------+-------+
              | solution_0 | ...  | objective | measures_0 | ...  | threshold | index |
              +============+======+===========+============+======+===========+=======+
              |            | ...  |           |            | ...  |           |       |
              +------------+------+-----------+------------+------+-----------+-------+

              Like the other return types, the columns can be adjusted with
              the ``fields`` parameter.

            All data returned by this method will be a copy, i.e., the data will
            not update as the archive changes.
        """ # pylint: disable = line-too-long
        raise NotImplementedError(
            "`data` has not been implemented in this archive")
