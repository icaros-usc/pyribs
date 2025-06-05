"""Provides ArchiveBase."""
from abc import ABC, abstractmethod


class ArchiveBase(ABC):
    """Base class for archives."""

    @property
    @abstractmethod
    def stats(self):
        """:class:`~ribs.archives.ArchiveStats`: Statistics about the archive.

        See :class:`~ribs.archives.ArchiveStats` for more info.
        """

    @property
    @abstractmethod
    def empty(self):
        """bool: Whether the archive is empty."""

    # TODO: Make this not required.
    @abstractmethod
    def clear(self):
        """Removes all elites from the archive.

        After this method is called, the archive will be :attr:`empty`.
        """

    # TODO: Tidy up return value; update docstring.
    @abstractmethod
    def add(self, solution, objective, measures, **fields):
        """Inserts a batch of solutions into the archive.

        Each solution is only inserted if it has a higher ``objective`` than the
        threshold of the corresponding cell. For the default values of
        ``learning_rate`` and ``threshold_min``, this threshold is simply the
        objective value of the elite previously in the cell.  If multiple
        solutions in the batch end up in the same cell, we only insert the
        solution with the highest objective. If multiple solutions end up in the
        same cell and tie for the highest objective, we insert the solution that
        appears first in the batch.

        For the default values of ``learning_rate`` and ``threshold_min``, the
        threshold for each cell is updated by taking the maximum objective value
        among all the solutions that landed in the cell, resulting in the same
        behavior as in the vanilla MAP-Elites archive. However, for other
        settings, the threshold is updated with the batch update rule described
        in the appendix of `Fontaine 2022 <https://arxiv.org/abs/2205.10752>`_.

        .. note:: The indices of all arguments should "correspond" to each
            other, i.e. ``solution[i]``, ``objective[i]``,
            ``measures[i]``, and should be the solution parameters,
            objective, and measures for solution ``i``.

        Args:
            solution (array-like): (batch_size, :attr:`solution_dim`) array of
                solution parameters.
            objective (array-like): (batch_size,) array with objective function
                evaluations of the solutions.
            measures (array-like): (batch_size, :attr:`measure_dim`) array with
                measure space coordinates of all the solutions.
            fields (keyword arguments): Additional data for each solution. Each
                argument should be an array with batch_size as the first
                dimension.

        Returns:
            dict: Information describing the result of the add operation. The
            dict contains the following keys:

            - ``"status"`` (:class:`numpy.ndarray` of :class:`int`): An array of
              integers that represent the "status" obtained when attempting to
              insert each solution in the batch. Each item has the following
              possible values:

              - ``0``: The solution was not added to the archive.
              - ``1``: The solution improved the objective value of a cell
                which was already in the archive.
              - ``2``: The solution discovered a new cell in the archive.

              All statuses (and values, below) are computed with respect to the
              *current* archive. For example, if two solutions both introduce
              the same new archive cell, then both will be marked with ``2``.

              The alternative is to depend on the order of the solutions in the
              batch -- for example, if we have two solutions ``a`` and ``b``
              which introduce the same new cell in the archive, ``a`` could be
              inserted first with status ``2``, and ``b`` could be inserted
              second with status ``1`` because it improves upon ``a``. However,
              our implementation does **not** do this.

              To convert statuses to a more semantic format, cast all statuses
              to :class:`AddStatus` e.g. with ``[AddStatus(s) for s in
              add_info["status"]]``.

            - ``"value"`` (:class:`numpy.ndarray` of
              :attr:`dtypes` ["objective"]): An array with values for each
              solution in the batch. With the default values of ``learning_rate
              = 1.0`` and ``threshold_min = -np.inf``, the meaning of each value
              depends on the corresponding ``status`` and is identical to that
              in CMA-ME (`Fontaine 2020 <https://arxiv.org/abs/1912.02400>`_):

              - ``0`` (not added): The value is the "negative improvement," i.e.
                the objective of the solution passed in minus the objective of
                the elite still in the archive (this value is negative because
                the solution did not have a high enough objective to be added to
                the archive).
              - ``1`` (improve existing cell): The value is the "improvement,"
                i.e. the objective of the solution passed in minus the objective
                of the elite previously in the archive.
              - ``2`` (new cell): The value is just the objective of the
                solution.

              In contrast, for other values of ``learning_rate`` and
              ``threshold_min``, each value is equivalent to the objective value
              of the solution minus the threshold of its corresponding cell in
              the archive.

        Raises:
            ValueError: The array arguments do not match their specified shapes.
            ValueError: ``objective`` or ``measures`` has non-finite values (inf
                or NaN).
        """

    # TODO: Specify it is not required to be implemented.
    def add_single(self, solution, objective, measures, **fields):
        """Inserts a single solution into the archive.

        The solution is only inserted if it has a higher ``objective`` than the
        threshold of the corresponding cell. For the default values of
        ``learning_rate`` and ``threshold_min``, this threshold is simply the
        objective value of the elite previously in the cell.  The threshold is
        also updated if the solution was inserted.

        .. note::
            To make it more amenable to modifications, this method's
            implementation is designed to be readable at the cost of
            performance, e.g., none of its operations are modified. If you need
            performance, we recommend using :meth:`add`.

        Args:
            solution (array-like): Parameters of the solution.
            objective (float): Objective function evaluation of the solution.
            measures (array-like): Coordinates in measure space of the solution.
            fields (keyword arguments): Additional data for the solution.

        Returns:
            dict: Information describing the result of the add operation. The
            dict contains ``status`` and ``value`` keys; refer to :meth:`add`
            for the meaning of status and value.

        Raises:
            ValueError: The array arguments do not match their specified shapes.
            ValueError: ``objective`` is non-finite (inf or NaN) or ``measures``
                has non-finite values.
        """
        # TODO
        raise NotImplementedError()

    @abstractmethod
    def retrieve(self, measures):
        """Retrieves the elites with measures in the same cells as the measures
        specified.

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
        # TODO
        raise NotImplementedError()

    # TODO: I'm unclear whether we want this here, as it seems to be a very
    # specific form of selection.
    @abstractmethod
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

    # TODO: Unclear what parameters should be.
    @abstractmethod
    def data(self, fields=None, return_type="dict"):
        """Retrieves data for all elites in the archive.

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
