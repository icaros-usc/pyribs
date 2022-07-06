"""Provides ArchiveBase."""
from abc import ABC, abstractmethod
from collections import OrderedDict

import numba as nb
import numpy as np
from numpy_groupies import aggregate_nb as aggregate

from ribs._utils import check_1d_shape, check_batch_shape
from ribs.archives._add_status import AddStatus
from ribs.archives._archive_data_frame import ArchiveDataFrame
from ribs.archives._archive_stats import ArchiveStats
from ribs.archives._elite import Elite, EliteBatch


def readonly(arr):
    """Sets an array to be readonly."""
    arr.flags.writeable = False
    return arr


class ArchiveIterator:
    """An iterator for an archive's elites."""

    # pylint: disable = protected-access

    def __init__(self, archive):
        self.archive = archive
        self.iter_idx = 0
        self.state = archive._state.copy()

    def __iter__(self):
        """This is the iterator, so it returns itself."""
        return self

    def __next__(self):
        """Raises RuntimeError if the archive was modified with add() or
        clear()."""
        if self.state != self.archive._state:
            # This check should go first because a call to clear() would clear
            # _occupied_indices and cause StopIteration to happen early.
            raise RuntimeError(
                "Archive was modified with add() or clear() during iteration.")
        if self.iter_idx >= len(self.archive):
            raise StopIteration

        idx = self.archive._occupied_indices[self.iter_idx]
        self.iter_idx += 1
        return Elite(
            self.archive._solutions[idx],
            self.archive._objective_values[idx],
            self.archive._behavior_values[idx],
            idx,
            self.archive._metadata[idx],
        )


class ArchiveBase(ABC):  # pylint: disable = too-many-instance-attributes
    """Base class for archives.

    This class assumes all archives use a fixed-size container with cells that
    hold (1) information about whether the cell is occupied (bool), (2) a
    solution (1D array), (3) objective function evaluation of the solution
    (float), (4) measure space coordinates of the solution (1D array), and (5)
    any additional metadata associated with the solution (object). In this
    class, the container is implemented with separate numpy arrays that share
    common dimensions. Using the ``solution_dim``, ``cells`, and
    ``behavior_dim`` arguments in ``__init__``, these arrays are as follows:

    +------------------------+----------------------------+
    | Name                   |  Shape                     |
    +========================+============================+
    | ``_occupied``          |  ``(cells,)``              |
    +------------------------+----------------------------+
    | ``_solutions``         |  ``(cells, solution_dim)`` |
    +------------------------+----------------------------+
    | ``_objective_values``  |  ``(cells,)``              |
    +------------------------+----------------------------+
    | ``_behavior_values``   |  ``(cells, behavior_dim)`` |
    +------------------------+----------------------------+
    | ``_metadata``          |  ``(cells,)``              |
    +------------------------+----------------------------+

    All of these arrays are accessed via a common integer index. If we have
    index ``i``, we access its solution at ``_solutions[i]``, its behavior
    values at ``_behavior_values[i]``, etc.

    Thus, child classes typically override the following methods:

    - ``__init__``: Child classes must invoke this class's ``__init__`` with the
      appropriate arguments.
    - :meth:`index_of`: Returns integer indices into the arrays above when
      given a batch of measures. Usually, each index has a meaning, e.g. in
      :class:`~ribs.archives.CVTArchive` it is the index of a centroid.
      Documentation for this method should describe the meaning of the index.

    .. note:: Attributes beginning with an underscore are only intended to be
        accessed by child classes (i.e. they are "protected" attributes).

    Args:
        solution_dim (int): Dimension of the solution space.
        cells (int): Number of cells in the archive. This is used to create the
            numpy arrays described above for storing archive info.
        behavior_dim (int): The dimension of the behavior space.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
        dtype (str or data-type): Data type of the solutions, objective values,
            and behavior values. We only support ``"f"`` / ``np.float32`` and
            ``"d"`` / ``np.float64``.
    Attributes:
        _solution_dim (int): See ``solution_dim`` arg.
        _rng (numpy.random.Generator): Random number generator, used in
            particular for generating random elites.
        _cells (int): See ``cells`` arg.
        _behavior_dim (int): See ``behavior_dim`` arg.
        _occupied (numpy.ndarray): Bool array storing whether each cell in the
            archive is occupied.
        _solutions (numpy.ndarray): Float array storing the solutions
            themselves.
        _objective_values (numpy.ndarray): Float array storing the objective
            value of each solution.
        _behavior_values (numpy.ndarray): Float array storing the behavior
            space coordinates of each solution.
        _metadata (numpy.ndarray): Object array storing the metadata associated
            with each solution.
        _occupied_indices (numpy.ndarray): A ``(cells,)`` array of integer
            (``np.int32``) indices that are occupied in the archive. This could
            be a list, but for efficiency, we make it a fixed-size array, where
            only the first ``_num_occupied`` entries will be valid.
        _num_occupied (int): Number of elites currently in the archive. This is
            used to index into ``_occupied_indices``.
    """

    def __init__(self,
                 solution_dim,
                 cells,
                 behavior_dim,
                 seed=None,
                 dtype=np.float64):

        ## Intended to be accessed by child classes. ##
        self._solution_dim = solution_dim
        self._rng = np.random.default_rng(seed)
        self._cells = cells
        self._behavior_dim = behavior_dim
        self._dtype = self._parse_dtype(dtype)

        self._num_occupied = 0
        self._occupied = np.zeros(self._cells, dtype=bool)
        self._occupied_indices = np.empty(self._cells, dtype=np.int32)

        self._solutions = np.empty((self._cells, solution_dim),
                                   dtype=self.dtype)
        self._objective_values = np.empty(self._cells, dtype=self.dtype)
        self._behavior_values = np.empty((self._cells, self._behavior_dim),
                                         dtype=self.dtype)
        self._metadata = np.empty(self._cells, dtype=object)

        self._stats = None
        self._stats_reset()

        # Tracks archive modifications by counting calls to clear() and add().
        self._state = {"clear": 0, "add": 0}

        ## Not intended to be accessed by children. ##
        self._seed = seed

    @staticmethod
    def _parse_dtype(dtype):
        """Parses the dtype passed into the constructor.

        Returns:
            np.float32 or np.float64
        Raises:
            ValueError: There is an error in the bounds configuration.
        """
        # First convert str dtype's to np.dtype.
        if isinstance(dtype, str):
            dtype = np.dtype(dtype)

        # np.dtype is not np.float32 or np.float64, but it compares equal.
        if dtype == np.float32:
            return np.float32
        if dtype == np.float64:
            return np.float64

        raise ValueError("Unsupported dtype. Must be np.float32 or np.float64")

    @property
    def cells(self):
        """int: Total number of cells in the archive."""
        return self._cells

    @property
    def empty(self):
        """bool: Whether the archive is empty."""
        return self._num_occupied == 0

    @property
    def behavior_dim(self):
        """int: Dimensionality of the behavior space."""
        return self._behavior_dim

    @property
    def solution_dim(self):
        """int: Dimensionality of the solutions in the archive."""
        return self._solution_dim

    @property
    def stats(self):
        """:class:`ArchiveStats`: Statistics about the archive.

        See :class:`ArchiveStats` for more info.
        """
        return self._stats

    @property
    def dtype(self):
        """data-type: The dtype of the solutions, objective values, and behavior
        values."""
        return self._dtype

    def __len__(self):
        """Number of elites in the archive."""
        return self._num_occupied

    def __iter__(self):
        """Creates an iterator over the :class:`Elite`'s in the archive.

        Example:

            ::

                for elite in archive:
                    elite.sol
                    elite.obj
                    ...
        """
        return ArchiveIterator(self)

    def _stats_reset(self):
        """Resets the archive stats."""
        self._stats = ArchiveStats(0, self.dtype(0.0), self.dtype(0.0), None,
                                   None)

    def _stats_update(self, old_objective_batch, new_objective_batch):
        """Updates the archive stats when the batch of old objectives is
        replaced by the batch of new objectives.

        A new namedtuple is created so that stats which have been collected
        previously do not change.
        """
        new_qd_score = self._stats.qd_score + np.sum(new_objective_batch -
                                                     old_objective_batch)
        max_new_obj = np.max(new_objective_batch)
        self._stats = ArchiveStats(
            num_elites=len(self),
            coverage=self.dtype(len(self) / self.cells),
            qd_score=new_qd_score,
            obj_max=max_new_obj if self._stats.obj_max is None else max(
                self._stats.obj_max, max_new_obj),
            obj_mean=new_qd_score / self.dtype(len(self)),
        )

    def clear(self):
        """Removes all elites from the archive.

        After this method is called, the archive will be :attr:`empty`.
        """
        # Only ``self._occupied_indices`` and ``self._occupied`` are cleared, as
        # a cell can have arbitrary values when its index is marked as
        # unoccupied.
        self._num_occupied = 0
        self._occupied.fill(False)

        self._state["clear"] += 1
        self._state["add"] = 0

        self._stats_reset()

    @abstractmethod
    def index_of(self, measures_batch):
        """Returns archive indices for the given batch of measures.

        If you need to retrieve the index of the measures for a *single*
        solution, consider using :meth:`index_of_single`.

        Args:
            measures_batch (array-like): (batch_size, :attr:`behavior_dim`)
                array of coordinates in measure space.
        Returns:
            (numpy.ndarray): (batch_size,) array with the indices of the
            batch of measures in the archive's storage arrays.
        """

    def index_of_single(self, measures):
        """Returns the index of the measures for one solution.

        While :meth:`index_of` takes in a *batch* of measures, this method takes
        in the measures for only *one* solution. If :meth:`index_of` is
        implemented correctly, this method should work immediately (i.e. `"out
        of the box" <https://idioms.thefreedictionary.com/Out-of-the-Box>`_).

        Args:
            measures (array-like): (:attr:`behavior_dim`,) array of measures for
                a single solution.
        Returns:
            int or numpy.integer: Integer index of the measures in the archive's
            storage arrays.
        Raises:
            ValueError: ``measures`` is not of shape (:attr:`behavior_dim`,).
        """
        measures = np.asarray(measures)
        check_1d_shape(measures, "measures", self.behavior_dim, "measure_dim")
        return self.index_of(measures[None])[0]

    @staticmethod
    @nb.jit(locals={"already_occupied": nb.types.b1}, nopython=True)
    def _add_numba(new_index, new_solution, new_objective_value,
                   new_behavior_values, occupied, solutions, objective_values,
                   behavior_values):
        """Numba helper for inserting solutions into the archive.

        See add() for usage.

        Returns:
            was_inserted (bool): Whether the new values were inserted into the
                archive.
            already_occupied (bool): Whether the index was occupied prior
                to this call; i.e. this is True only if there was already an
                item at the index.
        """
        already_occupied = occupied[new_index]
        if (not already_occupied or
                objective_values[new_index] < new_objective_value):
            # Track this index if it has not been seen before -- important that
            # we do this before inserting the solution.
            if not already_occupied:
                occupied[new_index] = True

            # Insert into the archive.
            objective_values[new_index] = new_objective_value
            behavior_values[new_index] = new_behavior_values
            solutions[new_index] = new_solution

            return True, already_occupied

        return False, already_occupied

    def _add_occupied_index(self, index):
        """Tracks a new occupied index."""
        self._occupied_indices[self._num_occupied] = index
        self._num_occupied += 1

    def add(self,
            solution_batch,
            objective_batch,
            measures_batch,
            metadata_batch=None):
        """Inserts a batch of solutions into the archive.

        Each solution is only inserted if it has a higher objective than the
        elite previously in the corresponding cell. If multiple solutions in the
        batch end up in the same cell, we only keep the solution with the
        highest objective.

        .. note:: The indices of all arguments should "correspond" to each
            other, i.e. ``solution_batch[i]``, ``objective_batch[i]``,
            ``measures_batch[i]``, and ``metadata_batch[i]`` should be the
            solution parameters, objective, measures, and metadata for solution
            ``i``.

        Args:
            solution_batch (array-like): (batch_size, :attr:`solution_dim`)
                array of solution parameters.
            objective_batch (array-like): (batch_size,) array with objective
                function evaluations of the solutions.
            measures_batch (array-like): (batch_size, :attr:`behavior_dim`)
                array with measure space coordinates of all the solutions.
            metadata_batch (array-like): (batch_size,) array of Python objects
                representing metadata for the solution. For instance, this could
                be a dict with several properties.
        Returns:
            tuple: 2-element tuple of (status_batch, value_batch) which
            describes the results of the additions. These outputs are
            particularly useful for algorithms such as CMA-ME.

            - **status_batch** (:class:`numpy.ndarray` of :class:`int`): An
              array of integers which represent the "status" obtained when
              attempting to insert each solution in the batch. Each item has the
              following possible values:

              - ``0``: The solution was not added to the archive.
              - ``1``: The solution improved the objective value of a cell
                which was already in the archive.
              - ``2``: The solution discovered a new cell in the archive.

              .. note:: All statuses (and values, below) are computed with
                  respect to the *current* archive. For example, if two
                  solutions both introduce the same new archive cell, then both
                  will be marked with ``2``.

                  The alternative is to depend on the order of the solutions in
                  the batch -- for example, if we have two solutions ``a`` and
                  ``b`` which introduce the same new cell in the archive, ``a``
                  could be inserted first with status ``2``, and ``b`` could be
                  inserted second with status ``1`` because it improves upon
                  ``a``. However, our implementation does **not** do this.

              To convert statuses to a more semantic format, cast all statuses
              to :class:`AddStatus` e.g. with ``[AddStatus(s) for s in
              status_batch]``.

            - **value_batch** (:attr:`dtype`): An array with values for each
              solution in the batch. The meaning of each ``value`` depends on
              the corresponding ``status``:

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
        Raises:
            ValueError: The array arguments do not match their specified shapes.
        """
        self._state["add"] += 1
        solution_batch = np.asarray(solution_batch)
        measures_batch = np.asarray(measures_batch)
        index_batch = self.index_of(measures_batch)
        metadata_batch = (np.empty(index_batch.size, dtype=object) if
                          metadata_batch is None else np.asarray(metadata_batch,
                                                                 dtype=object))

        # TODO: Shape checks.
        # TODO: Note that we switched from single to batch in new pyribs.

        objective_batch = np.asarray(objective_batch, self.dtype)
        # Copy old objectives since we will be modifying the objectives storage.
        old_objective_batch = np.copy(self._objective_values[index_batch])

        already_occupied = self._occupied[index_batch]
        is_new = ~already_occupied
        improve_existing = (objective_batch >
                            old_objective_batch) & already_occupied
        old_objective_batch[is_new] = 0.0

        # TODO: Rename to status_batch and value_batch

        # Set add_statuses and add_values -- these differ from what is actually
        # inserted into the archive since there may be conflicts within the
        # solution list.
        add_statuses = np.zeros(index_batch.size, dtype=np.int32)
        add_statuses[is_new] = 2
        add_statuses[improve_existing] = 1

        # Since we set the new solutions in the old objective batch to have
        # value 0.0, the add_values for new solutions are correct here.
        add_values = objective_batch - old_objective_batch

        # Retrieve indices of solutions that should be inserted into the
        # archive. First, we get the argmax for each archive index -- we use a
        # fill_value of -1 to indicate archive indices which were not covered in
        # the batch.
        can_insert = is_new | improve_existing

        # Return early if we cannot insert anything -- continuing would actually
        # throw a ValueError in aggregate() since index_batch[can_insert] would
        # be empty.
        if not np.any(can_insert):
            return add_statuses, add_values

        # Note that the length of argmax is only max(index_batch[can_insert]),
        # rather than the total number of grid cells. However, this is okay
        # because we only want to find the indices of the solutions in
        # should_insert.
        archive_argmax = aggregate(index_batch[can_insert],
                                   objective_batch[can_insert],
                                   func="argmax",
                                   fill_value=-1)
        should_insert = archive_argmax[archive_argmax != -1]

        # Indices in the archive where we should insert a solution.
        archive_indices = index_batch[should_insert]

        inserted_objectives = objective_batch[should_insert]
        self._objective_values[archive_indices] = inserted_objectives
        self._behavior_values[archive_indices] = measures_batch[should_insert]
        self._solutions[archive_indices] = solution_batch[should_insert]
        self._metadata[archive_indices] = metadata_batch[should_insert]
        self._occupied[archive_indices] = True

        # Only counts new solutions that were inserted.
        is_new_and_inserted = is_new[should_insert]
        n_new = np.sum(is_new_and_inserted)
        self._occupied_indices[self._num_occupied:self._num_occupied +
                               n_new] = (index_batch[should_insert]
                                         [is_new_and_inserted])
        self._num_occupied += n_new

        # Update stats -- only account for solutions that are actually added.
        # The old objective can be tricky because it needs to be 0 for new
        # solutions.

        # Since we set the new solutions in the old objective batch to have
        # value 0.0, the objectives for new solutions are added in properly
        # here.
        new_qd_score = (
            self._stats.qd_score +
            np.sum(inserted_objectives - old_objective_batch[should_insert]))
        max_new_obj = np.max(inserted_objectives)
        self._stats = ArchiveStats(
            num_elites=len(self),
            coverage=self.dtype(len(self) / self.cells),
            qd_score=new_qd_score,
            obj_max=max_new_obj if self._stats.obj_max is None else max(
                self._stats.obj_max, max_new_obj),
            obj_mean=new_qd_score / self.dtype(len(self)),
        )

        return add_statuses, add_values

    def add_single(self, solution, objective, measures, metadata=None):
        """Inserts a single solution into the archive.

        The solution is only inserted if it has a higher ``objective_value``
        than the elite previously in the corresponding cell.

        Args:
            solution (array-like): Parameters of the solution.
            objective (float): Objective function evaluation of the solution.
            measures (array-like): Coordinates in measure space of the solution.
            metadata (object): Python object representing metadata for the
                solution. For instance, this could be a dict with several
                properties.
        Returns:
            tuple: 2-element tuple of (status, value) describing the result of
            the add operation. Refer to :meth:`add` for the meaning of the
            status and value.
        """
        #  self._state["add"] += 1
        #  solution = np.asarray(solution)
        #  behavior_values = np.asarray(behavior_values)
        #  objective_value = self.dtype(objective_value)

        #  index = self.index_of(behavior_values[None])[0]
        #  old_objective = self._objective_values[index]
        #  was_inserted, already_occupied = self._add_numba(
        #      index, solution, objective_value, behavior_values, self._occupied,
        #      self._solutions, self._objective_values, self._behavior_values)

        #  if was_inserted:
        #      self._metadata[index] = metadata

        #  if was_inserted and not already_occupied:
        #      self._add_occupied_index(index)
        #      status = AddStatus.NEW
        #      value = objective_value
        #      self._stats_update(self.dtype(0.0), objective_value)
        #  elif was_inserted and already_occupied:
        #      status = AddStatus.IMPROVE_EXISTING
        #      value = objective_value - old_objective
        #      self._stats_update(old_objective, objective_value)
        #  else:
        #      status = AddStatus.NOT_ADDED
        #      value = objective_value - old_objective
        #  return status, value

        # TODO: Implement this better.
        status, value = self.add([solution], [objective_value],
                                 [behavior_values], [metadata])
        return status[0], value[0]

    def elites_with_measures(self, measures_batch):
        """Retrieves the elites with measures in the same cells as the measures
        specified.

        This method operates in batch, i.e. it takes in a batch of measures and
        outputs an :namedtuple:`EliteBatch`. Since :namedtuple:`EliteBatch` is a
        namedtuple, it can be unpacked::

            solution_batch, objective_batch, measures_batch, \\
                index_batch, metadata_batch = archive.elites_with_measures(...)

        Or the fields may be accessed by name::

            elite_batch = archive.elites_with_measures(...)
            elite_batch.solution_batch
            elite_batch.objective_batch
            elite_batch.measures_batch
            elite_batch.index_batch
            elite_batch.metadata_batch

        If the cell associated with ``measures_batch[i]`` has an elite in it,
        then ``elite_batch.solution_batch[i]``,
        ``elite_batch.objective_batch[i]``, ``elite_batch.measures_batch[i]``,
        ``elite_batch.index_batch[i]``, and ``elite_batch.metadata_batch[i]``
        will be set to the properties of the elite. Note that
        ``elite_batch.measures_batch[i]`` may not be equal to
        ``measures_batch[i]`` since the measures only need to be in the same
        archive cell.

        If the cell associated with ``measures_batch[i]`` *does not* have any
        elite in it, then the corresponding outputs are set to empty values --
        namely:

        * ``elite_batch.solution_batch[i]`` will be an array of NaN
        * ``elite_batch.objective_batch[i]`` will be NaN
        * ``elite_batch.measures_batch[i]`` will be an array of NaN
        * ``elite_batch.index_batch[i]`` will be -1
        * ``elite_batch.metadata_batch[i]`` will be None

        If you need to retrieve a *single* elite associated with some measures,
        consider using :meth:`elites_with_measures_single`.

        Args:
            measures_batch (array-like): (batch_size, :attr:`behavior_dim`)
                array of coordinates in measure space.
        Returns:
            EliteBatch: See above.
        Raises:
            ValueError: ``measures_batch`` is not of shape (batch_size,
                :attr:`behavior_dim`).
        """
        measures_batch = np.asarray(measures_batch)
        check_batch_shape(measures_batch, "measures_batch", self.behavior_dim,
                          "measure_dim")

        index_batch = self.index_of(measures_batch)
        occupied_batch = self._occupied[index_batch]
        expanded_occupied_batch = occupied_batch[:, None]

        return EliteBatch(
            solution_batch=readonly(
                # For each occupied_batch[i], this np.where selects
                # self._solutions[index_batch][i] if occupied_batch[i] is True.
                # Otherwise, it uses the alternate value (a solution array
                # consisting of np.nan).
                np.where(
                    expanded_occupied_batch,
                    self._solutions[index_batch],
                    np.full(self._solution_dim, np.nan),
                )),
            objective_batch=readonly(
                np.where(
                    occupied_batch,
                    self._objective_values[index_batch],
                    # Here the alternative is just a scalar np.nan.
                    np.nan,
                )),
            measures_batch=readonly(
                np.where(
                    expanded_occupied_batch,
                    self._behavior_values[index_batch],
                    # And here it is a measures array of np.nan.
                    np.full(self._behavior_dim, np.nan),
                )),
            index_batch=readonly(
                np.where(
                    occupied_batch,
                    index_batch,
                    # Indices must be integers, so np.nan would not work, hence
                    # we use -1.
                    -1,
                )),
            metadata_batch=readonly(
                np.where(
                    occupied_batch,
                    self._metadata[index_batch],
                    None,
                )),
        )

    def elites_with_measures_single(self, measures):
        """Retrieves the elite with measures in the same cell as the measures
        specified.

        While :meth:`elites_with_measures` takes in a *batch* of measures, this
        method takes in the measures for only *one* solution and returns a
        single :namedtuple:`Elite`.

        Args:
            measures (array-like): (:attr:`behavior_dim`,) array of measures.
        Returns:
            If there is an elite with measures in the same cell as the measures
            specified, then this method returns an :namedtuple:`Elite` where all
            the fields hold the info of that elite. Otherwise, this method
            returns an :namedtuple:`Elite` filled with the same "empty" values
            described in :meth:`elites_with_measures`.
        Raises:
            ValueError: ``measures`` is not of shape (:attr:`behavior_dim`,).
        """
        measures = np.asarray(measures)
        check_1d_shape(measures, "measures", self.behavior_dim, "measure_dim")

        elite_batch = self.elites_with_measures(measures[None])
        return Elite(
            elite_batch.solution_batch[0],
            elite_batch.objective_batch[0],
            elite_batch.measures_batch[0],
            elite_batch.index_batch[0],
            elite_batch.metadata_batch[0],
        )

    def sample_elites(self, n):
        """Randomly samples elites from the archive.

        Currently, this sampling is done uniformly at random. Furthermore, each
        sample is done independently, so elites may be repeated in the sample.
        Additional sampling methods may be supported in the future.

        Since :namedtuple:`EliteBatch` is a namedtuple, the result can be
        unpacked (here we show how to ignore some of the fields)::

            solution_batch, objective_batch, measures_batch, *_ = \\
                archive.sample_elites(32)

        Or the fields may be accessed by name::

            elite = archive.sample_elites(16)
            elite.solution_batch
            elite.objective_batch
            ...

        Args:
            n (int): Number of elites to sample.
        Returns:
            EliteBatch: A batch of elites randomly selected from the archive.
        Raises:
            IndexError: The archive is empty.
        """
        if self.empty:
            raise IndexError("No elements in archive.")

        random_indices = self._rng.integers(self._num_occupied, size=n)
        selected_indices = self._occupied_indices[random_indices]

        return EliteBatch(
            readonly(self._solutions[selected_indices]),
            readonly(self._objective_values[selected_indices]),
            readonly(self._behavior_values[selected_indices]),
            readonly(selected_indices),
            readonly(self._metadata[selected_indices]),
        )

    def as_pandas(self, include_solutions=True, include_metadata=False):
        """Converts the archive into an :class:`ArchiveDataFrame` (a child class
        of :class:`pandas.DataFrame`).

        The implementation of this method in :class:`ArchiveBase` creates a
        dataframe consisting of:

        - 1 column of integers (``np.int32``) for the index, named ``index``.
          See :meth:`index_of` for more info.
        - :attr:`behavior_dim` columns for the behavior characteristics, named
          ``behavior_0, behavior_1, ...``
        - 1 column for the objective values, named ``objective``
        - :attr:`solution_dim` columns for the solution vectors, named
          ``solution_0, solution_1, ...``
        - 1 column for the metadata objects, named ``metadata``

        In short, the dataframe looks like this:

        +-------+-------------+------+------------+-------------+-----+----------+
        | index | behavior_0  | ...  | objective  | solution_0  | ... | metadata |
        +=======+=============+======+============+=============+=====+==========+
        |       |             | ...  |            |             | ... |          |
        +-------+-------------+------+------------+-------------+-----+----------+

        Compared to :class:`pandas.DataFrame`, the :class:`ArchiveDataFrame`
        adds methods and attributes which make it easier to manipulate archive
        data. For more information, refer to the :class:`ArchiveDataFrame`
        documentation.

        Args:
            include_solutions (bool): Whether to include solution columns.
            include_metadata (bool): Whether to include the metadata column.
                Note that methods like :meth:`~pandas.DataFrame.to_csv` may not
                properly save the dataframe since the metadata objects may not
                be representable in a CSV.
        Returns:
            ArchiveDataFrame: See above.
        """ # pylint: disable = line-too-long
        data = OrderedDict()
        indices = self._occupied_indices[:self._num_occupied]

        # Copy indices so we do not overwrite.
        data["index"] = np.copy(indices)

        behavior_values = self._behavior_values[indices]
        for i in range(self._behavior_dim):
            data[f"behavior_{i}"] = behavior_values[:, i]

        data["objective"] = self._objective_values[indices]

        if include_solutions:
            solutions = self._solutions[indices]
            for i in range(self._solution_dim):
                data[f"solution_{i}"] = solutions[:, i]

        if include_metadata:
            data["metadata"] = self._metadata[indices]

        return ArchiveDataFrame(
            data,
            copy=False,  # Fancy indexing above already results in copying.
        )
