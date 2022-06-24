"""Provides ArchiveBase."""
from abc import ABC, abstractmethod
from collections import OrderedDict

import numba as nb
import numpy as np

from ribs._utils import check_measures_batch_shape, check_measures_shape
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

    def _stats_update(self, old_obj, new_obj):
        """Updates the archive stats when old_obj is replaced by new_obj.

        A new namedtuple is created so that stats which have been collected
        previously do not change.
        """
        new_qd_score = self._stats.qd_score + new_obj - old_obj
        self._stats = ArchiveStats(
            num_elites=len(self),
            coverage=self.dtype(len(self) / self.cells),
            qd_score=new_qd_score,
            obj_max=new_obj if self._stats.obj_max is None else max(
                self._stats.obj_max, new_obj),
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
        check_measures_shape(measures, self.behavior_dim)
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

    def add(self, solution, objective_value, behavior_values, metadata=None):
        """Attempts to insert a new solution into the archive.

        The solution is only inserted if it has a higher ``objective_value``
        than the elite previously in the corresponding cell.

        Args:
            solution (array-like): Parameters of the solution.
            objective_value (float): Objective function evaluation of the
                solution.
            behavior_values (array-like): Coordinates in behavior space of the
                solution.
            metadata (object): Any Python object representing metadata for the
                solution. For instance, this could be a dict with several
                properties.
        Returns:
            tuple: 2-element tuple describing the result of the add operation.
            These outputs are particularly useful for algorithms such as CMA-ME.

                **status** (:class:`AddStatus`): See :class:`AddStatus`.

                **value** (:attr:`dtype`): The meaning of this value depends on
                the value of ``status``:

                - ``NOT_ADDED`` -> the "negative improvement," i.e. objective
                  value of solution passed in minus objective value of the
                  solution still in the archive (this value is negative because
                  the solution did not have a high enough objective value to be
                  added to the archive)
                - ``IMPROVE_EXISTING`` -> the "improvement," i.e. objective
                  value of solution passed in minus objective value of solution
                  previously in the archive
                - ``NEW`` -> the objective value passed in
        """
        self._state["add"] += 1
        solution = np.asarray(solution)
        behavior_values = np.asarray(behavior_values)
        objective_value = self.dtype(objective_value)

        index = self.index_of(behavior_values[None])[0]
        old_objective = self._objective_values[index]
        was_inserted, already_occupied = self._add_numba(
            index, solution, objective_value, behavior_values, self._occupied,
            self._solutions, self._objective_values, self._behavior_values)

        if was_inserted:
            self._metadata[index] = metadata

        if was_inserted and not already_occupied:
            self._add_occupied_index(index)
            status = AddStatus.NEW
            value = objective_value
            self._stats_update(self.dtype(0.0), objective_value)
        elif was_inserted and already_occupied:
            status = AddStatus.IMPROVE_EXISTING
            value = objective_value - old_objective
            self._stats_update(old_objective, objective_value)
        else:
            status = AddStatus.NOT_ADDED
            value = objective_value - old_objective
        return status, value

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
        check_measures_batch_shape(measures_batch, self.behavior_dim)

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
        check_measures_shape(measures, self.behavior_dim)

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
