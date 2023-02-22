"""Provides ArchiveBase."""
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
from numpy_groupies import aggregate_nb as aggregate

from ribs._utils import (check_1d_shape, check_batch_shape, check_finite,
                         check_is_1d, readonly, validate_batch_args,
                         validate_single_args)
from ribs.archives._archive_data_frame import ArchiveDataFrame
from ribs.archives._archive_stats import ArchiveStats
from ribs.archives._cqd_score_result import CQDScoreResult
from ribs.archives._elite import Elite, EliteBatch

_ADD_WARNING = (" Note that starting in pyribs 0.5.0, add() takes in a "
                "batch of solutions unlike in pyribs 0.4.0, where add() "
                "only took in a single solution.")


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
            self.archive._solution_arr[idx],
            self.archive._objective_arr[idx],
            self.archive._measures_arr[idx],
            idx,
            self.archive._metadata_arr[idx],
        )


class ArchiveBase(ABC):  # pylint: disable = too-many-instance-attributes
    """Base class for archives.

    This class assumes all archives use a fixed-size container with cells that
    hold (1) information about whether the cell is occupied (bool), (2) a
    solution (1D array), (3) objective function evaluation of the solution
    (float), (4) measure space coordinates of the solution (1D array), (5)
    any additional metadata associated with the solution (object), and (6) a
    threshold which determines how high an objective value must be for a
    solution to be inserted into a cell (float). In this class, the container is
    implemented with separate numpy arrays that share common dimensions. Using
    the ``solution_dim``, ``cells`, and ``measure_dim`` arguments in
    ``__init__``, these arrays are as follows:

    +------------------------+----------------------------+
    | Name                   |  Shape                     |
    +========================+============================+
    | ``_occupied_arr``      |  ``(cells,)``              |
    +------------------------+----------------------------+
    | ``_solution_arr``      |  ``(cells, solution_dim)`` |
    +------------------------+----------------------------+
    | ``_objective_arr``     |  ``(cells,)``              |
    +------------------------+----------------------------+
    | ``_measures_arr``      |  ``(cells, measure_dim)``  |
    +------------------------+----------------------------+
    | ``_metadata_arr``      |  ``(cells,)``              |
    +------------------------+----------------------------+
    | ``_threshold_arr``     |  ``(cells,)``              |
    +------------------------+----------------------------+

    All of these arrays are accessed via a common integer index. If we have
    index ``i``, we access its solution at ``_solution_arr[i]``, its measure
    values at ``_measures_arr[i]``, etc.

    Thus, child classes typically override the following methods:

    - ``__init__``: Child classes must invoke this class's ``__init__`` with the
      appropriate arguments.
    - :meth:`index_of`: Returns integer indices into the arrays above when
      given a batch of measures. Usually, each index has a meaning, e.g. in
      :class:`~ribs.archives.CVTArchive` it is the index of a centroid.
      Documentation for this method should describe the meaning of the index.

    .. note:: Attributes beginning with an underscore are only intended to be
        accessed by child classes (i.e. they are "protected" attributes).

    .. note:: The idea of archive thresholds was introduced in `Fontaine 2022
        <https://arxiv.org/abs/2205.10752>`_. Refer to our `CMA-MAE tutorial
        <../../tutorials/cma_mae.html>`_ for more info on thresholds, including
        the ``learning_rate`` and ``threshold_min`` parameters.

    Args:
        solution_dim (int): Dimension of the solution space.
        cells (int): Number of cells in the archive. This is used to create the
            numpy arrays described above for storing archive info.
        measure_dim (int): The dimension of the measure space.
        learning_rate (float): The learning rate for threshold updates.
        threshold_min (float): The initial threshold value for all the cells.
        qd_score_offset (float): Archives often contain negative objective
            values, and if the QD score were to be computed with these negative
            objectives, the algorithm would be penalized for adding new cells
            with negative objectives. Thus, a standard practice is to normalize
            all the objectives so that they are non-negative by introducing an
            offset. This QD score offset will be *subtracted* from all
            objectives in the archive, e.g., if your objectives go as low as
            -300, pass in -300 so that each objective will be transformed as
            ``objective - (-300)``.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
        dtype (str or data-type): Data type of the solutions, objectives,
            and measures. We only support ``"f"`` / ``np.float32`` and ``"d"`` /
            ``np.float64``.
    Attributes:
        _solution_dim (int): See ``solution_dim`` arg.
        _rng (numpy.random.Generator): Random number generator, used in
            particular for generating random elites.
        _cells (int): See ``cells`` arg.
        _measure_dim (int): See ``measure_dim`` arg.
        _occupied_arr (numpy.ndarray): Bool array storing whether each cell in
            the archive is occupied.
        _solution_arr (numpy.ndarray): Float array storing the solutions
            themselves.
        _objective_arr (numpy.ndarray): Float array storing the objective value
            of each solution.
        _measures_arr (numpy.ndarray): Float array storing the measure space
            coordinates of each solution.
        _metadata_arr (numpy.ndarray): Object array storing the metadata
            associated with each solution.
        _threshold_arr (numpy.ndarray): Float array storing the threshold for
            insertion into each cell.
        _occupied_indices (numpy.ndarray): A ``(cells,)`` array of integer
            (``np.int32``) indices that are occupied in the archive. This could
            be a list, but for efficiency, we make it a fixed-size array, where
            only the first ``_num_occupied`` entries are valid.
        _num_occupied (int): Number of elites currently in the archive. This is
            used to index into ``_occupied_indices``.
    """

    def __init__(self,
                 *,
                 solution_dim,
                 cells,
                 measure_dim,
                 learning_rate=1.0,
                 threshold_min=-np.inf,
                 qd_score_offset=0.0,
                 seed=None,
                 dtype=np.float64):

        ## Intended to be accessed by child classes. ##
        self._solution_dim = solution_dim
        self._rng = np.random.default_rng(seed)
        self._cells = cells
        self._measure_dim = measure_dim
        self._dtype = self._parse_dtype(dtype)

        self._num_occupied = 0
        self._occupied_arr = np.zeros(self._cells, dtype=bool)
        self._occupied_indices = np.empty(self._cells, dtype=np.int32)

        self._solution_arr = np.empty((self._cells, solution_dim),
                                      dtype=self.dtype)
        self._objective_arr = np.empty(self._cells, dtype=self.dtype)
        self._measures_arr = np.empty((self._cells, self._measure_dim),
                                      dtype=self.dtype)
        self._metadata_arr = np.empty(self._cells, dtype=object)

        if threshold_min == -np.inf and learning_rate != 1.0:
            raise ValueError("threshold_min can only be -np.inf if "
                             "learning_rate is 1.0")
        self._learning_rate = self._dtype(learning_rate)
        self._threshold_min = self._dtype(threshold_min)
        self._threshold_arr = np.full(self._cells,
                                      threshold_min,
                                      dtype=self.dtype)

        self._qd_score_offset = self._dtype(qd_score_offset)

        self._stats = None
        # Sum of all objective values in the archive; useful for computing
        # qd_score and obj_mean.
        self._objective_sum = None
        self._stats_reset()

        self._best_elite = None

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
    def measure_dim(self):
        """int: Dimensionality of the measure space."""
        return self._measure_dim

    @property
    def solution_dim(self):
        """int: Dimensionality of the solutions in the archive."""
        return self._solution_dim

    @property
    def learning_rate(self):
        """float: The learning rate for threshold updates."""
        return self._learning_rate

    @property
    def threshold_min(self):
        """float: The initial threshold value for all the cells."""
        return self._threshold_min

    @property
    def qd_score_offset(self):
        """float: The offset which is subtracted from objective values when
        computing the QD score."""
        return self._qd_score_offset

    @property
    def stats(self):
        """:class:`ArchiveStats`: Statistics about the archive.

        See :class:`ArchiveStats` for more info.
        """
        return self._stats

    @property
    def best_elite(self):
        """:class:`Elite`: The elite with the highest objective in the archive.

        None if there are no elites in the archive.

        .. note::
            If the archive is non-elitist (this occurs when using the archive
            with a learning rate which is not 1.0, as in CMA-MAE), then this
            best elite may no longer exist in the archive because it was
            replaced with an elite with a lower objective value. This can happen
            because in non-elitist archives, new solutions only need to exceed
            the *threshold* of the cell they are being inserted into, not the
            *objective* of the elite currently in the cell. See `#314
            <https://github.com/icaros-usc/pyribs/pull/314>`_ for more info.
        """
        return self._best_elite

    @property
    def dtype(self):
        """data-type: The dtype of the solutions, objective, and measures."""
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
        self._stats = ArchiveStats(
            num_elites=0,
            coverage=self.dtype(0.0),
            qd_score=self.dtype(0.0),
            norm_qd_score=self.dtype(0.0),
            obj_max=None,
            obj_mean=None,
        )
        self._objective_sum = self.dtype(0.0)

    def _compute_new_thresholds(self, threshold_arr, objective_batch,
                                index_batch, learning_rate):
        """Update thresholds.

        Args:
            threshold_arr (np.ndarray): The threshold of the cells before
                updating. 1D array.
            objective_batch (np.ndarray): The objective values of the solution
                that is inserted into the archive for each cell. 1D array. We
                assume that the objective values are all higher than the
                thresholds of their respective cells.
            index_batch (np.ndarray): The archive index of the elements in
                objective batch.
        Returns:
            `new_threshold_batch` (A self.dtype array of new
            thresholds) and `threshold_update_indices` (A boolean
            array indicating which entries in `threshold_arr` should
            be updated.
        """
        # Even though we do this check, it should not be possible to have
        # empty objective_batch or index_batch in the add() method since
        # we check that at least one cell is being updated by seeing if
        # can_insert has any True values.
        if objective_batch.size == 0 or index_batch.size == 0:
            return np.array([], dtype=self.dtype), np.array([], dtype=bool)

        # Compute the number of objectives inserted into each cell.
        objective_sizes = aggregate(index_batch,
                                    objective_batch,
                                    func="len",
                                    fill_value=0,
                                    size=threshold_arr.size)

        # These indices are with respect to the archive, so we can directly pass
        # them to threshold_arr.
        threshold_update_indices = objective_sizes > 0

        # Compute the sum of the objectives inserted into each cell.
        objective_sums = aggregate(index_batch,
                                   objective_batch,
                                   func="sum",
                                   fill_value=np.nan,
                                   size=threshold_arr.size)

        # Throw away indices that we do not care about.
        objective_sizes = objective_sizes[threshold_update_indices]
        objective_sums = objective_sums[threshold_update_indices]

        # Unlike in add_single, we do not need to worry about
        # old_threshold having -np.inf here as a result of threshold_min
        # being -np.inf. This is because the case with threshold_min =
        # -np.inf is handled separately since we compute the new
        # threshold based on the max objective in each cell in that case.
        old_threshold = np.copy(threshold_arr[threshold_update_indices])

        ratio = self.dtype(1.0 - learning_rate)**objective_sizes
        new_threshold_batch = (ratio * old_threshold +
                               (objective_sums / objective_sizes) * (1 - ratio))

        return new_threshold_batch, threshold_update_indices

    def clear(self):
        """Removes all elites from the archive.

        After this method is called, the archive will be :attr:`empty`.
        """
        # Clear ``self._occupied_indices`` and ``self._occupied_arr`` since a
        # cell can have arbitrary values when its index is marked as unoccupied.
        self._num_occupied = 0  # Corresponds to clearing _occupied_indices.
        self._occupied_arr.fill(False)

        # We also need to reset thresholds since archive addition is based on
        # thresholds.
        self._threshold_arr.fill(self._threshold_min)

        self._state["clear"] += 1
        self._state["add"] = 0

        self._stats_reset()
        self._best_elite = None

    @abstractmethod
    def index_of(self, measures_batch):
        """Returns archive indices for the given batch of measures.

        If you need to retrieve the index of the measures for a *single*
        solution, consider using :meth:`index_of_single`.

        Args:
            measures_batch (array-like): (batch_size, :attr:`measure_dim`)
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
            measures (array-like): (:attr:`measure_dim`,) array of measures for
                a single solution.
        Returns:
            int or numpy.integer: Integer index of the measures in the archive's
            storage arrays.
        Raises:
            ValueError: ``measures`` is not of shape (:attr:`measure_dim`,).
            ValueError: ``measures`` has non-finite values (inf or NaN).
        """
        measures = np.asarray(measures)
        check_1d_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")
        return self.index_of(measures[None])[0]

    def add(self,
            solution_batch,
            objective_batch,
            measures_batch,
            metadata_batch=None):
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
            other, i.e. ``solution_batch[i]``, ``objective_batch[i]``,
            ``measures_batch[i]``, and ``metadata_batch[i]`` should be the
            solution parameters, objective, measures, and metadata for solution
            ``i``.

        Args:
            solution_batch (array-like): (batch_size, :attr:`solution_dim`)
                array of solution parameters.
            objective_batch (array-like): (batch_size,) array with objective
                function evaluations of the solutions.
            measures_batch (array-like): (batch_size, :attr:`measure_dim`)
                array with measure space coordinates of all the solutions.
            metadata_batch (array-like): (batch_size,) array of Python objects
                representing metadata for the solution. For instance, this could
                be a dict with several properties.

                .. warning:: Due to how NumPy's :func:`~numpy.asarray`
                    automatically converts array-like objects to arrays, passing
                    array-like objects as metadata may lead to unexpected
                    behavior. However, the metadata may be a dict or other
                    object which *contains* arrays, i.e. ``metadata_batch``
                    could be an array of dicts which contain arrays.
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
              status_batch]``.

            - **value_batch** (:attr:`dtype`): An array with values for each
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
            ValueError: ``objective_batch`` or ``measures_batch`` has non-finite
                values (inf or NaN).
        """
        self._state["add"] += 1

        ## Step 0: Preprocess input. ##
        solution_batch = np.asarray(solution_batch)
        objective_batch = np.asarray(objective_batch)
        measures_batch = np.asarray(measures_batch)
        batch_size = solution_batch.shape[0]
        metadata_batch = (np.empty(batch_size, dtype=object) if
                          metadata_batch is None else np.asarray(metadata_batch,
                                                                 dtype=object))

        ## Step 1: Validate input. ##
        validate_batch_args(
            archive=self,
            solution_batch=solution_batch,
            objective_batch=objective_batch,
            measures_batch=measures_batch,
            metadata_batch=metadata_batch,
        )

        ## Step 2: Compute status_batch and value_batch ##

        # Retrieve indices.
        index_batch = self.index_of(measures_batch)

        # Copy old objectives since we will be modifying the objectives storage.
        old_objective_batch = np.copy(self._objective_arr[index_batch])
        old_threshold_batch = np.copy(self._threshold_arr[index_batch])

        # Compute the statuses -- these are all boolean arrays of length
        # batch_size.
        already_occupied = self._occupied_arr[index_batch]
        # In the case where we want CMA-ME behavior, threshold_arr[index]
        # is -inf for new cells, which satisfies the condition for can_be_added.
        can_be_added = objective_batch > old_threshold_batch
        is_new = can_be_added & ~already_occupied
        improve_existing = can_be_added & already_occupied
        status_batch = np.zeros(batch_size, dtype=np.int32)
        status_batch[is_new] = 2
        status_batch[improve_existing] = 1

        # New solutions require special settings for old_objective and
        # old_threshold.
        old_objective_batch[is_new] = self.dtype(0)

        # If threshold_min is -inf, then we want CMA-ME behavior, which
        # will compute the improvement value of new solutions w.r.t zero.
        # Otherwise, we will compute w.r.t. threshold_min.
        old_threshold_batch[is_new] = (self.dtype(0) if self._threshold_min
                                       == -np.inf else self._threshold_min)
        value_batch = objective_batch - old_threshold_batch

        ## Step 3: Insert solutions into archive. ##

        # Return early if we cannot insert anything -- continuing would actually
        # throw a ValueError in aggregate() since index_batch[can_insert] would
        # be empty.
        can_insert = is_new | improve_existing
        if not np.any(can_insert):
            return status_batch, value_batch

        # Select only solutions that can be inserted into the archive.
        solution_batch_can = solution_batch[can_insert]
        objective_batch_can = objective_batch[can_insert]
        measures_batch_can = measures_batch[can_insert]
        index_batch_can = index_batch[can_insert]
        metadata_batch_can = metadata_batch[can_insert]
        old_objective_batch_can = old_objective_batch[can_insert]

        # Retrieve indices of solutions that should be inserted into the
        # archive. Currently, multiple solutions may be inserted at each
        # archive index, but we only want to insert the maximum among these
        # solutions. Thus, we obtain the argmax for each archive index.
        #
        # We use a fill_value of -1 to indicate archive indices which were not
        # covered in the batch. Note that the length of archive_argmax is only
        # max(index_batch[can_insert]), rather than the total number of grid
        # cells. However, this is okay because we only need the indices of the
        # solutions, which we store in should_insert.
        #
        # aggregate() always chooses the first item if there are ties, so the
        # first elite will be inserted if there is a tie. See their default
        # numpy implementation for more info:
        # https://github.com/ml31415/numpy-groupies/blob/master/numpy_groupies/aggregate_numpy.py#L107
        archive_argmax = aggregate(index_batch_can,
                                   objective_batch_can,
                                   func="argmax",
                                   fill_value=-1)
        should_insert = archive_argmax[archive_argmax != -1]

        # Select only solutions that will be inserted into the archive.
        solution_batch_insert = solution_batch_can[should_insert]
        objective_batch_insert = objective_batch_can[should_insert]
        measures_batch_insert = measures_batch_can[should_insert]
        index_batch_insert = index_batch_can[should_insert]
        metadata_batch_insert = metadata_batch_can[should_insert]
        old_objective_batch_insert = old_objective_batch_can[should_insert]

        # Set archive storage.
        self._objective_arr[index_batch_insert] = objective_batch_insert
        self._measures_arr[index_batch_insert] = measures_batch_insert
        self._solution_arr[index_batch_insert] = solution_batch_insert
        self._metadata_arr[index_batch_insert] = metadata_batch_insert
        self._occupied_arr[index_batch_insert] = True

        # Mark new indices as occupied.
        is_new_and_inserted = is_new[can_insert][should_insert]
        n_new = np.sum(is_new_and_inserted)
        self._occupied_indices[self._num_occupied:self._num_occupied +
                               n_new] = (
                                   index_batch_insert[is_new_and_inserted])
        self._num_occupied += n_new

        # Update the thresholds.
        if self._threshold_min == -np.inf:
            # Here we want regular archive behavior, so the thresholds
            # should just be the maximum objective.
            self._threshold_arr[index_batch_insert] = objective_batch_insert
        else:
            # Here we compute the batch threshold update described in the
            # appendix of Fontaine 2022 https://arxiv.org/abs/2205.10752
            # This computation is based on the mean objective of all
            # solutions in the batch that could have been inserted into
            # each cell. This method is separated out to facilitate
            # testing.
            (new_thresholds,
             update_thresholds_indices) = self._compute_new_thresholds(
                 self._threshold_arr, objective_batch_can, index_batch_can,
                 self._learning_rate)
            self._threshold_arr[update_thresholds_indices] = new_thresholds

        ## Step 4: Update archive stats. ##

        # Since we set the new solutions in the old objective batch to have
        # value 0.0, the objectives for new solutions are added in properly
        # here.
        self._objective_sum += np.sum(objective_batch_insert -
                                      old_objective_batch_insert)
        new_qd_score = (self._objective_sum -
                        self.dtype(len(self)) * self._qd_score_offset)
        max_idx = np.argmax(objective_batch_insert)
        max_obj_insert = objective_batch_insert[max_idx]

        if self._stats.obj_max is None or max_obj_insert > self._stats.obj_max:
            new_obj_max = max_obj_insert
            self._best_elite = Elite(
                readonly(np.copy(solution_batch_insert[max_idx])),
                objective_batch_insert[max_idx],
                readonly(np.copy(measures_batch_insert[max_idx])),
                index_batch_insert[max_idx],
                metadata_batch_insert[max_idx],
            )
        else:
            new_obj_max = self._stats.obj_max

        self._stats = ArchiveStats(
            num_elites=len(self),
            coverage=self.dtype(len(self) / self.cells),
            qd_score=new_qd_score,
            norm_qd_score=self.dtype(new_qd_score / self.cells),
            obj_max=new_obj_max,
            obj_mean=self._objective_sum / self.dtype(len(self)),
        )

        return status_batch, value_batch

    def add_single(self, solution, objective, measures, metadata=None):
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
            metadata (object): Python object representing metadata for the
                solution. For instance, this could be a dict with several
                properties.

                .. warning:: Due to how NumPy's :func:`~numpy.asarray`
                    automatically converts array-like objects to arrays, passing
                    array-like objects as metadata may lead to unexpected
                    behavior. However, the metadata may be a dict or other
                    object which *contains* arrays.
        Raises:
            ValueError: The array arguments do not match their specified shapes.
            ValueError: ``objective`` is non-finite (inf or NaN) or ``measures``
                has non-finite values.
        Returns:
            tuple: 2-element tuple of (status, value) describing the result of
            the add operation. Refer to :meth:`add` for the meaning of the
            status and value.
        """
        self._state["add"] += 1

        solution = np.asarray(solution)
        objective = self.dtype(objective)
        measures = np.asarray(measures)
        validate_single_args(
            self,
            solution=solution,
            objective=objective,
            measures=measures,
        )

        index = self.index_of_single(measures)

        # Only used for computing QD score.
        old_objective = self._objective_arr[index]

        # Used for computing improvement value.
        old_threshold = self._threshold_arr[index]

        # New solutions require special settings for old_objective and
        # old_threshold.
        was_occupied = self._occupied_arr[index]
        if not was_occupied:
            old_objective = self.dtype(0)
            # If threshold_min is -inf, then we want CMA-ME behavior, which will
            # compute the improvement value w.r.t. zero for new solutions.
            # Otherwise, we will compute w.r.t. threshold_min.
            old_threshold = (self.dtype(0) if self._threshold_min == -np.inf
                             else self._threshold_min)

        status = 0  # NOT_ADDED
        # In the case where we want CMA-ME behavior, threshold_arr[index]
        # is -inf for new cells, which satisfies this if condition.
        if self._threshold_arr[index] < objective:
            if was_occupied:
                status = 1  # IMPROVE_EXISTING
            else:
                # Set this index to be occupied.
                self._occupied_arr[index] = True
                self._occupied_indices[self._num_occupied] = index
                self._num_occupied += 1

                status = 2  # NEW

            # This calculation works in the case where threshold_min is -inf
            # because old_threshold will be set to 0.0 instead.
            self._threshold_arr[index] = (old_threshold *
                                          (1.0 - self._learning_rate) +
                                          objective * self._learning_rate)

            # Insert into the archive.
            self._objective_arr[index] = objective
            self._measures_arr[index] = measures
            self._solution_arr[index] = solution
            self._metadata_arr[index] = metadata

        if status:
            # Update archive stats.
            self._objective_sum += objective - old_objective
            new_qd_score = (self._objective_sum -
                            self.dtype(len(self)) * self._qd_score_offset)

            if self._stats.obj_max is None or objective > self._stats.obj_max:
                new_obj_max = objective
                self._best_elite = Elite(
                    readonly(np.copy(self._solution_arr[index])),
                    objective,
                    readonly(np.copy(self._measures_arr[index])),
                    index,
                    metadata,
                )
            else:
                new_obj_max = self._stats.obj_max

            self._stats = ArchiveStats(
                num_elites=len(self),
                coverage=self.dtype(len(self) / self.cells),
                qd_score=new_qd_score,
                norm_qd_score=self.dtype(new_qd_score / self.cells),
                obj_max=new_obj_max,
                obj_mean=self._objective_sum / self.dtype(len(self)),
            )

        return status, objective - old_threshold

    def retrieve(self, measures_batch):
        """Retrieves the elites with measures in the same cells as the measures
        specified.

        This method operates in batch, i.e. it takes in a batch of measures and
        outputs an :namedtuple:`EliteBatch`. Since :namedtuple:`EliteBatch` is a
        namedtuple, it can be unpacked::

            solution_batch, objective_batch, measures_batch, \\
                index_batch, metadata_batch = archive.retrieve(...)

        Or the fields may be accessed by name::

            elite_batch = archive.retrieve(...)
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
        consider using :meth:`retrieve_single`.

        Args:
            measures_batch (array-like): (batch_size, :attr:`measure_dim`)
                array of coordinates in measure space.
        Returns:
            EliteBatch: See above.
        Raises:
            ValueError: ``measures_batch`` is not of shape (batch_size,
                :attr:`measure_dim`).
            ValueError: ``measures_batch`` has non-finite values (inf or NaN).
        """
        measures_batch = np.asarray(measures_batch)
        check_batch_shape(measures_batch, "measures_batch", self.measure_dim,
                          "measure_dim")
        check_finite(measures_batch, "measures_batch")

        index_batch = self.index_of(measures_batch)
        occupied_batch = self._occupied_arr[index_batch]
        expanded_occupied_batch = occupied_batch[:, None]

        return EliteBatch(
            solution_batch=readonly(
                # For each occupied_batch[i], this np.where selects
                # self._solution_arr[index_batch][i] if occupied_batch[i] is
                # True. Otherwise, it uses the alternate value (a solution
                # array consisting of np.nan).
                np.where(
                    expanded_occupied_batch,
                    self._solution_arr[index_batch],
                    np.full(self._solution_dim, np.nan),
                )),
            objective_batch=readonly(
                np.where(
                    occupied_batch,
                    self._objective_arr[index_batch],
                    # Here the alternative is just a scalar np.nan.
                    np.nan,
                )),
            measures_batch=readonly(
                np.where(
                    expanded_occupied_batch,
                    self._measures_arr[index_batch],
                    # And here it is a measures array of np.nan.
                    np.full(self._measure_dim, np.nan),
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
                    self._metadata_arr[index_batch],
                    None,
                )),
        )

    def retrieve_single(self, measures):
        """Retrieves the elite with measures in the same cell as the measures
        specified.

        While :meth:`retrieve` takes in a *batch* of measures, this method takes
        in the measures for only *one* solution and returns a single
        :namedtuple:`Elite`.

        Args:
            measures (array-like): (:attr:`measure_dim`,) array of measures.
        Returns:
            If there is an elite with measures in the same cell as the measures
            specified, then this method returns an :namedtuple:`Elite` where all
            the fields hold the info of that elite. Otherwise, this method
            returns an :namedtuple:`Elite` filled with the same "empty" values
            described in :meth:`retrieve`.
        Raises:
            ValueError: ``measures`` is not of shape (:attr:`measure_dim`,).
            ValueError: ``measures`` has non-finite values (inf or NaN).
        """
        measures = np.asarray(measures)
        check_1d_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")

        elite_batch = self.retrieve(measures[None])
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
            readonly(self._solution_arr[selected_indices]),
            readonly(self._objective_arr[selected_indices]),
            readonly(self._measures_arr[selected_indices]),
            readonly(selected_indices),
            readonly(self._metadata_arr[selected_indices]),
        )

    def as_pandas(self, include_solutions=True, include_metadata=False):
        """Converts the archive into an :class:`ArchiveDataFrame` (a child class
        of :class:`pandas.DataFrame`).

        The implementation of this method in :class:`ArchiveBase` creates a
        dataframe consisting of:

        - 1 column of integers (``np.int32``) for the index, named ``index``.
          See :meth:`index_of` for more info.
        - :attr:`measure_dim` columns for the measures, named ``measure_0,
          measure_1, ...``
        - 1 column for the objectives, named ``objective``
        - :attr:`solution_dim` columns for the solution parameters, named
          ``solution_0, solution_1, ...``
        - 1 column for the metadata objects, named ``metadata``

        In short, the dataframe looks like this:

        +-------+------------+------+------------+-------------+-----+----------+
        | index | measure_0  | ...  | objective  | solution_0  | ... | metadata |
        +=======+============+======+============+=============+=====+==========+
        |       |            | ...  |            |             | ... |          |
        +-------+------------+------+------------+-------------+-----+----------+

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
        """  # pylint: disable = line-too-long
        data = OrderedDict()
        indices = self._occupied_indices[:self._num_occupied]

        # Copy indices so we do not overwrite.
        data["index"] = np.copy(indices)

        measures_batch = self._measures_arr[indices]
        for i in range(self._measure_dim):
            data[f"measure_{i}"] = measures_batch[:, i]

        data["objective"] = self._objective_arr[indices]

        if include_solutions:
            solutions = self._solution_arr[indices]
            for i in range(self._solution_dim):
                data[f"solution_{i}"] = solutions[:, i]

        if include_metadata:
            data["metadata"] = self._metadata_arr[indices]

        return ArchiveDataFrame(
            data,
            copy=False,  # Fancy indexing above already results in copying.
        )

    def cqd_score(self,
                  iterations,
                  target_points,
                  penalties,
                  obj_min,
                  obj_max,
                  dist_max=None,
                  dist_ord=None):
        """Computes the CQD score of the archive.

        The Continuous Quality Diversity (CQD) score was introduced in
        `Kent 2022 <https://dl.acm.org/doi/10.1145/3520304.3534018>`_.

        .. note:: This method by default assumes that the archive has an
            ``upper_bounds`` and ``lower_bounds`` property which delineate the
            bounds of the measure space, as is the case in
            :class:`~ribs.archives.GridArchive`,
            :class:`~ribs.archives.CVTArchive`, and
            :class:`~ribs.archives.SlidingBoundariesArchive`.  If this is not
            the case, ``dist_max`` must be passed in, and ``target_points`` must
            be an array of custom points.

        Args:
            iterations (int): Number of times to compute the CQD score. We
                return the mean CQD score across these iterations.
            target_points (int or array-like): Number of target points to
                generate, or an (iterations, n, measure_dim) array which
                lists n target points to list on each iteration. When an int is
                passed, the points are sampled uniformly within the bounds of
                the measure space.
            penalties (int or array-like): Number of penalty values over which
                to compute the score (the values are distributed evenly over the
                range [0,1]). Alternatively, this may be a 1D array which
                explicitly lists the penalty values. Known as :math:`\\theta` in
                Kent 2022.
            obj_min (float): Minimum objective value, used when normalizing the
                objectives.
            obj_max (float): Maximum objective value, used when normalizing the
                objectives.
            dist_max (float): Maximum distance between points in measure space.
                Defaults to the distance between the extremes of the measure
                space bounds (the type of distance is computed with the order
                specified by ``dist_ord``). Known as :math:`\\delta_{max}` in
                Kent 2022.
            dist_ord: Order of the norm to use for calculating measure space
                distance; this is passed to :func:`numpy.linalg.norm` as the
                ``ord`` argument. See :func:`numpy.linalg.norm` for possible
                values. The default is to use Euclidean distance (L2 norm).
        Returns:
            The mean CQD score obtained with ``iterations`` rounds of
            calculations.
        Raises:
            RuntimeError: The archive does not have the bounds properties
                mentioned above, and dist_max is not specified or the target
                points are not provided.
            ValueError: target_points or penalties is an array with the wrong
                shape.
        """
        if (not (hasattr(self, "upper_bounds") and
                 hasattr(self, "lower_bounds")) and
            (dist_max is None or np.isscalar(target_points))):
            raise RuntimeError(
                "When the archive does not have lower_bounds and "
                "upper_bounds properties, dist_max must be specified, "
                "and target_points must be an array")

        if np.isscalar(target_points):
            # pylint: disable = no-member
            target_points = self._rng.uniform(
                low=self.lower_bounds,
                high=self.upper_bounds,
                size=(iterations, target_points, self.measure_dim),
            )
        else:
            # Copy since we return this.
            target_points = np.copy(target_points)
            if (target_points.ndim != 3 or
                    target_points.shape[0] != iterations or
                    target_points.shape[2] != self.measure_dim):
                raise ValueError(
                    "Expected target_points to be a 3D array with "
                    f"shape ({iterations}, n, {self.measure_dim}) "
                    "(i.e. shape (iterations, n, measure_dim)) but it had "
                    f"shape {target_points.shape}")

        if dist_max is None:
            # pylint: disable = no-member
            dist_max = np.linalg.norm(self.upper_bounds - self.lower_bounds,
                                      ord=dist_ord)

        if np.isscalar(penalties):
            penalties = np.linspace(0, 1, penalties)
        else:
            penalties = np.copy(penalties)  # Copy since we return this.
            check_is_1d(penalties, "penalties")

        index_batch = self._occupied_indices[:self._num_occupied]
        measures_batch = self._measures_arr[index_batch]
        objective_batch = self._objective_arr[index_batch]

        norm_objectives = objective_batch / (obj_max - obj_min)

        scores = np.zeros(iterations)

        for itr in range(iterations):
            # Distance calculation -- start by taking the difference between
            # each measure i and all the target points.
            distances = measures_batch[:, None] - target_points[itr]

            # (len(archive), n_target_points) array of distances.
            distances = np.linalg.norm(distances, ord=dist_ord, axis=2)

            norm_distances = distances / dist_max

            for penalty in penalties:
                # Known as omega in Kent 2022 -- a (len(archive),
                # n_target_points) array.
                values = norm_objectives[:, None] - penalty * norm_distances

                # (n_target_points,) array.
                max_values_per_target = np.max(values, axis=0)

                scores[itr] += np.sum(max_values_per_target)

        return CQDScoreResult(
            iterations=iterations,
            mean=np.mean(scores),
            scores=scores,
            target_points=target_points,
            penalties=penalties,
            obj_min=obj_min,
            obj_max=obj_max,
            dist_max=dist_max,
            dist_ord=dist_ord,
        )
