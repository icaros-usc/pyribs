"""Contains the SlidingBoundariesArchive."""

from collections import deque

# TODO: Remove numba.
import numba as nb
import numpy as np
from sortedcontainers import SortedList

from ribs._utils import check_batch_shape
from ribs.archives._archive_base import ArchiveBase
from ribs.archives._grid_archive import GridArchive


class SolutionBuffer:
    """An internal class that stores relevant data to re-add after remapping.

    Maintains two data structures:
    - Queue storing the buffer_capacity last entries (solution + objective
      + measures + metadata). When new items are inserted, the oldest ones are
      popped.
    - Sorted lists with the sorted measures in each dimension. Measures are
      removed from theses lists when they are removed from the queue.
    """

    def __init__(self, buffer_capacity, measure_dim):
        self._buffer_capacity = buffer_capacity
        self._queue = deque()
        self._measures_lists = [SortedList() for _ in range(measure_dim)]
        self._iter_idx = 0

    def __iter__(self):
        """Enables iterating over solutions stored in the buffer."""
        return self

    def __next__(self):
        """Returns the next entry in the buffer."""
        if self._iter_idx >= self.size:
            self._iter_idx = 0
            raise StopIteration
        result = self._queue[self._iter_idx]
        self._iter_idx += 1
        return result

    def add(self, solution, objective, measures, metadata=None):
        """Inserts a new entry.

        Pops the oldest if it is full.
        """
        if self.full():
            # Remove item from the deque.
            _, _, measures_deleted, _ = self._queue.popleft()
            # Remove measures from sorted lists.
            for i, mea in enumerate(measures_deleted):
                self._measures_lists[i].remove(mea)

        self._queue.append((solution, objective, measures, metadata))

        # Add measures to sorted lists.
        for i, mea in enumerate(measures):
            self._measures_lists[i].add(mea)

    def full(self):
        """Whether buffer is full."""
        return len(self._queue) >= self._buffer_capacity

    @property
    def measures_sorted(self):
        """(measure_dim, self.size) numpy.ndarray: Sorted measures of each
        dimension."""
        return np.array(self._measures_lists, dtype=np.float64)

    @property
    def size(self):
        """Number of solutions stored in the buffer."""
        return len(self._queue)

    @property
    def capacity(self):
        """Capacity of the buffer."""
        return self._buffer_capacity


class SlidingBoundariesArchive(ArchiveBase):
    """An archive with a fixed number of sliding boundaries on each dimension.

    This archive is the container described in `Fontaine 2019
    <https://arxiv.org/abs/1904.10656>`_. Just like the
    :class:`~ribs.archives.GridArchive`, it can be visualized as an
    n-dimensional grid in the measure space that is divided into a certain
    number of cells in each dimension. Internally, this archive stores a buffer
    with the ``buffer_capacity`` most recent solutions and uses them to
    determine the boundaries of the measures along each dimension. After every
    ``remap_frequency`` solutions are inserted, the archive remaps the
    boundaries based on the solutions in the buffer.

    Initially, the archive has no solutions, so it cannot automatically
    calculate the boundaries. Thus, until the first remap, this archive divides
    the measure space defined by ``ranges`` into equally sized cells.

    Overall, this archive attempts to make the distribution of the space
    illuminated by the archive more accurately match the true distribution of
    the measures when they are not uniformly distributed.

    Args:
        solution_dim (int): Dimension of the solution space.
        dims (array-like): Number of cells in each dimension of the measure
            space, e.g. ``[20, 30, 40]`` indicates there should be 3 dimensions
            with 20, 30, and 40 cells. (The number of dimensions is implicitly
            defined in the length of this argument).
        ranges (array-like of (float, float)): `Initial` upper and lower bound
            of each dimension of the measure space, e.g. ``[(-1, 1), (-2, 2)]``
            indicates the first dimension should have bounds :math:`[-1,1]`
            (inclusive), and the second dimension should have bounds
            :math:`[-2,2]` (inclusive). ``ranges`` should be the same length as
            ``dims``.
        epsilon (float): Due to floating point precision errors, we add a small
            epsilon when computing the archive indices in the :meth:`index_of`
            method -- refer to the implementation `here
            <../_modules/ribs/archives/_sliding_boundaries_archive.html#SlidingBoundariesArchive.index_of>`_.
            Pass this parameter to configure that epsilon.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
        dtype (str or data-type): Data type of the solutions, objectives,
            and measures. We only support ``"f"`` / ``np.float32`` and ``"d"`` /
            ``np.float64``.
        remap_frequency (int): Frequency of remapping. Archive will remap once
            after ``remap_frequency`` number of solutions has been found.
        buffer_capacity (int): Number of solutions to keep in the buffer.
            Solutions in the buffer will be reinserted into the archive after
            remapping.
    """

    def __init__(self,
                 solution_dim,
                 dims,
                 ranges,
                 epsilon=1e-6,
                 seed=None,
                 dtype=np.float64,
                 remap_frequency=100,
                 buffer_capacity=1000):
        self._dims = np.array(dims)
        if len(self._dims) != len(ranges):
            raise ValueError(f"dims (length {len(self._dims)}) and ranges "
                             f"(length {len(ranges)}) must be the same length")

        ArchiveBase.__init__(
            self,
            solution_dim=solution_dim,
            cells=np.product(self._dims),
            measure_dim=len(self._dims),
            seed=seed,
            dtype=dtype,
        )

        ranges = list(zip(*ranges))
        self._lower_bounds = np.array(ranges[0], dtype=self.dtype)
        self._upper_bounds = np.array(ranges[1], dtype=self.dtype)
        self._interval_size = self._upper_bounds - self._lower_bounds
        self._epsilon = self.dtype(epsilon)

        # Specifics for sliding boundaries.
        self._remap_frequency = remap_frequency

        # Allocate an extra entry in each row so we can put the upper bound at
        # the end.
        self._boundaries = np.full((self._measure_dim, np.max(self._dims) + 1),
                                   np.nan,
                                   dtype=self.dtype)

        # Initialize the boundaries.
        for i, (dim, lower_bound, upper_bound) in enumerate(
                zip(self._dims, self._lower_bounds, self._upper_bounds)):
            self._boundaries[i, :dim + 1] = np.linspace(lower_bound,
                                                        upper_bound, dim + 1)

        # Create buffer.
        self._buffer = SolutionBuffer(buffer_capacity, self._measure_dim)

        # Total number of solutions encountered.
        self._total_num_sol = 0

    @property
    def dims(self):
        """(measure_dim,) numpy.ndarray: Number of cells in each dimension."""
        return self._dims

    @property
    def lower_bounds(self):
        """(measure_dim,) numpy.ndarray: Lower bound of each dimension."""
        return self._lower_bounds

    @property
    def upper_bounds(self):
        """(measure_dim,) numpy.ndarray: Upper bound of each dimension."""
        return self._upper_bounds

    @property
    def interval_size(self):
        """(measure_dim,) numpy.ndarray: The size of each dim (upper_bounds -
        lower_bounds)."""
        return self._interval_size

    @property
    def epsilon(self):
        """:attr:`dtype`: Epsilon for computing archive indices. Refer to
        the documentation for this class."""
        return self._epsilon

    @property
    def remap_frequency(self):
        """int: Frequency of remapping. Archive will remap once after
        ``remap_frequency`` number of solutions has been found.
        """
        return self._remap_frequency

    @property
    def buffer_capacity(self):
        """int: Maximum capacity of the buffer."""
        return self._buffer.capacity

    @property
    def boundaries(self):
        """list of numpy.ndarray: The dynamic boundaries of the cells in each
        dimension.

        Entry ``i`` in this list is an array that contains the boundaries of the
        cells in dimension ``i``. The array contains ``self.dims[i] + 1``
        entries laid out like this::

            Archive cells:  | 0 | 1 |   ...   |    self.dims[i]    |
            boundaries[i]:  0   1   2   self.dims[i] - 1     self.dims[i]

        Thus, ``boundaries[i][j]`` and ``boundaries[i][j + 1]`` are the lower
        and upper bounds of cell ``j`` in dimension ``i``. To access the lower
        bounds of all the cells in dimension ``i``, use ``boundaries[i][:-1]``,
        and to access all the upper bounds, use ``boundaries[i][1:]``.
        """
        return [
            bound[:dim + 1] for bound, dim in zip(self._boundaries, self._dims)
        ]

    @staticmethod
    @nb.jit(nopython=True)
    def _index_of_numba(measures_batch, upper_bounds, lower_bounds, boundaries,
                        dims, epsilon):
        """Numba helper for index_of().

        See index_of() for usage.
        """
        # Clip measures_batch + epsilon to the range
        # [lower_bounds, upper_bounds - epsilon].
        measures_batch = np.minimum(
            np.maximum(measures_batch + epsilon, lower_bounds),
            upper_bounds - epsilon)

        idx_cols = []
        for boundary, dim, measures_col in zip(boundaries, dims,
                                               measures_batch.T):
            idx_col = np.searchsorted(boundary[:dim], measures_col)
            # The maximum index returned by searchsorted is `dim`, and since we
            # subtract 1, the max will be dim - 1 which is within the range of
            # the archive indices.
            idx_cols.append(np.maximum(0, idx_col - 1))
        return idx_cols

    def index_of(self, measures_batch):
        """Returns archive indices for the given batch of measures.

        First, values are clipped to the bounds of the measure space. Then, the
        values are mapped to cells via a binary search along the boundaries in
        each dimension.

        At this point, we have "grid indices" -- indices of each measure in each
        dimension. Since indices returned by this method must be single integers
        (as opposed to a tuple of grid indices), we convert these grid indices
        into integer indices with :func:`numpy.ravel_multi_index` and return the
        result.

        It may be useful to have the original grid indices. Thus, we provide the
        :meth:`grid_to_int_index` and :meth:`int_to_grid_index` methods for
        converting between grid and integer indices.

        As an example, the grid indices can be used to access boundaries of a
        measure value's cell. For example, the following retrieves the lower
        and upper bounds of the cell along dimension 0::

            # Access only element 0 since this method operates in batch.
            idx = archive.int_to_grid_index(archive.index_of(...))[0]

            lower = archive.boundaries[0][idx[0]]
            upper = archive.boundaries[0][idx[0] + 1]

        See :attr:`boundaries` for more info.

        Args:
            measures_batch (array-like): (batch_size, :attr:`measure_dim`)
                array of coordinates in measure space.
        Returns:
            numpy.ndarray: (batch_size,) array of integer indices representing
            the flattened grid coordinates.
        Raises:
            ValueError: ``measures_batch`` is not of shape (batch_size,
                :attr:`measure_dim`).
        """
        measures_batch = np.asarray(measures_batch)
        check_batch_shape(measures_batch, "measures_batch", self._measure_dim,
                          "measure_dim")

        index_cols = SlidingBoundariesArchive._index_of_numba(
            measures_batch,
            self._upper_bounds,
            self._lower_bounds,
            self._boundaries,
            self._dims,
            self._epsilon,
        )

        # We cannot use `grid_to_int_index` since that takes in an array of
        # indices, not index columns.
        #
        # pylint seems to think that ravel_multi_index returns a list and thus
        # has no astype method.
        # pylint: disable = no-member
        return np.ravel_multi_index(index_cols, self._dims).astype(np.int32)

    # Copy these methods from GridArchive.
    int_to_grid_index = GridArchive.int_to_grid_index
    grid_to_int_index = GridArchive.grid_to_int_index

    # TODO remove numba
    @staticmethod
    @nb.jit(nopython=True)
    def _remap_numba_helper(measures_sorted, buffer_size, boundaries,
                            measure_dim, dims):
        """Numba helper for _remap().

        See _remap() for usage.
        """
        for i in range(measure_dim):
            for j in range(dims[i]):
                sample_idx = int(j * buffer_size / dims[i])
                boundaries[i][j] = measures_sorted[i][sample_idx]
            # Set the upper bound to be the largest measures.
            boundaries[i][dims[i]] = measures_sorted[i][-1]

    def _remap(self):
        """Remaps the archive.

        The boundaries are relocated to the percentage marks of the distribution
        of solutions stored in the archive.

        Also re-adds all of the solutions to the archive.

        Returns:
            tuple: The result of calling :meth:`ArchiveBase.add` on the last
            item in the buffer.
        """
        # Sort all measures along the axis of each measures.
        measures_sorted = self._buffer.measures_sorted

        # Calculate new boundaries.
        SlidingBoundariesArchive._remap_numba_helper(measures_sorted,
                                                     self._buffer.size,
                                                     self._boundaries,
                                                     self._measure_dim,
                                                     self.dims)

        indices = self._occupied_indices[:self._num_occupied]
        old_sols = self._solution_batch[indices].copy()
        old_objs = self._objective_batch[indices].copy()
        old_meas = self._measure_batch[indices].copy()
        old_metas = self._metadata[indices].copy()

        self.clear()
        for sol, obj, beh, meta in zip(old_sols, old_objs, old_behs, old_metas):
            # Add solutions from old archive.
            status, value = ArchiveBase.add(self, sol, obj, beh, meta)
        for sol, obj, beh, meta in self._buffer:
            # Add solutions from buffer.
            status, value = ArchiveBase.add(self, sol, obj, beh, meta)
        return status, value

    # TODO: Update this method to take in batches.
    def add(self, solution, objective, measures, metadata=None):
        """Inserts a batch of solutions into the archive.

        This method remaps the archive after every :attr:`remap_frequency`
        solutions are added. Remapping involves changing the boundaries of the
        archive to the percentage marks of the measures stored in the buffer and
        re-adding all of the solutions stored in the buffer `and` the current
        archive.

        See :meth:`ArchiveBase.add` for arguments and return values. Note that
        return values are computed with respect to the *current* archive, i.e.
        before doing any remapping.
        """
        solution = np.asarray(solution)
        measures = np.asarray(measures)

        self._buffer.add(solution, objective, measures, metadata)
        self._total_num_sol += 1

        if self._total_num_sol % self._remap_frequency == 0:
            status, value = self._remap()
            self._lower_bounds = np.array(
                [bound[0] for bound in self._boundaries])
            self._upper_bounds = np.array([
                bound[dim] for bound, dim in zip(self._boundaries, self._dims)
            ])
        else:
            status, value = ArchiveBase.add(self, solution, objective, measures,
                                            metadata)
        return status, value
