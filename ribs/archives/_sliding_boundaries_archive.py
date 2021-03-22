"""Contains the SlidingBoundariesArchive."""

from collections import deque

import numba as nb
import numpy as np
from sortedcontainers import SortedList

from ribs.archives._archive_base import ArchiveBase, require_init

_EPSILON = 1e-6


class SolutionBuffer:
    """An internal class that stores relevant data to re-add after remapping.

    Maintains two data structures:
    - Queue storing the buffer_capacity last entries (solution + objective value
      + behavior values). When new items are inserted, the oldest ones are
      popped.
    - Sorted lists with the sorted behavior values in each dimension. Behavior
      values are removed from theses lists when they are removed from the queue.
    """

    def __init__(self, buffer_capacity, behavior_dim):
        self._buffer_capacity = buffer_capacity
        self._queue = deque()
        self._bc_lists = [SortedList() for _ in range(behavior_dim)]
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

    def add(self, solution, objective_value, behavior_values):
        """Inserts a new entry.

        Pops the oldest if it is full.
        """
        if self.full():
            # Remove item from the deque.
            _, _, bc_deleted = self._queue.popleft()
            # Remove bc from sorted lists.
            for i, bc in enumerate(bc_deleted):
                self._bc_lists[i].remove(bc)

        self._queue.append((solution, objective_value, behavior_values))

        # Add bc to sorted lists.
        for i, bc in enumerate(behavior_values):
            self._bc_lists[i].add(bc)

    def full(self):
        """Whether buffer is full."""
        return len(self._queue) >= self._buffer_capacity

    @property
    def sorted_behavior_values(self):
        """(behavior_dim, self.size) numpy.ndarray: Sorted behavior values of
        each dimension."""
        return np.array(self._bc_lists, dtype=np.float)

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
    n-dimensional grid in the behavior space that is divided into a certain
    number of bins in each dimension. Internally, this archive stores a buffer
    with the ``buffer_capacity`` most recent solutions and uses them to
    determine the boundaries of the behavior characteristics along each
    dimension. After every ``remap_frequency`` solutions are inserted, the
    archive remaps the boundaries based on the solutions in the buffer.

    Initially, the archive has no solutions, so it cannot automatically
    calculate the boundaries. Thus, until the first remap, this archive divides
    the behavior space defined by ``ranges`` into equally sized bins.

    Overall, this archive attempts to make the distribution of the space
    illuminated by the archive more accurately match the true distribution of
    the behavior characteristics when they are not uniformly distributed.

    Args:
        dims (array-like): Number of bins in each dimension of the behavior
            space, e.g. ``[20, 30, 40]`` indicates there should be 3 dimensions
            with 20, 30, and 40 bins. (The number of dimensions is implicitly
            defined in the length of this argument).
        ranges (array-like of (float, float)): `Initial` upper and lower bound
            of each dimension of the behavior space, e.g. ``[(-1, 1), (-2, 2)]``
            indicates the first dimension should have bounds :math:`[-1,1]`
            (inclusive), and the second dimension should have bounds
            :math:`[-2,2]` (inclusive). ``ranges`` should be the same length as
            ``dims``.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
        dtype (str or data-type): Data type of the solutions, objective values,
            and behavior values. We only support ``"f"`` / :class:`np.float32`
            and ``"d"`` / :class:`np.float64`.
        remap_frequency (int): Frequency of remapping. Archive will remap once
            after ``remap_frequency`` number of solutions has been found.
        buffer_capacity (int): Number of solutions to keep in the buffer.
            Solutions in the buffer will be reinserted into the archive after
            remapping.
    """

    def __init__(self,
                 dims,
                 ranges,
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
            storage_dims=tuple(self._dims),
            behavior_dim=len(self._dims),
            seed=seed,
            dtype=dtype,
        )

        ranges = list(zip(*ranges))
        self._lower_bounds = np.array(ranges[0], dtype=self.dtype)
        self._upper_bounds = np.array(ranges[1], dtype=self.dtype)
        self._interval_size = self._upper_bounds - self._lower_bounds

        # Specifics for sliding boundaries.
        self._remap_frequency = remap_frequency

        # Allocate an extra entry in each row so we can put the upper bound at
        # the end.
        self._boundaries = np.full((self._behavior_dim, np.max(self._dims) + 1),
                                   np.nan,
                                   dtype=self.dtype)

        # Initialize the boundaries.
        for i, (dim, lower_bound, upper_bound) in enumerate(
                zip(self._dims, self._lower_bounds, self._upper_bounds)):
            self._boundaries[i, :dim + 1] = np.linspace(lower_bound,
                                                        upper_bound, dim + 1)

        # Create buffer.
        self._buffer = SolutionBuffer(buffer_capacity, self._behavior_dim)

        # Total number of solutions encountered.
        self._total_num_sol = 0

    @property
    def dims(self):
        """(behavior_dim,) numpy.ndarray: Number of bins in each dimension."""
        return self._dims

    @property
    def lower_bounds(self):
        """(behavior_dim,) numpy.ndarray: Lower bound of each dimension."""
        return self._lower_bounds

    @property
    def upper_bounds(self):
        """(behavior_dim,) numpy.ndarray: Upper bound of each dimension."""
        return self._upper_bounds

    @property
    def interval_size(self):
        """(behavior_dim,) numpy.ndarray: The size of each dim (upper_bounds -
        lower_bounds)."""
        return self._interval_size

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
        """list of numpy.ndarray: The dynamic boundaries of the bins in each
        dimension.

        Entry ``i`` in this list is an array that contains the boundaries of the
        bins in dimension ``i``. The array contains ``self.dims[i] + 1`` entries
        laid out like this::

            Archive bins:   | 0 | 1 |   ...   |    self.dims[i]    |
            boundaries[i]:  0   1   2   self.dims[i] - 1     self.dims[i]

        Thus, ``boundaries[i][j]`` and ``boundaries[i][j + 1]`` are the lower
        and upper bounds of bin ``j`` in dimension ``i``. To access the lower
        bounds of all the bins in dimension ``i``, use ``boundaries[i][:-1]``,
        and to access all the upper bounds, use ``boundaries[i][1:]``.
        """
        return [
            bound[:dim + 1] for bound, dim in zip(self._boundaries, self._dims)
        ]

    @staticmethod
    @nb.jit(nopython=True)
    def _get_index_numba(behavior_values, upper_bounds, lower_bounds,
                         boundaries, dims):
        """Numba helper for _get_index().

        See _get_index() for usage.
        """
        behavior_values = np.minimum(
            np.maximum(behavior_values + _EPSILON, lower_bounds),
            upper_bounds - _EPSILON)
        index = []
        for i, behavior_value in enumerate(behavior_values):
            idx = np.searchsorted(boundaries[i][:dims[i]], behavior_value)
            index.append(max(0, idx - 1))
        return index

    def _get_index(self, behavior_values):
        """Index is determined based on sliding boundaries.

        :meta private:
        """
        index = SlidingBoundariesArchive._get_index_numba(
            behavior_values, self.upper_bounds, self.lower_bounds,
            self._boundaries, self._dims)
        return tuple(index)

    def _reset_archive(self):
        """Reset the archive.

        Only ``self._occupied_indices`` and ``self._occupied`` are reset because
        other members do not matter.
        """
        self._occupied_indices.clear()
        self._occupied.fill(False)

    @staticmethod
    @nb.jit(nopython=True)
    def _remap_numba_helper(sorted_bc, buffer_size, boundaries, behavior_dim,
                            dims):
        """Numba helper for _remap().

        See _remap() for usage.
        """
        for i in range(behavior_dim):
            for j in range(dims[i]):
                sample_idx = int(j * buffer_size / dims[i])
                boundaries[i][j] = sorted_bc[i][sample_idx]
            # Set the upper bound to be the greatest BC.
            boundaries[i][dims[i]] = sorted_bc[i][-1]

    def _remap(self):
        """Remaps the archive.

        The boundaries are relocated to the percentage marks of the distribution
        of solutions stored in the archive.

        Also re-adds all of the solutions to the archive.

        Returns:
            tuple: The result of calling :meth:`ArchiveBase.add` on the last
            item in the buffer.
        """
        # Sort all behavior values along the axis of each bc.
        sorted_bc = self._buffer.sorted_behavior_values

        # Calculate new boundaries.
        SlidingBoundariesArchive._remap_numba_helper(sorted_bc,
                                                     self._buffer.size,
                                                     self._boundaries,
                                                     self._behavior_dim,
                                                     self.dims)

        index_columns = tuple(map(list, zip(*self._occupied_indices)))
        old_sols = self._solutions[index_columns].copy()
        old_objs = self._objective_values[index_columns].copy()
        old_behs = self._behavior_values[index_columns].copy()

        self._reset_archive()
        for sol, obj, beh in zip(old_sols, old_objs, old_behs):
            # Add solutions from old archive.
            status, value = ArchiveBase.add(self, sol, obj, beh)
        for sol, obj, beh in self._buffer:
            # Add solutions from buffer.
            status, value = ArchiveBase.add(self, sol, obj, beh)
        return status, value

    @require_init
    def add(self, solution, objective_value, behavior_values):
        """Attempts to insert a new solution into the archive.

        This method remaps the archive after every ``self.remap_frequency``
        solutions are added. Remapping involves changing the boundaries of the
        archive to the percentage marks of the behavior values stored in the
        buffer and re-adding all of the solutions stored in the buffer `and` the
        current archive.

        Args:
            solution (array-like): See :meth:`ArchiveBase.add`
            objective_value (float): See :meth:`ArchiveBase.add`
            behavior_values (array-like): See :meth:`ArchiveBase.add`
        Returns:
            See :meth:`ArchiveBase.add`
        """
        solution = np.asarray(solution)
        behavior_values = np.asarray(behavior_values)

        self._buffer.add(solution, objective_value, behavior_values)
        self._total_num_sol += 1

        if self._total_num_sol % self._remap_frequency == 0:
            status, value = self._remap()
            self._lower_bounds = np.array(
                [bound[0] for bound in self._boundaries])
            self._upper_bounds = np.array([
                bound[dim] for bound, dim in zip(self._boundaries, self._dims)
            ])
        else:
            status, value = ArchiveBase.add(self, solution, objective_value,
                                            behavior_values)
        return status, value

    @require_init
    def as_pandas(self, include_solutions=True):
        """Converts the archive into a Pandas dataframe.

        Args:
            include_solutions (bool): Whether to include solution columns.
        Returns:
            pandas.DataFrame: A dataframe where each row is an elite in the
            archive. The dataframe has ``behavior_dim`` columns called
            ``index_{i}`` for the archive index, ``behavior_dim`` columns called
            ``behavior_{i}`` for the behavior values, 1 column for the objective
            function value called ``objective``, and ``solution_dim`` columns
            called ``solution_{i}`` for the solution values.
        """
        return ArchiveBase.as_pandas(self, include_solutions)
