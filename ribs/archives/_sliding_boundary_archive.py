"""Contains the SlidingBoundaryArchive."""

from collections import deque

import numba as nb
import numpy as np
from sortedcontainers import SortedList

from ribs.archives._archive_base import ArchiveBase, require_init

_EPSILON = 1e-9


class IndividualBuffer:
    """An internal class that stores relevant data to re-add after remapping.

    It buffers solutions, objective values, and sorted behavior values of each
    dimension. It will pop the oldest element if it is full while putting new
    elements.
    """

    def __init__(self, buffer_capacity, behavior_dim):
        self._buffer_capacity = buffer_capacity
        self._inds_dq = deque()
        self._bc_lists = [SortedList() for _ in range(behavior_dim)]
        self._iter_idx = 0

    def __iter__(self):
        """Return self as the iterator."""
        return self

    def __next__(self):
        """Return the next solution, objective value, and behavior values from
        the buffer."""
        if self._iter_idx >= self.size:
            self._iter_idx = 0
            raise StopIteration
        result = self._inds_dq[self._iter_idx]
        self._iter_idx += 1
        return result

    def add(self, solution, objective_value, behavior_values):
        """Put a new element.

        Pop the oldest if it is full.
        """
        if self.full():
            # Remove item from the deque.
            _, _, bc_deleted = self._inds_dq.popleft()
            # Remove bc from sorted lists.
            for i, bc in enumerate(bc_deleted):
                self._bc_lists[i].remove(bc)

        self._inds_dq.append((solution, objective_value, behavior_values))

        # Add bc to sorted lists.
        for i, bc in enumerate(behavior_values):
            self._bc_lists[i].add(bc)

    def full(self):
        """Whether buffer is full."""
        return len(self._inds_dq) >= self._buffer_capacity

    @property
    def sorted_behavior_values(self):
        """(behavior_dim, self.size) numpy.ndarray: Sorted behaviors of each
        dimension."""
        return np.array(self._bc_lists, dtype=np.float)

    @property
    def size(self):
        """Number of solutions stored in the buffer."""
        return len(self._inds_dq)

    @property
    def capacity(self):
        """Capacity of the buffer."""
        return self._buffer_capacity


class SlidingBoundaryArchive(ArchiveBase):
    """An archive with a fixed number of sliding boundaries on each dimension.

    This archive is the container described in the `Hearthstone Deck Space
    paper <https://arxiv.org/pdf/1904.10656.pdf>`_. Same as the :class:`~ribs.
    archives.GridArchive`, it can be visualized as an n-dimensional grid in the
    behavior space that is divided into a certain number of bins in each
    dimension. However, it places the boundaries at the percentage marks of
    the behavior characteristics along each dimension. At a certain frequency,
    the archive will remap the boundary in accordance with all of the solutions
    stored in the buffer.

    This archive attempts to enable the distribution of the space illuminated
    by the archive to more accurately match the true distribution of the
    behavior characteristics are not uniformly distributed.

    Args:
        dims (array-like): Number of bins in each dimension of the behavior
            space, e.g. ``[20, 30, 40]`` indicates there should be 3 dimensions
            with 20, 30, and 40 bins. (The number of dimensions is implicitly
            defined in the length of this argument).
        ranges (array-like of (float, float)): Upper and lower bound of each
            dimension of the behavior space, e.g. ``[(-1, 1), (-2, 2)]``
            indicates the first dimension should have bounds ``(-1, 1)``, and
            the second dimension should have bounds ``(-2, 2)``. ``ranges``
            should be the same length as ``dims``.
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

        # Sliding boundary specifics.
        self._remap_frequency = remap_frequency
        self._boundaries = np.full((self._behavior_dim, np.max(self._dims)),
                                   np.inf,
                                   dtype=self.dtype)

        # Create buffer.
        self._buffer = IndividualBuffer(buffer_capacity, self._behavior_dim)

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
        """list of numpy.ndarray: The dynamic boundaries of each dimension.

        The number of boundaries is determined by ``dims``. e.g. if ``dims`` is
        ``[20, 30, 40]``, ``boundaries`` is ``[b1, b2, b3]`` where ``b1``,
        ``b2``, and ``b3`` are arrays of size 20, 30, and 40 respectively. To
        access the j-th boundary of the i-th dimension, use
        ``boundaries[i][j]``.
        """
        return [bound[:dim] for bound, dim in zip(self._boundaries, self._dims)]

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
        index = SlidingBoundaryArchive._get_index_numba(behavior_values,
                                                        self.upper_bounds,
                                                        self.lower_bounds,
                                                        self._boundaries,
                                                        self._dims)
        return tuple(index)

    def _reset_archive(self):
        """Reset the archive.

        Only ``self._occupied_indices`` and ``self._occupied`` are reset because
        other members do not matter.
        """
        self._occupied_indices.clear()
        self._occupied.fill(False)

    @require_init
    def add(self, solution, objective_value, behavior_values):
        """Attempts to insert a new solution into the archive.

        This method will remap the archive once every ``self.remap_frequency``
        solutions are found by changing the boundaries of the archive to the
        percentage marks of the behavior values stored in the buffer and
        re-adding all of the solutions stored in the buffer.

        .. note:: Remapping will not just add solutions in the current archive,
            but **ALL** of the solutions stored in the buffer.

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

        if self._total_num_sol % self._remap_frequency == 1:
            status, value = self._re_map()
        else:
            status, value = ArchiveBase.add(self, solution, objective_value,
                                            behavior_values)
        return status, value

    @staticmethod
    @nb.jit(nopython=True)
    def _re_map_numba_helper(sorted_bc, buffer_size, boundaries, behavior_dim,
                             dims):
        """Numba helper for _re_map().

        See _re_map() for usage.
        """
        for i in range(behavior_dim):
            for j in range(dims[i]):
                sample_idx = int(j * buffer_size / dims[i])
                boundaries[i][j] = sorted_bc[i][sample_idx]

    def _re_map(self):
        """Remaps the archive.

        The boundaries will be relocated at the percentage marks of the
        solutions stored in the archive.

        Also re-adds all of the solutions in the buffer.

        Returns:
            tuple: The result of calling :meth:`ArchiveBase.add` on the last
            item in the buffer.
        """

        # Sort all behavior values along the axis of each bc.
        sorted_bc = self._buffer.sorted_behavior_values

        # Calculate new boundaries.
        SlidingBoundaryArchive._re_map_numba_helper(sorted_bc,
                                                    self._buffer.size,
                                                    self._boundaries,
                                                    self._behavior_dim,
                                                    self.dims)

        # Add all solutions to the new empty archive.
        self._reset_archive()
        status, value = None, None
        for solution, objective_value, behavior_value in self._buffer:
            status, value = ArchiveBase.add(self, solution, objective_value,
                                            behavior_value)
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
