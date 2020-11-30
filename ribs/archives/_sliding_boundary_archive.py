"""Contains the SlidingBoundaryArchive and corresponding
SlidingBoundaryArchiveConfig."""

from collections import deque
from queue import Queue
import numpy as np
import pandas as pd
from sortedcontainers import SortedList

from ribs.archives._archive_base import ArchiveBase

_EPSILON = 1e-9


class IndividualBuffer():
    """ A buffer that stores the buffered solutions, objectives, and sorted
    behavior values of each dimension. It pops the oldest element if it is
    full while putting new elements.
    """

    def __init__(self, buffer_capacity, behavior_dim):
        self._buffer_capacity = buffer_capacity
        self._solutions_q = deque()
        self._objectives_q = deque()
        self._behaviors_q = deque()
        self._bc_list = [SortedList() for _ in range(behavior_dim)]
        self._iter_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter_idx >= self.size:
            self._iter_idx = 0
            raise StopIteration()
        result = (self._solutions_q[self._iter_idx],
                  self._objectives_q[self._iter_idx],
                  self._behaviors_q[self._iter_idx])
        self._iter_idx += 1
        return result

    def add(self, solution, objective_value, behavior_values):
        """Put a new element. Pop the oldest if it is full."""
        if self.full():
            # remove from queue
            self._solutions_q.popleft()
            self._objectives_q.popleft()
            bc_deleted = self._behaviors_q.popleft()
            # remove bc from sorted lists
            for i, bc in enumerate(bc_deleted):
                self._bc_list[i].remove(bc)

        self._solutions_q.append(solution)
        self._objectives_q.append(objective_value)
        self._behaviors_q.append(behavior_values)

        # add bc to sorted lists
        for i, bc in enumerate(behavior_values):
            self._bc_list[i].add(bc)

        print(self._bc_list)

    def full(self):
        """Whether buffer is full."""
        return len(self._solutions_q) == self._buffer_capacity and \
               len(self._objectives_q) == self._buffer_capacity and \
               len(self._behaviors_q) == self._buffer_capacity

    @property
    def sorted_behavior_values(self):
        """list of SortedList: Sorted behaviors of each dimension"""
        return self._bc_list

    @property
    def size(self):
        """Number of solutions stored in the buffer"""
        return len(self._solutions_q)

    @property
    def capacity(self):
        """Capacity of the buffer."""
        return self._buffer_capacity

class SlidingBoundaryArchive(ArchiveBase):
    """An archive that divides each dimension into a fixed number of bins with
    sliding boundaries that are placed at percentage marks of the behavior
    characteristics

    This archive is the container described in the `Hearthstone Deck Space
    paper <https://arxiv.org/pdf/1904.10656.pdf>`_. Same as the GridArchive, it
    can be visualized as an n-dimensional grid in the behavior space that is
    divided into a certain number of bins in each dimension. However, it places
    the boundaries at the percentage marks of the behavior characteristics along
    each dimension. At a certain frequency, the archive will remap the boundary
    in accordance with all of the solutions stored in the buffer (Note: not
    only those already in the archive, but ALL of the solutions stored in the
    buffer).

    This archive attempts to enable the distribution of the space illuminated
    by the archive to more accurately matches the true distribution if the
    behavior characteristics are not uniformly distributed.

    Args:
        dims (array-like): Number of bins in each dimension of the behavior
            space, e.g. ``[20, 30, 40]`` indicates there should be 3 dimensions
            with 20, 30, and 40 bins. (The number of dimensions is implicitly
            defined in the length of this argument).
        ranges (array-like of (float, float)): Upper and lower bound of each
            dimension of the behavior space, e.g. ``[(-1, 1), (-2, 2)]``
            indicates the first dimension should have bounds ``(-1, 1)``, and
            the second dimension should have bounds ``(-2, 2)``.
        seed (float or int): Value to seed the random number generator. Set to
            None to avoid any seeding. Default: None
        remap_frequency (int): Frequency of remapping. Archive will remap once
            after ``remap_frequency`` number of solutions has been found.
            Default: 100
        buffer_capacity (int): Number of solutions to keep in the buffer.
            Solutions in the buffer will be reinserted into the archive after
            remapping. Default: 1000
    """

    def __init__(self,
                 dims,
                 ranges,
                 seed=None,
                 remap_frequency=100,
                 buffer_capacity=1000):
        self._dims = np.array(dims)
        self._behavior_dim = len(self._dims)

        ArchiveBase.__init__(
            self,
            storage_dims=tuple(self._dims),
            behavior_dim=self._behavior_dim,
            seed=seed,
        )

        ranges = list(zip(*ranges))
        self._lower_bounds = np.array(ranges[0])
        self._upper_bounds = np.array(ranges[1])
        self._interval_size = self._upper_bounds - self._lower_bounds

        # Sliding boundary specifics
        self._remap_frequency = remap_frequency
        self._boundaries = [
            np.full(self._dims[i], None, dtype=float)
            for i in range(self._behavior_dim)
        ]

        # create buffer
        self._buffer = IndividualBuffer(buffer_capacity, self._behavior_dim)
        self._total_num_sol = 0  # total number of solutions encountered

    @property
    def dims(self):
        """(behavior_dim,) np.ndarray: Number of bins in each dimension."""
        return self._dims

    @property
    def lower_bounds(self):
        """(behavior_dim,) np.ndarray: Lower bound of each dimension."""
        return self._lower_bounds

    @property
    def upper_bounds(self):
        """(behavior_dim,) np.ndarray: Upper bound of each dimension."""
        return self._upper_bounds

    @property
    def interval_size(self):
        """(behavior_dim,) np.ndarray: The size of each dim (upper_bounds -
        lower_bounds).
        """
        return self._interval_size

    @property
    def remap_frequency(self):
        """int: Frequency of remapping. Archive will remap once after
        ``remap_frequency`` number of solutions has been found.
        """
        return self._remap_frequency

    @property
    def boundaries(self):
        """list of np.ndarray: The dynamic boundaries of each dimension of the
        behavior space. The number of boundaries is determined by ``dims``.
        """
        return self._boundaries

    @property
    def buffer(self):
        """IndividualBuffer: Buffer of solutions, behavior values, and
        objective values of the archive.
        """
        return self._buffer

    def _get_index(self, behavior_values):
        """Index is determined based on sliding boundaries
        """
        behavior_values = np.clip(behavior_values + _EPSILON,
                                  self._lower_bounds,
                                  self._upper_bounds - _EPSILON)

        index = []
        for i, behavior_value in enumerate(behavior_values):
            idx = np.searchsorted(self._boundaries[i], behavior_value)
            index.append(np.max([0, idx - 1]))
        return tuple(index)

    def _reset_archive(self):
        """Reset the archive.

        Note that we do not have to reset ``self.
        behavior_values`` because it won't affect solution insertion
        """
        self._objective_values.fill(-np.inf)
        self._solutions.fill(0)
        self._occupied_indices.clear()
        self._initialized.fill(False)

    def add(self, solution, objective_value, behavior_values):
        """ Attempt to insert the solution into the archive. Remap the archive
        once every ``self._remap_frequency`` solutions are found.

        Remap: change the boundaries of the archive to the percentage marks of
        the behavior values stored in the archive. and re-add all of the
        solutions stored in the buffer.

        Note: remapping will not just add solutions in the current archive, but
        ALL of the solutions stored in the buffer.

        Args:
            solution (np.ndarray): Parameters for the solution.
            objective_value (float): Objective function evaluation of this
                solution.
            behavior_values (np.ndarray): Coordinates in behavior space of this
                solution.
        Returns:
            bool: Whether the value was inserted into the archive.
        """
        self._buffer.add(solution, objective_value, behavior_values)
        self._total_num_sol += 1

        if self._total_num_sol % self._remap_frequency == 1:
            self._re_map()
        else:
            ArchiveBase.add(self, solution, objective_value, behavior_values)

    def _re_map(self,):
        """Remap the archive so that the boundaries locate at the percentage
        marks of the solutions stored in the archive.

        Re-add all of the solutions in the buffer.
        """

        # sort all behavior values along the axis of each bc
        sorted_bc = self._buffer.sorted_behavior_values

        for i in range(self._behavior_dim):
            for j in range(self._dims[i]):
                sample_idx = int(j * self._buffer.size / self._dims[i])
                self._boundaries[i][j] = sorted_bc[i][sample_idx]

        # add all solutions to the new empty archive
        self._reset_archive()
        for solution, objective_value, behavior_value in self._buffer:
            ArchiveBase.add(self, solution, objective_value, behavior_value)

    def as_pandas(self):
        """Converts the archive into a Pandas dataframe.

        Returns:
            A dataframe where each row is an elite in the archive. The dataframe
            has ``n_dims`` columns called ``index-{i}`` for the archive index,
            ``n_dims`` columns called ``behavior-{i}`` for the behavior values,
            1 column for the objective function value called ``objective``, and
            1 column for solution objects called ``solution``.
        """
        column_titles = [
            *[f"index-{i}" for i in range(self._behavior_dim)],
            *[f"behavior-{i}" for i in range(self._behavior_dim)],
            "objective",
            *[f"solution-{i}" for i in range(self._solution_dim)],
        ]

        rows = []
        for index in self._occupied_indices:
            row = [
                *index,
                *self._behavior_values[index],
                self._objective_values[index],
                *self._solutions[index],
            ]
            rows.append(row)
        return pd.DataFrame(rows, columns=column_titles)
