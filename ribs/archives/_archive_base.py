"""Provides ArchiveBase."""
from abc import ABC, abstractmethod

import numba as nb
import numpy as np


class RandomBuffer:
    """An internal class that stores a buffer of random numbers.

    Generating random indices in get_random_elite() takes a lot of time if done
    individually. As such, this class generates many random numbers at once and
    slowly dispenses them. Since the calls in get_random_elite() vary in their
    range, this class does not store random integers; it stores random floats in
    the range [0,1) that can be multiplied to get a number in the range [0, x).

    Args:
        seed (float or int): Seed for the random number generator. None
            (default) means no seed.
        buf_size (int): How many random floats to store at once in the buffer.
    """

    def __init__(self, seed=None, buf_size=10):
        assert buf_size > 0, "buf_size must be at least 1"

        self._rng = np.random.default_rng(seed)
        self._buf_size = buf_size
        self._buffer = self._rng.random(buf_size)
        self._buf_idx = 0

    def get(self, max_val):
        """Returns a random int in the range [0, max_val)."""
        val = int(self._buffer[self._buf_idx] * max_val)
        self._buf_idx += 1

        # Reset the buffer if necessary.
        if self._buf_idx >= self._buf_size:
            self._buf_idx = 0
            self._buffer = self._rng.random(self._buf_size)

        return val


class ArchiveBase(ABC):
    """Base class for archives.

    This class assumes that all archives will use a fixed-size container with
    cells that hold 1) information about whether the cell is initialized (bool),
    2) a solution (1D array), 3) objective function evaluation of the solution
    (float), and 4) behavior space coordinates of the solution (1D array). In
    this class, this is implemented with 4 separate numpy arrays with common
    dimensions. Using the `storage_dims` and `behavior_dim` arguments in
    :meth:`__init__` and the ``solution_dim`` argument in ``initialize``, these
    arrays are as follows:

    - ``_initialized`` (shape ``(*storage_dims)``)
    - ``_solutions`` (shape ``(*storage_dims, solution_dim)``)
    - ``_objective_values`` (shape ``(*storage_dims)``)
    - ``_behavior_values`` (shape ``(*storage_dims, behavior_dim)``)

    All of these arrays are accessed by a common index. If we have index ``i``,
    we can access its solution at ``_solutions[i]``, its behavior values at
    ``_behavior_values[i]``, etc.

    Thus, child classes must override the following methods:

    - :meth:`__init__`: child classes must invoke this class's :meth:`__init__`
      with the appropriate arguments
    - :meth:`_get_index`: this method returns an index into those arrays given
      the behavior values of a solution
    - :meth:`initialize`: since this method sets up the arrays described, child
      classes should invoke this in their own implementation -- however, child
      classes may not need to override this method at all
    - :meth:`as_pandas`: necessary for converting the archive into a Pandas
      DataFrame

    .. note:: Members beginning with an underscore are only intended to be
        accessed by child classes.

    Args:
        storage_dims (tuple of int): Primary dimensions of the archive storage.
            This is used to create numpy arrays for items such as objective
            values and behavior values.
        behavior_dim (int): The dimension of the behavior space.
        seed (float or int): Seed for the random number generator. None
            (default) means no seed.
    Attributes:
        _rng (np.random.Generator): Random number generator, used in particular
            for generating random elites.
        _storage_dims (tuple of int): See ``storage_dims`` arg.
        _behavior_dim (int): See ``behavior_dim`` arg.
        _solution_dim (int): Dimension of the solution space, passed in with
            :meth:`initialize`.
        _initialized (np.ndarray): Bool array storing whether each cell in the
            archive has been initialized. This attribute is None until
            :meth:`initialize` is called.
        _solutions (np.ndarray): Float array storing the solutions themselves.
            This attribute is None until :meth:`initialize` is called.
        _objective_values (np.ndarray): Float array storing the objective values
            of each solution. This attribute is None until :meth:`initialize` is
            called.
        _behavior_value_dim (np.ndarray): Float array storing the behavior
            values of each solution. This attribute is None until
            :meth:`initialize` is called.
        _occupied_indices (list of (int or tuple of int)): A list of indices
            that are occupied in the archive.
    """

    def __init__(self, storage_dims, behavior_dim, seed=None):
        # Intended to be accessed by child classes.
        self._rng = np.random.default_rng(seed)
        self._storage_dims = storage_dims
        self._behavior_dim = behavior_dim
        self._solution_dim = None
        self._initialized = None
        self._objective_values = None
        self._behavior_values = None
        self._solutions = None
        self._occupied_indices = []

        # Not intended to be accessed by children (and thus not mentioned in the
        # docstring).
        self._rand_buf = None
        self._seed = seed

    def initialize(self, solution_dim):
        """Initializes the archive by allocating storage space.

        Child classes should call this method in their implementation if they
        are overriding it.

        Args:
            solution_dim (int): The dimension of the solution space.
        """
        self._rand_buf = RandomBuffer(self._seed)
        self._solution_dim = solution_dim
        self._initialized = np.zeros(self._storage_dims, dtype=bool)
        self._objective_values = np.empty(self._storage_dims, dtype=float)
        self._behavior_values = np.empty(
            (*self._storage_dims, self._behavior_dim), dtype=float)
        self._solutions = np.empty((*self._storage_dims, solution_dim),
                                   dtype=float)

    @abstractmethod
    def _get_index(self, behavior_values):
        """Returns archive indices for the given behavior values.

        :meta public:
        """

    @staticmethod
    @nb.jit(locals={"already_initialized": nb.types.b1}, nopython=True)
    def _add_numba(new_index, new_solution, new_objective_value,
                   new_behavior_values, initialized, solutions,
                   objective_values, behavior_values):
        """Numba helper for inserting solutions into the archive.

        See add() for usage.

        Returns:
            was_inserted (bool): Whether the new values were inserted into the
                archive.
            already_initialized (bool): Whether the index was initialized prior
                to this call; i.e. this is True only if there was already an
                item at the index.
        """
        already_initialized = initialized[new_index]
        if (not already_initialized or
                objective_values[new_index] < new_objective_value):
            # Track this index if it has not been seen before -- important that
            # we do this before inserting the solution.
            if not already_initialized:
                initialized[new_index] = True

            # Insert into the archive.
            objective_values[new_index] = new_objective_value
            behavior_values[new_index] = new_behavior_values
            solutions[new_index] = new_solution

            return True, already_initialized

        return False, already_initialized

    def add(self, solution, objective_value, behavior_values):
        """Attempts to insert a new solution into the archive.

        The solution is only inserted if it has a higher objective_value than
        the solution previously in the corresponding bin.

        Args:
            solution (np.ndarray): Parameters for the solution.
            objective_value (float): Objective function evaluation of this
                solution.
            behavior_values (np.ndarray): Coordinates in behavior space of this
                solution.
        Returns:
            bool: Whether the value was inserted into the archive.
        """
        index = self._get_index(behavior_values)

        was_inserted, already_initialized = self._add_numba(
            index, solution, objective_value, behavior_values,
            self._initialized, self._solutions, self._objective_values,
            self._behavior_values)

        if was_inserted and not already_initialized:
            self._occupied_indices.append(index)

        return was_inserted

    def is_2d(self):
        """Checks if the archive is 2D.

        This is useful when checking whether we can visualize the archive.

        Returns:
            bool: True if the archive is 2D, False otherwise.
        """
        return self._behavior_dim == 2

    def is_empty(self):
        """Checks if the archive has no elements in it.

        Returns:
            bool: True if the archive is empty, False otherwise.
        """
        return not self._occupied_indices

    def get_random_elite(self):
        """Select an elite uniformly at random from one of the archive's bins.

        Returns:
            tuple: 3-element tuple containing:

                **solution** (*np.ndarray*): Parameters for the solution.

                **objective_value** (*float*): Objective function evaluation.

                **behavior_values** (*np.ndarray*): Behavior space coordinates.
        Raises:
            IndexError: The archive is empty.
        """
        if self.is_empty():
            raise IndexError("No elements in archive.")

        random_idx = self._rand_buf.get(len(self._occupied_indices))
        index = self._occupied_indices[random_idx]
        return (
            self._solutions[index],
            self._objective_values[index],
            self._behavior_values[index],
        )

    @abstractmethod
    def as_pandas(self):
        """Converts the archive into a Pandas dataframe."""
