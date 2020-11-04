"""Provides ArchiveBase."""

import numpy as np


class ArchiveBase:
    """Base class for archives; contains several useful methods.

    Args:
        n_dims (int): Number of dimensions in the archive's behavior space.
        objective_value_dim (array-like): The dimensions to use for the
            objective value.
        behavior_value_dim (array-like): The dimensions to use for the behavior
            value array.
        solution_dim (array-like): The dimensions to use for the solutions
            array.
        seed (float): Seed for the random number generator.
    Attributes:
        _n_dims (int): See ``n_dims``.
        _objective_values (np.ndarray): Float array storing the objective values
            of each solution, shape is ``objective_value_dim``.
        _behavior_value_dim (np.ndarray): Float array storing the behavior
            values of each solution, shape is ``behavior_value_dim``.
        _solutions (np.ndarray): Object array storing the solution. We use
            object because we do not now the shape of the solution in advance.
            Shape is ``solution_dim``.
        _rng (np.random.Generator): Random number generator, used in particular
            for generating random elites.
    """

    def __init__(self, n_dims, objective_value_dim, behavior_value_dim,
                 solution_dim, seed):
        """Initializes the components of the archive."""
        self._rng = np.random.default_rng(seed)

        self._n_dims = n_dims

        # Create components of the grid. We separate the components so that they
        # can each be efficiently represented as numpy arrays.
        self._objective_values = np.empty(objective_value_dim, dtype=float)
        # Stores an array of behavior values at each index.
        self._behavior_values = np.empty(behavior_value_dim, dtype=float)
        self._solutions = np.full(solution_dim, None, dtype=object)

        # Having a list of occupied indices allows us to efficiently choose
        # random elites.
        self._occupied_indices = []

    def _get_index(self, behavior_values):
        """Returns archive indices for the given behavior values.

        If the behavior values are outside the dimensions of the container, they
        are clipped.
        """
        raise NotImplementedError

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

        if (self._solutions[index] is None or
                self._objective_values[index] < objective_value):
            # Track this index if it has not been seen before -- important that
            # we do this before inserting the solution.
            if self._solutions[index] is None:
                self._occupied_indices.append(index)

            # Insert into the archive.
            self._objective_values[index] = objective_value
            self._behavior_values[index] = behavior_values
            self._solutions[index] = solution

            return True

        return False

    def is_2d(self):
        """Checks if the archive has 2D.

        This is useful when checking whether we can visualize the archive.

        Returns:
            bool: True if the archive is 2D, False otherwise.
        """
        return self._n_dims == 2

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

        random_idx = self._rng.integers(len(self._occupied_indices))
        index = self._occupied_indices[random_idx]
        return (
            self._solutions[index],
            self._objective_values[index],
            self._behavior_values[index],
        )
