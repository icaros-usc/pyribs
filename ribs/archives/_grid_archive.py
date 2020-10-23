"""Contains the GridArchive class."""
from random import choice

import numpy as np
import pandas as pd


class GridArchive:
    """An archive that divides each dimension into a fixed number of bins.

    This archive is the container described in the original MAP-Elites paper:
    https://arxiv.org/pdf/1504.04909.pdf. It can be visualized as an
    n-dimensional grid in the behavior space that is divided into a certain
    number of bins in each dimension.

    Args:
        dims (array-like): Number of bins in each dimension of the behavior
            space, e.g. ``[20, 30, 40]`` indicates there should be 3 dimensions
            with 20, 30, and 40 bins. (The number of dimensions is implicitly
            defined in the length of this argument).
        ranges (array-like of array-like): Upper and lower bound of each
            dimension of the behavior space, e.g. ``[(-1, 1), (-2, 2)]``
            indicates the first dimension should have bounds ``(-1, 1)``, and
            the second dimension should have bounds ``(-2, 2)``.
    Attributes:
        dims (np.ndarray): Number of bins in each dimension.
        lower_bounds (np.ndarray): Lower bound of each dimension.
        upper_bounds (np.ndarray): Upper bound of each dimension.
        interval_size (np.ndarray): The size of each dimension (``upper_bounds -
            lower_bounds``).
    """

    def __init__(self, dims, ranges):
        self.dims = np.array(dims)
        self._grid = dict()

        ranges = list(zip(*ranges))
        self.lower_bounds = np.array(ranges[0])
        self.upper_bounds = np.array(ranges[1])
        self.interval_size = self.upper_bounds - self.lower_bounds

    def _get_index(self, behavior_values):
        """Returns a tuple of archive indices for the given behavior values.

        If the behavior values are outside the dimensions of the container, they
        are clipped.
        """
        # epsilon = 1e-9 accounts for floating point precision errors that
        # happen from transforming the behavior values into grid coordinates.
        behavior_values = np.clip(behavior_values + 1e-9, self.lower_bounds,
                                  self.upper_bounds)
        index = ((behavior_values - self.lower_bounds) \
                / self.interval_size) * self.dims
        return tuple(index.astype(int))

    def add(self, solution, objective_value, behavior_values):
        """Attempts to insert a new solution into the archive.

        The behavior values are first clipped to fit in the dimensions of the
        grid. Then, the behavior values are discretized to find the appropriate
        bin in the archive, and the solution is inserted if it has `higher`
        performance than the previous solution.

        Args:
            solution (np.ndarray):
            objective_value (float):
            behavior_values (np.ndarray):
        Returns:
            Whether the value was inserted into the archive.
        """
        index = self._get_index(behavior_values)

        if index not in self._grid or self._grid[index][0] < objective_value:
            self._grid[index] = (objective_value, behavior_values, solution)
            return True
        return False

    def get(self, index):
        """Returns the entry at the given index in the archive.

        Args:
            index (tuple): Tuple of bin indices in the archive.
        Returns:
            None if there is no entry, otherwise the elite in that bin.
        """
        # TODO: document what exactly is in this elite.
        return self._grid[index]

    def is_empty(self):
        """Checks if the archive has no elements in it.

        Returns:
            True if the archive is empty, False otherwise.
        """
        return not self._grid

    def get_random_elite(self):
        """Select a random elite from one of the archive's bins.

        Returns:
            An elite from the archive, chosen uniformly at random.
        Raises:
            IndexError: The archive is empty.
        """
        if len(self._grid) == 0:
            raise IndexError("No elements in archive.")

        index = choice(list(self._grid))
        # TODO: document what exactly is in this elite.
        return self._grid[index]

    def as_pandas(self):
        """Convert the archive into a Pandas dataframe.

        Returns:
            A dataframe where each row is an entry in the archive. The dataframe
            has `n_dims` columns called ``index-{i}`` for the archive index,
            `n_dims` columns called ``behavior-{i}`` for the behavior values, 1
            column for the objective function value called ``objective``, and 1
            column for solution objects called ``solution``.
        """
        num_dims = len(self.dims)
        column_titles = ['index-{}'.format(i) for i in range(num_dims)]
        column_titles += ['behavior-{}'.format(i) for i in range(num_dims)]
        column_titles += ['objective', 'solution']

        rows = []
        for index in self._grid:
            solution = self._grid[index]
            row = list(index)
            row += list(solution[1])
            row.append(solution[0])
            row.append(solution[2])
            rows.append(row)

        return pd.DataFrame(rows, columns=column_titles)
