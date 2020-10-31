"""Contains the GridArchive and corresponding GridArchiveConfig."""
from collections import namedtuple

import numpy as np
import pandas as pd

from ribs.archives._individual import Individual
from ribs.config import create_config

#: Configuration for the GridArchive.
#:
#: Attributes:
#:     seed (float or int): Value to seed the random number generator. Set to
#:         None to avoid any seeding. Default: None
GridArchiveConfig = namedtuple("GridArchiveConfig", [
    "seed",
])
GridArchiveConfig.__new__.__defaults__ = (None,)


class GridArchive:
    """An archive that divides each dimension into a fixed number of bins.

    This archive is the container described in the original MAP-Elites paper:
    https://arxiv.org/pdf/1504.04909.pdf. It can be visualized as an
    n-dimensional grid in the behavior space that is divided into a certain
    number of bins in each dimension. Each bin contains an elite, i.e. a
    solution that `maximizes` the objective function for the behavior values in
    that bin.

    Args:
        dims (array-like): Number of bins in each dimension of the behavior
            space, e.g. ``[20, 30, 40]`` indicates there should be 3 dimensions
            with 20, 30, and 40 bins. (The number of dimensions is implicitly
            defined in the length of this argument).
        ranges (array-like of (float, float)): Upper and lower bound of each
            dimension of the behavior space, e.g. ``[(-1, 1), (-2, 2)]``
            indicates the first dimension should have bounds ``(-1, 1)``, and
            the second dimension should have bounds ``(-2, 2)``.
        config (GridArchiveConfig): Configuration object. If None, a default
            GridArchiveConfig is constructed. A dict may also be passed in, in
            which case its arguments will be passed into GridArchiveConfig.
    Attributes:
        config (GridArchiveConfig): Configuration object.
        dims (np.ndarray): Number of bins in each dimension.
        lower_bounds (np.ndarray): Lower bound of each dimension.
        upper_bounds (np.ndarray): Upper bound of each dimension.
        interval_size (np.ndarray): The size of each dimension (``upper_bounds -
            lower_bounds``).
    """

    def __init__(self, dims, ranges, config=None):
        self.config = create_config(config, GridArchiveConfig)
        self._rng = np.random.default_rng(self.config.seed)

        self.dims = np.array(dims)
        ranges = list(zip(*ranges))
        self.lower_bounds = np.array(ranges[0])
        self.upper_bounds = np.array(ranges[1])
        self.interval_size = self.upper_bounds - self.lower_bounds

        # Create components of the grid. We separate the components so that they
        # each be efficiently represented as a numpy array.
        self._initialized = np.zeros(self.dims, dtype=bool)
        self._objective_values = np.empty(self.dims, dtype=float)
        # Stores an array of behavior values at each index.
        self._behavior_values = np.empty(list(self.dims) + [len(self.dims)],
                                         dtype=float)
        self._solutions = np.full(self.dims, None, dtype=object)

        # Having a list of occupied indices allows us to efficiently choose
        # random elites.
        self._occupied_indices = []

    def _get_index(self, behavior_values):
        """Returns a tuple of archive indices for the given behavior values.

        If the behavior values are outside the dimensions of the container, they
        are clipped.
        """
        # Adding epsilon to behavior values accounts for floating point
        # precision errors from transforming behavior values. Subtracting
        # epsilon from upper bounds makes sure we do not have indices outside
        # the grid.
        epsilon = 1e-9
        behavior_values = np.clip(behavior_values + epsilon, self.lower_bounds,
                                  self.upper_bounds - epsilon)
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

        if (not self._initialized[index] or
                self._objective_values[index] < objective_value):
            # Insert into the archive.
            self._objective_values[index] = objective_value
            self._behavior_values[index] = behavior_values
            self._solutions[index] = solution

            # Track this index if it has not been seen before.
            if not self._initialized[index]:
                self._initialized[index] = True
                self._occupied_indices.append(index)

            return True

        return False

    def is_empty(self):
        """Checks if the archive has no elements in it.

        Returns:
            True if the archive is empty, False otherwise.
        """
        return not self._occupied_indices

    def get_random_elite(self):
        """Select a random elite from one of the archive's bins.

        Returns:
            (Individual) An elite from the archive, chosen uniformly at random.
        Raises:
            IndexError: The archive is empty.
        """
        if len(self._occupied_indices) == 0:
            raise IndexError("No elements in archive.")

        random_idx = self._rng.integers(len(self._occupied_indices))
        index = self._occupied_indices[random_idx]
        return Individual(
            self._objective_values[index],
            self._behavior_values[index],
            self._solutions[index],
        )

    def as_pandas(self):
        """Converts the archive into a Pandas dataframe.

        Returns:
            A dataframe where each row is an elite in the archive. The dataframe
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
        for index in self._occupied_indices:
            row = list(index)
            row += list(self._behavior_values[index])
            row.append(self._objective_values[index])
            row.append(self._solutions[index])
            rows.append(row)

        return pd.DataFrame(rows, columns=column_titles)
