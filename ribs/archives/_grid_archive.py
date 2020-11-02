"""Contains the GridArchive and corresponding GridArchiveConfig."""
import numpy as np
import pandas as pd

from ribs.archives._archive_base import ArchiveBase
from ribs.config import create_config


class GridArchiveConfig:
    """Configuration for the GridArchive.

    Args:
        seed (float or int): Value to seed the random number generator. Set to
            None to avoid any seeding. Default: None
    """

    def __init__(
        self,
        seed=None,
    ):
        self.seed = seed


class GridArchive(ArchiveBase):
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
        self.dims = np.array(dims)
        ArchiveBase.__init__(
            self,
            len(self.dims),  # n_dims
            self.dims,  # objective_value_dim
            (*self.dims, len(self.dims)),  # behavior_value_dim
            self.dims,  # solution_dim
            self.config.seed,
        )

        ranges = list(zip(*ranges))
        self.lower_bounds = np.array(ranges[0])
        self.upper_bounds = np.array(ranges[1])
        self.interval_size = self.upper_bounds - self.lower_bounds

    def _get_index(self, behavior_values):
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
            *[f"index-{i}" for i in range(self._n_dims)],
            *[f"behavior-{i}" for i in range(self._n_dims)],
            "objective",
            "solution",
        ]

        rows = []
        for index in self._occupied_indices:
            row = [
                *index,
                *self._behavior_values[index],
                self._objective_values[index],
                self._solutions[index],
            ]
            rows.append(row)

        return pd.DataFrame(rows, columns=column_titles)
