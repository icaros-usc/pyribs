"""Contains the GridArchive and corresponding GridArchiveConfig."""
import numpy as np
import pandas as pd
from numba import jit

from ribs.archives._archive_base import ArchiveBase
from ribs.config import create_config

_EPSILON = 1e-9


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
        dims (array-like of int): Number of bins in each dimension of the
            behavior space, e.g. ``[20, 30, 40]`` indicates there should be 3
            dimensions with 20, 30, and 40 bins. (The number of dimensions is
            implicitly defined in the length of this argument).
        ranges (array-like of (float, float)): Upper and lower bound of each
            dimension of the behavior space, e.g. ``[(-1, 1), (-2, 2)]``
            indicates the first dimension should have bounds ``(-1, 1)``, and
            the second dimension should have bounds ``(-2, 2)``.
        config (None or dict or GridArchiveConfig): Configuration object. If
            None, a default GridArchiveConfig is constructed. A dict may also be
            passed in, in which case its arguments will be passed into
            GridArchiveConfig.
    """

    def __init__(self, dims, ranges, config=None):
        self._config = create_config(config, GridArchiveConfig)
        self._dims = np.array(dims)
        behavior_dim = len(self._dims)
        ArchiveBase.__init__(
            self,
            storage_dims=tuple(self._dims),
            behavior_dim=behavior_dim,
            seed=self._config.seed,
        )

        ranges = list(zip(*ranges))
        self._lower_bounds = np.array(ranges[0])
        self._upper_bounds = np.array(ranges[1])
        self._interval_size = self._upper_bounds - self._lower_bounds

    @property
    def config(self):
        """GridArchiveConfig: Configuration object."""
        return self._config

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
        lower_bounds)."""
        return self._interval_size

    @staticmethod
    @jit(nopython=True)
    def _get_index_numba(behavior_values, upper_bounds, lower_bounds,
                         interval_size, dims):
        """Numba helper for _get_index().

        See _get_index() for usage.
        """
        # Adding epsilon to behavior values accounts for floating point
        # precision errors from transforming behavior values. Subtracting
        # epsilon from upper bounds makes sure we do not have indices outside
        # the grid.
        behavior_values = np.minimum(
            np.maximum(behavior_values + _EPSILON, lower_bounds),
            upper_bounds - _EPSILON)

        return (behavior_values - lower_bounds) / interval_size * dims

    def _get_index(self, behavior_values):
        index = GridArchive._get_index_numba(behavior_values,
                                             self._upper_bounds,
                                             self._lower_bounds,
                                             self._interval_size, self._dims)
        return tuple(index.astype(int))

    def as_pandas(self):
        """Converts the archive into a Pandas dataframe.

        Returns:
            A dataframe where each row is an elite in the archive. The dataframe
            has ``behavior_dim`` columns called ``index-{i}`` for the archive
            index, ``behavior_dim`` columns called ``behavior-{i}`` for the
            behavior values, 1 column for the objective function value called
            ``objective``, and ``solution_dim`` columns called ``solution-{i}``
            for the solution values.
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
