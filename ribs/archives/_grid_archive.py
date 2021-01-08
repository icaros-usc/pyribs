"""Contains the GridArchive."""
import numpy as np
from numba import jit

from ribs.archives._archive_base import ArchiveBase, require_init

_EPSILON = 1e-9


class GridArchive(ArchiveBase):
    """Divides each dimension of behavior space into a fixed number of bins.

    This archive is the container described in the `original MAP-Elites paper
    <https://arxiv.org/pdf/1504.04909.pdf>`_. It can be visualized as an
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
        seed (int): Value to seed the random number generator. Set to None to
            avoid seeding.
    """

    def __init__(self, dims, ranges, seed=None):
        self._dims = np.array(dims)
        behavior_dim = len(self._dims)
        ArchiveBase.__init__(
            self,
            storage_dims=tuple(self._dims),
            behavior_dim=behavior_dim,
            seed=seed,
        )

        ranges = list(zip(*ranges))
        self._lower_bounds = np.array(ranges[0])
        self._upper_bounds = np.array(ranges[1])
        self._interval_size = self._upper_bounds - self._lower_bounds

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

        index = (behavior_values - lower_bounds) / interval_size * dims
        return index.astype(np.int32)

    def _get_index(self, behavior_values):
        """Retrieves grid indices. Clips behavior values to behavior bounds.

        :meta private:
        """
        index = GridArchive._get_index_numba(behavior_values,
                                             self._upper_bounds,
                                             self._lower_bounds,
                                             self._interval_size, self._dims)
        return tuple(index)

    @require_init
    def as_pandas(self, include_solutions=True):
        """Converts the archive into a Pandas dataframe.

        Args:
            include_solutions (bool): Whether to include solution columns.
        Returns:
            pandas.DataFrame: A dataframe where each row is an elite in the
            archive. The dataframe has ``behavior_dim`` columns called
            ``index-{i}`` for the archive index, ``behavior_dim`` columns called
            ``behavior-{i}`` for the behavior values, 1 column for the objective
            function value called ``objective``, and ``solution_dim`` columns
            called ``solution-{i}`` for the solution values.
        """
        return ArchiveBase.as_pandas(self, include_solutions)
