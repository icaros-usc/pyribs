"""Contains the GridArchive."""
import numpy as np
from numba import jit

from ribs.archives._archive_base import ArchiveBase

_EPSILON = 1e-6


class GridArchive(ArchiveBase):
    """An archive that divides each dimension into uniformly-sized bins.

    This archive is the container described in `Mouret 2015
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
            indicates the first dimension should have bounds :math:`[-1,1]`
            (inclusive), and the second dimension should have bounds
            :math:`[-2,2]` (inclusive). ``ranges`` should be the same length as
            ``dims``.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
        dtype (str or data-type): Data type of the solutions, objective values,
            and behavior values. We only support ``"f"`` / :class:`np.float32`
            and ``"d"`` / :class:`np.float64`.
    Raises:
        ValueError: ``dims`` and ``ranges`` are not the same length.
    """

    def __init__(self, dims, ranges, seed=None, dtype=np.float64):
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

        self._boundaries = []
        for dim, lower_bound, upper_bound in zip(self._dims, self._lower_bounds,
                                                 self._upper_bounds):
            self._boundaries.append(
                np.linspace(lower_bound, upper_bound, dim + 1))

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
    def boundaries(self):
        """list of numpy.ndarray: The boundaries of the bins in each dimension.

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
        return self._boundaries

    @staticmethod
    @jit(nopython=True)
    def _get_index_numba(behavior_values, upper_bounds, lower_bounds,
                         interval_size, dims):
        """Numba helper for get_index().

        See get_index() for usage.
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

    def get_index(self, behavior_values):
        """Returns indices of the entry within the archive's grid.

        First, values are clipped to the bounds of the behavior space. Then, the
        values are mapped to bins; e.g. bin 5 along dimension 0 and bin 3 along
        dimension 1.

        The indices can be used to access boundaries of a behavior value's bin.
        For example, the following retrieves the lower and upper bounds of the
        bin along dimension 0::

            idx = archive.get_index(...)  # Other methods also return indices.
            lower = archive.boundaries[0][idx[0]]
            upper = archive.boundaries[0][idx[0] + 1]

        See :attr:`boundaries` for more info.

        Args:
            behavior_values (numpy.ndarray): (:attr:`behavior_dim`,) array of
                coordinates in behavior space.
        Returns:
            tuple of int: The grid indices.
        """
        index = GridArchive._get_index_numba(behavior_values,
                                             self._upper_bounds,
                                             self._lower_bounds,
                                             self._interval_size, self._dims)
        return tuple(index)
