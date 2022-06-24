"""Contains the GridArchive."""
import numpy as np
from numba import jit

from ribs._utils import check_measures_batch_shape
from ribs.archives._archive_base import ArchiveBase

_EPSILON = 1e-6


class GridArchive(ArchiveBase):
    """An archive that divides each dimension into uniformly-sized cells.

    This archive is the container described in `Mouret 2015
    <https://arxiv.org/pdf/1504.04909.pdf>`_. It can be visualized as an
    n-dimensional grid in the behavior space that is divided into a certain
    number of cells in each dimension. Each cell contains an elite, i.e. a
    solution that `maximizes` the objective function for the behavior values in
    that cell.

    Args:
        solution_dim (int): Dimension of the solution space.
        dims (array-like of int): Number of cells in each dimension of the
            behavior space, e.g. ``[20, 30, 40]`` indicates there should be 3
            dimensions with 20, 30, and 40 cells. (The number of dimensions is
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

    def __init__(self, solution_dim, dims, ranges, seed=None, dtype=np.float64):
        self._dims = np.array(dims)
        if len(self._dims) != len(ranges):
            raise ValueError(f"dims (length {len(self._dims)}) and ranges "
                             f"(length {len(ranges)}) must be the same length")

        ArchiveBase.__init__(
            self,
            solution_dim=solution_dim,
            cells=np.product(self._dims),
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
        """(behavior_dim,) numpy.ndarray: Number of cells in each dimension."""
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
        """list of numpy.ndarray: The boundaries of the cells in each dimension.

        Entry ``i`` in this list is an array that contains the boundaries of the
        cells in dimension ``i``. The array contains ``self.dims[i] + 1``
        entries laid out like this::

            Archive cells:  | 0 | 1 |   ...   |    self.dims[i]    |
            boundaries[i]:  0   1   2   self.dims[i] - 1     self.dims[i]

        Thus, ``boundaries[i][j]`` and ``boundaries[i][j + 1]`` are the lower
        and upper bounds of cell ``j`` in dimension ``i``. To access the lower
        bounds of all the cells in dimension ``i``, use ``boundaries[i][:-1]``,
        and to access all the upper bounds, use ``boundaries[i][1:]``.
        """
        return self._boundaries

    @staticmethod
    @jit(nopython=True)
    def _index_of_numba(measures_batch, upper_bounds, lower_bounds,
                        interval_size, dims):
        """Numba helper for index_of().

        See index_of() for usage.
        """
        # Adding epsilon to measures accounts for floating point precision
        # errors from transforming measures. Subtracting epsilon from
        # upper_bounds ensures we do not have indices outside the grid.
        measures_batch = np.minimum(
            np.maximum(measures_batch + _EPSILON, lower_bounds),
            upper_bounds - _EPSILON)

        grid_indices_batch = (measures_batch -
                              lower_bounds) / interval_size * dims

        # Casting to int is necessary for rounding down since grid_indices_batch
        # is currently float.
        return grid_indices_batch.astype(np.int32)

    def index_of(self, measures_batch):
        """Returns archive indices for the given batch of measures.

        First, values are clipped to the bounds of the measure space. Then, the
        values are mapped to cells; e.g. cell 5 along dimension 0 and cell 3
        along dimension 1.

        At this point, we have "grid indices" -- indices of each measure in each
        dimension. Since indices returned by this method must be single integers
        (as opposed to a tuple of grid indices), we convert these grid indices
        into integer indices with :func:`numpy.ravel_multi_index` and return the
        result.

        It may be useful to have the original grid indices. Thus, we provide the
        :meth:`grid_to_int_index` and :meth:`int_to_grid_index` methods for
        converting between grid and integer indices.

        As an example, the grid indices can be used to access boundaries of a
        measure value's cell. For example, the following retrieves the lower
        and upper bounds of the cell along dimension 0::

            # Access only element 0 since this method operates in batch.
            idx = archive.int_to_grid_index(archive.index_of(...))[0]

            lower = archive.boundaries[0][idx[0]]
            upper = archive.boundaries[0][idx[0] + 1]

        See :attr:`boundaries` for more info.

        Args:
            measures_batch (array-like): (batch_size, :attr:`behavior_dim`)
                array of coordinates in measure space.
        Returns:
            numpy.ndarray: (batch_size,) array of integer indices representing
            the flattened grid coordinates.
        Raises:
            ValueError: ``measures_batch`` is not of shape (batch_size,
                :attr:`behavior_dim`).
        """
        measures_batch = np.asarray(measures_batch)
        check_measures_batch_shape(measures_batch, self.behavior_dim)

        return self.grid_to_int_index(
            self._index_of_numba(
                measures_batch,
                self._upper_bounds,
                self._lower_bounds,
                self._interval_size,
                self._dims,
            ))

    def grid_to_int_index(self, grid_index_batch):
        """Converts a batch of grid indices into a batch of integer indices.

        Refer to :meth:`index_of` for more info.

        Args:
            grid_index_batch (array-like): (batch_size, :attr:`behavior_dim`)
                array of indices in the archive grid.
        Returns:
            numpy.ndarray: (batch_size,) array of integer indices.
        Raises:
            ValueError: ``grid_index_batch`` is not of shape (batch_size,
                :attr:`behavior_dim`)
        """
        grid_index_batch = np.asarray(grid_index_batch)
        check_measures_batch_shape(grid_index_batch,
                                   self.behavior_dim,
                                   name="grid_index_batch")

        return np.ravel_multi_index(grid_index_batch.T,
                                    self._dims).astype(np.int32)

    def int_to_grid_index(self, int_index_batch):
        """Converts a batch of indices into indices in the archive's grid.

        Refer to :meth:`index_of` for more info.

        Args:
            int_index_batch (array-like): (batch_size,) array of integer
                indices such as those output by :meth:`index_of`.
        Returns:
            numpy.ndarray: (batch_size, :attr:`behavior_dim`) array of indices
            in the archive grid.
        Raises:
            ValueError: ``int_index_batch`` is not of shape (batch_size,).
        """
        int_index_batch = np.asarray(int_index_batch)
        if len(int_index_batch.shape) != 1:
            raise ValueError("Expected int_index_batch to be a 1D array "
                             f"but it had shape {int_index_batch.shape}")

        return np.asarray(np.unravel_index(
            int_index_batch,
            self._dims,
        )).T.astype(np.int32)
