"""Contains the GridArchive."""
import numpy as np

from ribs._utils import check_batch_shape, check_finite, check_is_1d
from ribs.archives._archive_base import ArchiveBase


class GridArchive(ArchiveBase):
    """An archive that divides each dimension into uniformly-sized cells.

    This archive is the container described in `Mouret 2015
    <https://arxiv.org/pdf/1504.04909.pdf>`_. It can be visualized as an
    n-dimensional grid in the measure space that is divided into a certain
    number of cells in each dimension. Each cell contains an elite, i.e. a
    solution that `maximizes` the objective function for the measures in that
    cell.

    .. note:: The idea of archive thresholds was introduced in `Fontaine 2022
        <https://arxiv.org/abs/2205.10752>`_. Refer to our `CMA-MAE tutorial
        <../../tutorials/cma_mae.html>`_ for more info on thresholds, including
        the ``learning_rate`` and ``threshold_min`` parameters.

    Args:
        solution_dim (int): Dimension of the solution space.
        dims (array-like of int): Number of cells in each dimension of the
            measure space, e.g. ``[20, 30, 40]`` indicates there should be 3
            dimensions with 20, 30, and 40 cells. (The number of dimensions is
            implicitly defined in the length of this argument).
        ranges (array-like of (float, float)): Upper and lower bound of each
            dimension of the measure space, e.g. ``[(-1, 1), (-2, 2)]``
            indicates the first dimension should have bounds :math:`[-1,1]`
            (inclusive), and the second dimension should have bounds
            :math:`[-2,2]` (inclusive). ``ranges`` should be the same length as
            ``dims``.
        epsilon (float): Due to floating point precision errors, we add a small
            epsilon when computing the archive indices in the :meth:`index_of`
            method -- refer to the implementation `here
            <../_modules/ribs/archives/_grid_archive.html#GridArchive.index_of>`_.
            Pass this parameter to configure that epsilon.
        learning_rate (float): The learning rate for threshold updates.
        threshold_min (float): The initial threshold value for all the cells.
        qd_score_offset (float): Archives often contain negative objective
            values, and if the QD score were to be computed with these negative
            objectives, the algorithm would be penalized for adding new cells
            with negative objectives. Thus, a standard practice is to normalize
            all the objectives so that they are non-negative by introducing an
            offset. This QD score offset will be *subtracted* from all
            objectives in the archive, e.g., if your objectives go as low as
            -300, pass in -300 so that each objective will be transformed as
            ``objective - (-300)``.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
        dtype (str or data-type): Data type of the solutions, objectives,
            and measures. We only support ``"f"`` / ``np.float32`` and ``"d"`` /
            ``np.float64``.
    Raises:
        ValueError: ``dims`` and ``ranges`` are not the same length.
    """

    def __init__(self,
                 *,
                 solution_dim,
                 dims,
                 ranges,
                 learning_rate=1.0,
                 threshold_min=-np.inf,
                 epsilon=1e-6,
                 qd_score_offset=0.0,
                 seed=None,
                 dtype=np.float64):
        self._dims = np.array(dims, dtype=np.int32)
        if len(self._dims) != len(ranges):
            raise ValueError(f"dims (length {len(self._dims)}) and ranges "
                             f"(length {len(ranges)}) must be the same length")

        ArchiveBase.__init__(
            self,
            solution_dim=solution_dim,
            cells=np.product(self._dims),
            measure_dim=len(self._dims),
            learning_rate=learning_rate,
            threshold_min=threshold_min,
            qd_score_offset=qd_score_offset,
            seed=seed,
            dtype=dtype,
        )

        ranges = list(zip(*ranges))
        self._lower_bounds = np.array(ranges[0], dtype=self.dtype)
        self._upper_bounds = np.array(ranges[1], dtype=self.dtype)
        self._interval_size = self._upper_bounds - self._lower_bounds
        self._epsilon = self.dtype(epsilon)

        self._boundaries = []
        for dim, lower_bound, upper_bound in zip(self._dims, self._lower_bounds,
                                                 self._upper_bounds):
            self._boundaries.append(
                np.linspace(lower_bound, upper_bound, dim + 1))

    @property
    def dims(self):
        """(measure_dim,) numpy.ndarray: Number of cells in each dimension."""
        return self._dims

    @property
    def lower_bounds(self):
        """(measure_dim,) numpy.ndarray: Lower bound of each dimension."""
        return self._lower_bounds

    @property
    def upper_bounds(self):
        """(measure_dim,) numpy.ndarray: Upper bound of each dimension."""
        return self._upper_bounds

    @property
    def interval_size(self):
        """(measure_dim,) numpy.ndarray: The size of each dim (upper_bounds -
        lower_bounds)."""
        return self._interval_size

    @property
    def epsilon(self):
        """:attr:`dtype`: Epsilon for computing archive indices. Refer to
        the documentation for this class."""
        return self._epsilon

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
            measures_batch (array-like): (batch_size, :attr:`measure_dim`)
                array of coordinates in measure space.
        Returns:
            numpy.ndarray: (batch_size,) array of integer indices representing
            the flattened grid coordinates.
        Raises:
            ValueError: ``measures_batch`` is not of shape (batch_size,
                :attr:`measure_dim`).
            ValueError: ``measures_batch`` has non-finite values (inf or NaN).
        """
        measures_batch = np.asarray(measures_batch)
        check_batch_shape(measures_batch, "measures_batch", self.measure_dim,
                          "measure_dim")
        check_finite(measures_batch, "measures_batch")

        # Adding epsilon accounts for floating point precision errors from
        # transforming measures. We then cast to int32 to obtain integer
        # indices.
        grid_index_batch = (
            (self._dims *
             (measures_batch - self._lower_bounds) + self._epsilon) /
            self._interval_size).astype(np.int32)

        # Clip indices to the archive dimensions (for example, for 20 cells, we
        # want indices to run from 0 to 19).
        grid_index_batch = np.clip(grid_index_batch, 0, self._dims - 1)

        return self.grid_to_int_index(grid_index_batch)

    def grid_to_int_index(self, grid_index_batch):
        """Converts a batch of grid indices into a batch of integer indices.

        Refer to :meth:`index_of` for more info.

        Args:
            grid_index_batch (array-like): (batch_size, :attr:`measure_dim`)
                array of indices in the archive grid.
        Returns:
            numpy.ndarray: (batch_size,) array of integer indices.
        Raises:
            ValueError: ``grid_index_batch`` is not of shape (batch_size,
                :attr:`measure_dim`)
        """
        grid_index_batch = np.asarray(grid_index_batch)
        check_batch_shape(grid_index_batch, "grid_index_batch",
                          self.measure_dim, "measure_dim")

        return np.ravel_multi_index(grid_index_batch.T,
                                    self._dims).astype(np.int32)

    def int_to_grid_index(self, int_index_batch):
        """Converts a batch of indices into indices in the archive's grid.

        Refer to :meth:`index_of` for more info.

        Args:
            int_index_batch (array-like): (batch_size,) array of integer
                indices such as those output by :meth:`index_of`.
        Returns:
            numpy.ndarray: (batch_size, :attr:`measure_dim`) array of indices
            in the archive grid.
        Raises:
            ValueError: ``int_index_batch`` is not of shape (batch_size,).
        """
        int_index_batch = np.asarray(int_index_batch)
        check_is_1d(int_index_batch, "int_index_batch")

        return np.asarray(np.unravel_index(
            int_index_batch,
            self._dims,
        )).T.astype(np.int32)
