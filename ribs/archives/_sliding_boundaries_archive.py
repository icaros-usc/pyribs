"""Contains the SlidingBoundariesArchive."""

from collections import deque

import numpy as np
from sortedcontainers import SortedList

from ribs._utils import (
    check_batch_shape,
    check_finite,
    check_shape,
    validate_batch,
    validate_single,
)
from ribs.archives._archive_base import ArchiveBase
from ribs.archives._archive_stats import ArchiveStats
from ribs.archives._array_store import ArrayStore
from ribs.archives._grid_archive import GridArchive
from ribs.archives._utils import fill_sentinel_values, parse_dtype


class SolutionBuffer:
    """An internal class that stores relevant data to re-add after remapping.

    Maintains two data structures:
    - Queue storing the buffer_capacity last entries (solution + objective + measures).
      When new items are inserted, the oldest ones are popped.
    - Sorted lists with the sorted measures in each dimension. Measures are removed from
      these lists when they are removed from the queue.
    """

    def __init__(self, buffer_capacity, measure_dim):
        self._buffer_capacity = buffer_capacity
        self._queue = deque()
        self._measure_lists = [SortedList() for _ in range(measure_dim)]
        self._iter_idx = 0

    def __iter__(self):
        """Enables iterating over solutions stored in the buffer."""
        return self

    def __next__(self):
        """Returns the next entry in the buffer."""
        if self._iter_idx >= self.size:
            self._iter_idx = 0
            raise StopIteration
        result = self._queue[self._iter_idx]
        self._iter_idx += 1
        return result

    def add(self, data):
        """Inserts a new entry.

        Pops the oldest if it is full.
        """
        if self.full():
            # Remove item from the deque.
            deleted_data = self._queue.popleft()
            # Remove measures from sorted lists.
            for i, m in enumerate(deleted_data["measures"]):
                self._measure_lists[i].remove(m)

        self._queue.append(data)

        # Add measures to sorted lists.
        for i, m in enumerate(data["measures"]):
            self._measure_lists[i].add(m)

    def full(self):
        """Whether buffer is full."""
        return len(self._queue) >= self._buffer_capacity

    @property
    def sorted_measures(self):
        """(measure_dim, self.size) numpy.ndarray: Sorted measures of each dimension."""
        return np.array(self._measure_lists, dtype=np.float64)

    @property
    def size(self):
        """Number of solutions stored in the buffer."""
        return len(self._queue)

    @property
    def capacity(self):
        """Capacity of the buffer."""
        return self._buffer_capacity


class SlidingBoundariesArchive(ArchiveBase):
    # pylint: disable = too-many-public-methods
    """An archive with a fixed number of sliding boundaries in each dimension.

    This archive is the container described in `Fontaine 2019
    <https://arxiv.org/abs/1904.10656>`_. Just like the
    :class:`~ribs.archives.GridArchive`, it can be visualized as an n-dimensional grid
    in the measure space that is divided into a certain number of cells in each
    dimension. Internally, this archive stores a buffer with the ``buffer_capacity``
    most recent solutions and uses them to determine the boundaries of each dimension of
    the measure space. After every ``remap_frequency`` solutions are inserted, the
    archive remaps the boundaries based on the solutions in the buffer.

    Initially, the archive has no solutions, so it cannot automatically calculate the
    boundaries. Thus, until the first remap, this archive divides the measure space
    defined by ``ranges`` into equally-sized cells.

    Overall, this archive attempts to make the distribution of the space illuminated by
    the archive more accurately match the true distribution of the measures when they
    are not uniformly distributed.

    By default, this archive stores the following data fields: ``solution``,
    ``objective``, ``measures``, and ``index``. The integer ``index`` uniquely
    identifies each cell.

    Args:
        solution_dim (int or tuple of int): Dimensionality of the solution space. Scalar
            or multi-dimensional solution shapes are allowed by passing an empty tuple
            or tuple of integers, respectively.
        dims (array-like): Number of cells in each dimension of the measure space, e.g.
            ``[20, 30, 40]`` indicates there should be 3 dimensions with 20, 30, and 40
            cells. (The number of dimensions is implicitly defined in the length of this
            argument).
        ranges (array-like of (float, float)): `Initial` upper and lower bound of each
            dimension of the measure space, e.g. ``[(-1, 1), (-2, 2)]`` indicates the
            first dimension should have bounds :math:`[-1,1]` (inclusive), and the
            second dimension should have bounds :math:`[-2,2]` (inclusive). ``ranges``
            should be the same length as ``dims``.
        epsilon (float): Due to floating point precision errors, we add a small epsilon
            when computing the archive indices in the :meth:`index_of` method -- refer
            to the implementation `here
            <../_modules/ribs/archives/_sliding_boundaries_archive.html#SlidingBoundariesArchive.index_of>`_.
            Pass this parameter to configure that epsilon.
        qd_score_offset (float): Archives often contain negative objective values, and
            if the QD score were to be computed with these negative objectives, the
            algorithm would be penalized for adding new cells with negative objectives.
            Thus, a standard practice is to normalize all the objectives so that they
            are non-negative by introducing an offset. This QD score offset will be
            *subtracted* from all objectives in the archive, e.g., if your objectives go
            as low as -300, pass in -300 so that each objective will be transformed as
            ``objective - (-300)``.
        seed (int): Value to seed the random number generator. Set to None to avoid a
            fixed seed.
        dtype (str or data-type or dict): Data type of the solutions, objectives, and
            measures. This can be ``"f"`` / ``np.float32``, ``"d"`` / ``np.float64``, or
            a dict specifying separate dtypes, of the form ``{"solution": <dtype>,
            "objective": <dtype>, "measures": <dtype>}``.
        extra_fields (dict): Description of extra fields of data that is stored next to
            elite data like solutions and objectives. The description is a dict mapping
            from a field name (str) to a tuple of ``(shape, dtype)``. For instance,
            ``{"foo": ((), np.float32), "bar": ((10,), np.float32)}`` will create a
            "foo" field that contains scalar values and a "bar" field that contains 10D
            values. Note that field names must be valid Python identifiers, and names
            already used in the archive are not allowed.
        remap_frequency (int): Frequency of remapping. Archive will remap once after
            ``remap_frequency`` number of solutions has been found.
        buffer_capacity (int): Number of solutions to keep in the buffer. Solutions in
            the buffer will be reinserted into the archive after remapping.
    """

    def __init__(
        self,
        *,
        solution_dim,
        dims,
        ranges,
        epsilon=1e-6,
        qd_score_offset=0.0,
        seed=None,
        dtype=np.float64,
        extra_fields=None,
        remap_frequency=100,
        buffer_capacity=1000,
    ):
        self._rng = np.random.default_rng(seed)
        self._dims = np.array(dims)

        ArchiveBase.__init__(
            self,
            solution_dim=solution_dim,
            objective_dim=(),
            measure_dim=len(self._dims),
        )

        # Set up the ArrayStore, which is a data structure that stores all the elites'
        # data in arrays sharing a common index.
        extra_fields = extra_fields or {}
        reserved_fields = {"solution", "objective", "measures", "index"}
        if reserved_fields & extra_fields.keys():
            raise ValueError(
                "The following names are not allowed in "
                f"extra_fields: {reserved_fields}"
            )
        dtype = parse_dtype(dtype)
        self._store = ArrayStore(
            field_desc={
                "solution": (self.solution_dim, dtype["solution"]),
                "objective": ((), dtype["objective"]),
                "measures": (self.measure_dim, dtype["measures"]),
                **extra_fields,
            },
            capacity=np.prod(self._dims),
        )

        # Set up constant properties.
        if len(self._dims) != len(ranges):
            raise ValueError(
                f"dims (length {len(self._dims)}) and ranges "
                f"(length {len(ranges)}) must be the same length"
            )
        ranges = list(zip(*ranges))
        self._lower_bounds = np.array(ranges[0], dtype=self.dtypes["measures"])
        self._upper_bounds = np.array(ranges[1], dtype=self.dtypes["measures"])
        self._interval_size = self._upper_bounds - self._lower_bounds
        self._epsilon = self.dtypes["measures"](epsilon)
        self._qd_score_offset = self.dtypes["objective"](qd_score_offset)
        self._remap_frequency = remap_frequency

        # Initialize the boundaries -- allocate an extra entry in each row so we can put
        # the upper bound at the end.
        self._boundaries = np.full(
            (self.measure_dim, np.max(self._dims) + 1),
            np.nan,
            dtype=self.dtypes["measures"],
        )
        for i, (dim, lower_bound, upper_bound) in enumerate(
            zip(self._dims, self._lower_bounds, self._upper_bounds)
        ):
            self._boundaries[i, : dim + 1] = np.linspace(
                lower_bound, upper_bound, dim + 1
            )

        # Create buffer.
        self._buffer = SolutionBuffer(buffer_capacity, self.measure_dim)

        # Total number of solutions encountered.
        self._total_num_sol = 0

        # Set up statistics -- objective_sum is the sum of all objective values in the
        # archive; it is useful for computing qd_score and obj_mean.
        self._best_elite = None
        self._objective_sum = None
        self._stats = None
        self._stats_reset()

    ## Properties inherited from ArchiveBase ##

    @property
    def field_list(self):
        return self._store.field_list_with_index

    @property
    def dtypes(self):
        return self._store.dtypes_with_index

    @property
    def stats(self):
        return self._stats

    @property
    def empty(self):
        return len(self._store) == 0

    ## Properties that are not in ArchiveBase ##
    ## Roughly ordered by the parameter list in the constructor. ##

    @property
    def dims(self):
        """(measure_dim,) numpy.ndarray: Number of cells in each dimension."""
        return self._dims

    @property
    def cells(self):
        """int: Total number of cells in the archive."""
        return self._store.capacity

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
        """dtypes["measures"]: Epsilon for computing archive indices. Refer to the
        documentation for this class."""
        return self._epsilon

    @property
    def qd_score_offset(self):
        """float: The offset which is subtracted from objective values when computing
        the QD score."""
        return self._qd_score_offset

    @property
    def remap_frequency(self):
        """int: Frequency of remapping.

        The archive will remap once after ``remap_frequency`` number of solutions has
        been found.
        """
        return self._remap_frequency

    @property
    def buffer_capacity(self):
        """int: Maximum capacity of the buffer."""
        return self._buffer.capacity

    ## Boundaries property (updates over time) ##

    @property
    def boundaries(self):
        """list of numpy.ndarray: The dynamic boundaries of the cells in each dimension.

        Entry ``i`` in this list is an array that contains the boundaries of the cells
        in dimension ``i``. The array contains ``self.dims[i] + 1`` entries laid out
        like this::

            Archive cells:  | 0 | 1 |   ...   |    self.dims[i]    |
            boundaries[i]:  0   1   2   self.dims[i] - 1     self.dims[i]

        Thus, ``boundaries[i][j]`` and ``boundaries[i][j + 1]`` are the lower and upper
        bounds of cell ``j`` in dimension ``i``. To access the lower bounds of all the
        cells in dimension ``i``, use ``boundaries[i][:-1]``, and to access all the
        upper bounds, use ``boundaries[i][1:]``.
        """
        return [bound[: dim + 1] for bound, dim in zip(self._boundaries, self._dims)]

    ## dunder methods ##

    def __len__(self):
        return len(self._store)

    def __iter__(self):
        return iter(self._store)

    ## Utilities ##

    def _stats_reset(self):
        """Resets the archive stats."""
        self._best_elite = None
        self._objective_sum = self.dtypes["objective"](0.0)
        self._stats = ArchiveStats(
            num_elites=0,
            coverage=self.dtypes["objective"](0.0),
            qd_score=self.dtypes["objective"](0.0),
            norm_qd_score=self.dtypes["objective"](0.0),
            obj_max=None,
            obj_mean=None,
        )

    def _stats_update(self, new_objective_sum, new_best_index):
        """Updates statistics based on a new sum of objective values (new_objective_sum)
        and the index of a potential new best elite (new_best_index)."""
        _, new_best_elite = self._store.retrieve([new_best_index])
        new_best_elite = {k: v[0] for k, v in new_best_elite.items()}

        if (
            self._stats.obj_max is None
            or new_best_elite["objective"] > self._stats.obj_max
        ):
            self._best_elite = new_best_elite
            new_obj_max = new_best_elite["objective"]
        else:
            new_obj_max = self._stats.obj_max

        self._objective_sum = new_objective_sum
        new_qd_score = (
            self._objective_sum
            - self.dtypes["objective"](len(self)) * self._qd_score_offset
        )
        self._stats = ArchiveStats(
            num_elites=len(self),
            coverage=self.dtypes["objective"](len(self) / self.cells),
            qd_score=new_qd_score,
            norm_qd_score=self.dtypes["objective"](new_qd_score / self.cells),
            obj_max=new_obj_max,
            obj_mean=self.dtypes["objective"](self._objective_sum / len(self)),
        )

    def index_of(self, measures):
        """Returns archive indices for the given batch of measures.

        First, values are clipped to the bounds of the measure space. Then, the values
        are mapped to cells via a binary search along the boundaries in each dimension.

        At this point, we have "grid indices" -- indices of each measure in each
        dimension. Since indices returned by this method must be single integers (as
        opposed to a tuple of grid indices), we convert these grid indices into integer
        indices with :func:`numpy.ravel_multi_index` and return the result.

        It may be useful to have the original grid indices. Thus, we provide the
        :meth:`grid_to_int_index` and :meth:`int_to_grid_index` methods for converting
        between grid and integer indices.

        As an example, the grid indices can be used to access boundaries of a measure
        value's cell. For example, the following retrieves the lower and upper bounds of
        the cell along dimension 0::

            # Access only element 0 since this method operates in batch.
            idx = archive.int_to_grid_index(archive.index_of(...))[0]

            lower = archive.boundaries[0][idx[0]]
            upper = archive.boundaries[0][idx[0] + 1]

        See :attr:`boundaries` for more info.

        Args:
            measures (array-like): (batch_size, :attr:`measure_dim`) array of
                coordinates in measure space.
        Returns:
            numpy.ndarray: (batch_size,) array of integer indices representing the
            flattened grid coordinates.
        Raises:
            ValueError: ``measures`` is not of shape (batch_size, :attr:`measure_dim`).
        """
        measures = np.asarray(measures)
        check_batch_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")

        # Clip measures + epsilon to the range
        # [lower_bounds, upper_bounds - epsilon].
        measures = np.clip(
            measures + self._epsilon,
            self._lower_bounds,
            self._upper_bounds - self._epsilon,
        )

        idx_cols = []
        for boundary, dim, measures_col in zip(
            self._boundaries, self._dims, measures.T
        ):
            idx_col = np.searchsorted(boundary[:dim], measures_col)
            # The maximum index returned by searchsorted is `dim`, and since we subtract
            # 1, the max will be dim - 1 which is within the range of the archive
            # indices.
            idx_cols.append(np.maximum(0, idx_col - 1))

        # We cannot use `grid_to_int_index` since that takes in an array of indices, not
        # index columns.
        #
        # pylint seems to think that ravel_multi_index returns a list and thus has no
        # astype method.
        # pylint: disable = no-member
        return np.ravel_multi_index(idx_cols, self._dims).astype(np.int32)

    def index_of_single(self, measures):
        """Returns the index of the measures for one solution.

        See :meth:`index_of`.

        Args:
            measures (array-like): (:attr:`measure_dim`,) array of measures for a single
                solution.
        Returns:
            int or numpy.integer: Integer index of the measures in the archive's storage
            arrays.
        Raises:
            ValueError: ``measures`` is not of shape (:attr:`measure_dim`,).
            ValueError: ``measures`` has non-finite values (inf or NaN).
        """
        measures = np.asarray(measures)
        check_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")
        return self.index_of(measures[None])[0]

    # Copy these methods from GridArchive.
    int_to_grid_index = GridArchive.int_to_grid_index
    grid_to_int_index = GridArchive.grid_to_int_index

    def _remap(self):
        """Remaps the archive.

        The boundaries are relocated to the percentage marks of the distribution of
        solutions stored in the archive.

        Also re-adds all of the solutions in the buffer and the previous archive to the
        archive.

        Returns:
            tuple: The result of calling :meth:`ArchiveBase.add` on the last item in the
            buffer.
        """
        # Sort each measure along its dimension.
        sorted_measures = self._buffer.sorted_measures

        # Calculate new boundaries.
        for i in range(self.measure_dim):
            for j in range(self.dims[i]):
                sample_idx = int(j * self._buffer.size / self.dims[i])
                self._boundaries[i][j] = sorted_measures[i][sample_idx]
            # Set the upper bound to be the greatest BC.
            self._boundaries[i][self.dims[i]] = sorted_measures[i][-1]

        cur_data = self._store.data()

        # These fields are only computed by the archive.
        cur_data.pop("index")

        new_data_single = list(self._buffer)  # List of dicts.
        new_data = {name: None for name in new_data_single[0]}
        for name in new_data:
            new_data[name] = [d[name] for d in new_data_single]

        # The last solution must be added on its own so that we get an accurate status
        # and value to return to the user; hence we pop it from all the batches (note
        # that pop removes the last value and returns it).
        last_data = {name: arr.pop() for name, arr in new_data.items()}

        self.clear()

        final_data = {
            name: np.concatenate((cur_data[name], new_data[name])) for name in cur_data
        }

        for i in range(len(final_data["solution"])):
            self._basic_add_single({name: arr[i] for name, arr in final_data.items()})
        return self._basic_add_single(last_data)

    ## Methods for writing to the archive ##

    def add(self, solution, objective, measures, **fields):
        """Inserts a batch of solutions into the archive.

        .. note:: Unlike in other archives, this method is not truly batched; rather, it
            is implemented by calling :meth:`add_single` on the solutions in the batch,
            in the order that they are passed in. As such, this method is *not*
            invariant to the ordering of the solutions in the batch.

        See :meth:`~add_single` and :meth:`ribs.archives.GridArchive.add` for arguments
        and return values.
        """
        new_data = validate_batch(
            self,
            {
                "solution": solution,
                "objective": objective,
                "measures": measures,
                **fields,
            },
        )
        batch_size = new_data["solution"].shape[0]

        add_info = {}
        for i in range(batch_size):
            single_info = self.add_single(
                **{name: arr[i] for name, arr in new_data.items()}
            )

            if i == 0:
                # Initialize add_info.
                for name, val in single_info.items():
                    add_info[name] = np.empty(batch_size, dtype=val.dtype)
            for name, val in single_info.items():
                add_info[name][i] = val

        return add_info

    def _basic_add_single(self, data):
        """Regular addition following standard MAP-Elites procedures.

        `data` should be similar to the data created in `add_single`.
        """

        # Information to return about the addition.
        add_info = {}

        # Identify the archive cell.
        index = self.index_of_single(data["measures"])

        # Retrieve current data of the cell.
        cur_occupied, cur_data = self._store.retrieve([index])
        cur_occupied = cur_occupied[0]
        cur_objective = (
            cur_data["objective"][0] if cur_occupied else self.dtypes["objective"](0.0)
        )

        # Retrieve candidate objective.
        objective = data["objective"]

        # Set up status.
        add_info["status"] = np.int32(0)  # NOT_ADDED

        # Now we check whether a solution should be added to the archive. We use the
        # addition rule from MAP-Elites (Fig. 2 of Mouret 2015
        # https://arxiv.org/pdf/1504.04909.pdf).

        if not cur_occupied or (cur_occupied and objective > cur_objective):
            if cur_occupied:
                add_info["status"] = np.int32(1)  # IMPROVE_EXISTING
            else:
                add_info["status"] = np.int32(2)  # NEW

            # Insert elite into the store.
            self._store.add(
                index[None],
                {name: np.expand_dims(arr, axis=0) for name, arr in data.items()},
            )

            # Update stats.
            self._stats_update(self._objective_sum + objective - cur_objective, index)

        # Value is the improvement over the current objective (can be negative).
        add_info["value"] = objective - cur_objective

        return add_info

    def add_single(self, solution, objective, measures, **fields):
        """Inserts a single solution into the archive.

        This method remaps the archive after every :attr:`remap_frequency` solutions are
        added. Remapping involves changing the boundaries of the archive to the
        percentage marks of the measures stored in the buffer and re-adding all of the
        solutions stored in the buffer `and` the current archive.

        Args:
            solution (array-like): Parameters of the solution.
            objective (float): Objective function evaluation of the solution.
            measures (array-like): Coordinates in measure space of the solution.
            fields (keyword arguments): Additional data for the solution.

        Returns:
            dict: Information describing the result of the add operation. The dict
            contains ``status`` and ``value`` keys, exactly as in
            :meth:`ribs.archives.GridArchive.add`.

        Raises:
            ValueError: The array arguments do not match their specified shapes.
            ValueError: ``objective`` is non-finite (inf or NaN) or ``measures`` has
                non-finite values.
        """
        data = validate_single(
            self,
            {
                "solution": solution,
                "objective": objective,
                "measures": measures,
                **fields,
            },
        )

        # Delete these so that we only use the clean, validated data in `data`.
        del solution, objective, measures, fields

        # Copy since other methods like basic_add_single can modify it.
        self._buffer.add(data.copy())
        self._total_num_sol += 1

        if self._total_num_sol % self._remap_frequency == 0:
            add_info = self._remap()
            self._lower_bounds = np.array([bound[0] for bound in self._boundaries])
            self._upper_bounds = np.array(
                [bound[dim] for bound, dim in zip(self._boundaries, self._dims)]
            )
            self._interval_size = self._upper_bounds - self._lower_bounds
        else:
            add_info = self._basic_add_single(data)
        return add_info

    def clear(self):
        """Removes all elites in the archive."""
        self._store.clear()
        self._stats_reset()

    ## Methods for reading from the archive ##
    ## Refer to ArchiveBase for documentation of these methods. ##

    def retrieve(self, measures):
        measures = np.asarray(measures)
        check_batch_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")

        occupied, data = self._store.retrieve(self.index_of(measures))
        fill_sentinel_values(occupied, data)

        return occupied, data

    def retrieve_single(self, measures):
        measures = np.asarray(measures)
        check_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")

        occupied, data = self.retrieve(measures[None])

        return occupied[0], {field: arr[0] for field, arr in data.items()}

    def data(self, fields=None, return_type="dict"):
        return self._store.data(fields, return_type)

    def sample_elites(self, n):
        if self.empty:
            raise IndexError("No elements in archive.")

        random_indices = self._rng.integers(len(self._store), size=n)
        selected_indices = self._store.occupied_list[random_indices]
        _, elites = self._store.retrieve(selected_indices)
        return elites
