"""Contains the CategoricalArchive."""

import numpy as np
from numpy_groupies import aggregate_nb as aggregate

from ribs._utils import check_batch_shape, check_shape, validate_batch, validate_single
from ribs.archives._archive_base import ArchiveBase
from ribs.archives._archive_stats import ArchiveStats
from ribs.archives._array_store import ArrayStore
from ribs.archives._grid_archive import GridArchive
from ribs.archives._utils import (
    fill_sentinel_values,
    parse_dtype,
    validate_cma_mae_settings,
)


class CategoricalArchive(ArchiveBase):
    # pylint: disable = too-many-public-methods
    """An archive where each dimension is divided into categories.

    This archive is similar to a :class:`~ribs.archives.GridArchive`, except that each
    measure is a categorical variable. Just like GridArchive, it can be visualized as an
    n-dimensional grid in the measure space that is divided into cells along each
    dimension. Each cell contains an elite, i.e., a solution that *maximizes* the
    objective function and has measures that lie within that cell. This archive also
    implements the idea of *soft archives* that have *thresholds*, as introduced in
    `Fontaine 2023 <https://arxiv.org/abs/2205.10752>`_.

    By default, this archive stores the following data fields: ``solution``,
    ``objective``, ``measures``, ``threshold``, and ``index``. The ``threshold`` is the
    value that a solution's objective value must exceed to be inserted into a cell,
    while the integer ``index`` uniquely identifies each cell.

    Args:
        solution_dim (int or tuple of int): Dimensionality of the solution space. Scalar
            or multi-dimensional solution shapes are allowed by passing an empty tuple
            or tuple of integers, respectively.
        categories (list of list of any): The name of each category for each dimension
            of the measure space. The length of this list is the dimensionality of the
            measure space. An example is ``[["A", "B", "C"], ["One", "Two", "Three",
            "Four"]]``, which defines a 2D measure space where the first dimension has
            categories ``["A", "B", "C"]`` and the second has categories ``["One",
            "Two", "Three", "Four"]``. While any object can be used for the category
            name, strings are expected to be the typical use case.
        learning_rate (float): The learning rate for threshold updates. Defaults to 1.0.
        threshold_min (float): The initial threshold value for all the cells.
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
        dtype (str or data-type or dict): There are two options for this parameter.
            First, it can be just the data type of the solutions and objectives, with
            the measures defaulting to a dtype of ``object``. In this case, ``dtype``
            can be ``"f"`` / ``np.float32`` or ``"d"`` / ``np.float64``. Second,
            ``dtype`` can be a dict specifying separate dtypes, of the form
            ``{"solution": <dtype>, "objective": <dtype>, "measures": <dtype>}``.
        extra_fields (dict): Description of extra fields of data that is stored next to
            elite data like solutions and objectives. The description is a dict mapping
            from a field name (str) to a tuple of ``(shape, dtype)``. For instance,
            ``{"foo": ((), np.float32), "bar": ((10,), np.float32)}`` will create a
            "foo" field that contains scalar values and a "bar" field that contains 10D
            values. Note that field names must be valid Python identifiers, and names
            already used in the archive are not allowed.
    Raises:
        ValueError: Invalid values for learning_rate and threshold_min.
        ValueError: Invalid names in extra_fields.
    """

    def __init__(
        self,
        *,
        solution_dim,
        categories,
        learning_rate=None,
        threshold_min=-np.inf,
        qd_score_offset=0.0,
        seed=None,
        dtype=np.float64,
        extra_fields=None,
    ):
        self._rng = np.random.default_rng(seed)
        self._categories = [list(measure_dim) for measure_dim in categories]
        self._dims = np.array(
            [len(measure_dim) for measure_dim in categories], dtype=np.int32
        )

        ArchiveBase.__init__(
            self,
            solution_dim=solution_dim,
            objective_dim=(),
            measure_dim=len(self._categories),
        )

        # Set up the ArrayStore, which is a data structure that stores all the elites'
        # data in arrays sharing a common index.
        extra_fields = extra_fields or {}
        reserved_fields = {"solution", "objective", "measures", "threshold", "index"}
        if reserved_fields & extra_fields.keys():
            raise ValueError(
                "The following names are not allowed in "
                f"extra_fields: {reserved_fields}"
            )
        if not isinstance(dtype, dict):
            # Make measures default to `object` dtype.
            dtype = {
                "solution": dtype,
                "measures": object,
                "objective": dtype,
            }
        dtype = parse_dtype(dtype)
        self._store = ArrayStore(
            field_desc={
                "solution": (self.solution_dim, dtype["solution"]),
                "objective": ((), dtype["objective"]),
                "measures": (self.measure_dim, dtype["measures"]),
                # Must be same dtype as the objective since they share calculations.
                "threshold": ((), dtype["objective"]),
                **extra_fields,
            },
            capacity=np.prod(self._dims),
        )

        # Set up constant properties.
        self._category_to_idx = [
            # Map from the category names in each dimension to integer indices.
            dict(zip(measure_dim, range(len(measure_dim))))
            for measure_dim in categories
        ]
        self._learning_rate, self._threshold_min = validate_cma_mae_settings(
            learning_rate, threshold_min, self.dtypes["threshold"]
        )
        self._qd_score_offset = self.dtypes["objective"](qd_score_offset)

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
    def best_elite(self):
        """dict: The elite with the highest objective in the archive.

        None if there are no elites in the archive.

        .. note::
            If the archive is non-elitist (this occurs when using the archive with a
            learning rate which is not 1.0, as in CMA-MAE), then this best elite may no
            longer exist in the archive because it was replaced with an elite with a
            lower objective value. This can happen because in non-elitist archives, new
            solutions only need to exceed the *threshold* of the cell they are being
            inserted into, not the *objective* of the elite currently in the cell. See
            :pr:`314` for more info.

        .. note::
            The best elite will contain a "threshold" key. This threshold is the
            threshold of the best elite's cell after the best elite was inserted into
            the archive.
        """
        return self._best_elite

    @property
    def categories(self):
        """list of list of any: The categories in each dimension of the measure
        space."""
        return self._categories

    @property
    def dims(self):
        """(measure_dim,) numpy.ndarray: Number of cells in each dimension."""
        return self._dims

    @property
    def cells(self):
        """int: Total number of cells in the archive."""
        return self._store.capacity

    @property
    def learning_rate(self):
        """float: The learning rate for threshold updates."""
        return self._learning_rate

    @property
    def threshold_min(self):
        """float: The initial threshold value for all the cells."""
        return self._threshold_min

    @property
    def qd_score_offset(self):
        """float: The offset which is subtracted from objective values when computing
        the QD score."""
        return self._qd_score_offset

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

        This is by done by mapping from the category name to the cell indices, and then
        converting to integer indices with :meth:`grid_to_int_index`.

        Args:
            measures (array-like): (batch_size, :attr:`measure_dim`) array of
                coordinates in measure space.
        Returns:
            numpy.ndarray: (batch_size,) array of integer indices representing the
            flattened grid coordinates.
        Raises:
            ValueError: ``measures`` is not of shape (batch_size, :attr:`measure_dim`).
            ValueError: ``measures`` has non-finite values (inf or NaN).
        """
        measures = np.asarray(measures, dtype=self.dtypes["measures"])
        check_batch_shape(measures, "measures", self.measure_dim, "measure_dim")

        grid_indices = [
            [self._category_to_idx[i][m] for i, m in enumerate(measure)]
            for measure in measures
        ]

        return self.grid_to_int_index(grid_indices)

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
        measures = np.asarray(measures, dtype=self.dtypes["measures"])
        check_shape(measures, "measures", self.measure_dim, "measure_dim")
        return self.index_of(measures[None])[0]

    # Copy these methods from GridArchive.
    int_to_grid_index = GridArchive.int_to_grid_index
    grid_to_int_index = GridArchive.grid_to_int_index

    ## Methods for writing to the archive ##

    @staticmethod
    def _compute_thresholds(indices, objective, cur_threshold, learning_rate, dtype):
        """Computes new thresholds with the CMA-MAE batch threshold update rule.

        If entries in `indices` are duplicated, they receive the same threshold.
        """
        if len(indices) == 0:
            return np.array([], dtype=dtype)

        # Compute the number of objectives inserted into each cell. Note that we index
        # with `indices` to place the counts at all relevant indices. For instance, if
        # we had an array [1,2,3,1,5], we would end up with [2,1,1,2,1] (there are 2
        # 1's, 1 2, 1 3, 2 1's, and 1 5).
        #
        # All objective_sizes should be > 0 since we only retrieve counts for indices in
        # `indices`.
        objective_sizes = aggregate(indices, 1, func="len", fill_value=0)[indices]

        # Compute the sum of the objectives inserted into each cell -- again, we index
        # with `indices`.
        objective_sums = aggregate(indices, objective, func="sum", fill_value=np.nan)[
            indices
        ]

        # Update the threshold with the batch update rule from Fontaine 2023
        # (https://arxiv.org/abs/2205.10752).
        #
        # Unlike in single_entry_with_threshold, we do not need to worry about
        # cur_threshold having -np.inf here as a result of threshold_min being -np.inf.
        # This is because the case with threshold_min = -np.inf is handled separately
        # since we compute the new threshold based on the max objective in each cell in
        # that case.
        ratio = dtype(1.0 - learning_rate) ** objective_sizes
        new_threshold = ratio * cur_threshold + (objective_sums / objective_sizes) * (
            1 - ratio
        )

        return new_threshold

    def add(self, solution, objective, measures, **fields):
        """Inserts a batch of solutions into the archive.

        Each solution is only inserted if it has a higher ``objective`` than the
        threshold of the corresponding cell. For the default values of ``learning_rate``
        and ``threshold_min``, this threshold is simply the objective value of the elite
        previously in the cell. If multiple solutions in the batch end up in the same
        cell, we only insert the solution with the highest objective. If multiple
        solutions end up in the same cell and tie for the highest objective, we insert
        the solution that appears first in the batch.

        For the default values of ``learning_rate`` and ``threshold_min``, the threshold
        for each cell is updated by taking the maximum objective value among all the
        solutions that landed in the cell, resulting in the same behavior as in the
        vanilla MAP-Elites archive. However, for other settings, the threshold is
        updated with the batch update rule described in the appendix of `Fontaine 2023
        <https://arxiv.org/abs/2205.10752>`_.

        .. note:: The indices of all arguments should "correspond" to each other, i.e.,
            ``solution[i]``, ``objective[i]``, and ``measures[i]`` should be the
            solution parameters, objective, and measures for solution ``i``.

        Args:
            solution (array-like): (batch_size, :attr:`solution_dim`) array of solution
                parameters.
            objective (array-like): (batch_size,) array with objective function
                evaluations of the solutions.
            measures (array-like): (batch_size, :attr:`measure_dim`) array with measure
                space coordinates of all the solutions.
            fields (keyword arguments): Additional data for each solution. Each argument
                should be an array with batch_size as the first dimension.

        Returns:
            dict: Information describing the result of the add operation. The dict
            contains the following keys:

            - ``"status"`` (:class:`numpy.ndarray` of :class:`numpy.int32`): An array of
              integers that represent the "status" obtained when attempting to insert
              each solution in the batch. Each item has the following possible values:

              - ``0``: The solution was not added to the archive.
              - ``1``: The solution improved the objective value of a cell which was
                already in the archive.
              - ``2``: The solution discovered a new cell in the archive.

              All statuses (and values, below) are computed with respect to the
              *current* archive. For example, if two solutions both introduce the same
              new archive cell, then both will be marked with ``2``.

              The alternative is to depend on the order of the solutions in the batch --
              for example, if we have two solutions ``a`` and ``b`` that introduce the
              same new cell in the archive, ``a`` could be inserted first with status
              ``2``, and ``b`` could be inserted second with status ``1`` because it
              improves upon ``a``. However, our implementation does **not** do this.

              To convert statuses to a more semantic format, cast all statuses to
              :class:`AddStatus`, e.g., with ``[AddStatus(s) for s in
              add_info["status"]]``.

            - ``"value"`` (:class:`numpy.ndarray` of :attr:`dtypes` ["objective"]): An
              array with values for each solution in the batch. With the default values
              of ``learning_rate = 1.0`` and ``threshold_min = -np.inf``, the meaning of
              each value depends on the corresponding ``status`` and is identical to
              that in CMA-ME (`Fontaine 2020 <https://arxiv.org/abs/1912.02400>`_):

              - ``0`` (not added): The value is the "negative improvement," i.e., the
                objective of the solution passed in minus the objective of the elite
                still in the archive (this value is negative because the solution did
                not have a high enough objective to be added to the archive).
              - ``1`` (improve existing cell): The value is the "improvement," i.e., the
                objective of the solution passed in minus the objective of the elite
                previously in the archive.
              - ``2`` (new cell): The value is just the objective of the solution.

              In contrast, for other values of ``learning_rate`` and ``threshold_min``,
              each value is equivalent to the objective value of the solution minus the
              threshold of its corresponding cell in the archive.

        Raises:
            ValueError: The array arguments do not match their specified shapes.
            ValueError: ``objective`` or ``measures`` has non-finite values (inf or
                NaN).
        """
        data = validate_batch(
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

        # Information to return about the addition.
        add_info = {}

        # Retrieve indices of the archive cells.
        indices = self.index_of(data["measures"])
        batch_size = len(indices)

        # Retrieve current data and thresholds. Unoccupied cells default to
        # threshold_min.
        cur_occupied, cur_data = self._store.retrieve(indices)
        cur_threshold = cur_data["threshold"]
        cur_threshold[~cur_occupied] = self.threshold_min

        # Compute status -- arrays below are all boolean arrays of length batch_size.
        #
        # When we want CMA-ME behavior, the threshold defaults to -inf for new cells,
        # which satisfies the condition for can_insert.
        can_insert = data["objective"] > cur_threshold
        is_new = can_insert & ~cur_occupied
        improve_existing = can_insert & cur_occupied
        add_info["status"] = np.zeros(batch_size, dtype=np.int32)
        add_info["status"][is_new] = 2
        add_info["status"][improve_existing] = 1

        # If threshold_min is -inf, then we want CMA-ME behavior, which computes the
        # improvement value of new solutions w.r.t zero. Otherwise, we compute
        # improvement with respect to threshold_min.
        cur_threshold[is_new] = (
            self.dtypes["threshold"](0.0)
            if self.threshold_min == -np.inf
            else self.threshold_min
        )
        add_info["value"] = data["objective"] - cur_threshold

        # Return early if we cannot insert anything -- continuing throws a ValueError in
        # aggregate() since index[can_insert] would be empty.
        if not np.any(can_insert):
            return add_info

        # Select all solutions that _can_ be inserted -- at this point, there are still
        # conflicts in the insertions, e.g., multiple solutions can map to index 0.
        indices = indices[can_insert]
        data = {name: arr[can_insert] for name, arr in data.items()}
        cur_threshold = cur_threshold[can_insert]

        # Compute the new threshold associated with each entry.
        if self.threshold_min == -np.inf:
            # Regular archive behavior: thresholds are just the objectives.
            new_threshold = data["objective"]
        else:
            # Batch threshold update described in Fontaine 2023
            # (https://arxiv.org/abs/2205.10752). This computation is based on the mean
            # objective of all solutions in the batch that could have been inserted into
            # each cell.
            new_threshold = self._compute_thresholds(
                indices,
                data["objective"],
                cur_threshold,
                self.learning_rate,
                self.dtypes["threshold"],
            )

        # Retrieve indices of solutions that _should_ be inserted into the archive.
        # Currently, multiple solutions may be inserted at each archive index, but we
        # only want to insert the maximum among these solutions. Thus, we obtain the
        # argmax for each archive index.
        #
        # We use a fill_value of -1 to indicate archive indices that were not covered in
        # the batch. Note that the length of archive_argmax is only max(indices), rather
        # than the total number of grid cells. However, this is okay because we only
        # need the indices of the solutions, which we store in should_insert.
        #
        # aggregate() always chooses the first item if there are ties, so the first
        # elite will be inserted if there is a tie. See their default numpy
        # implementation for more info:
        # https://github.com/ml31415/numpy-groupies/blob/master/numpy_groupies/aggregate_numpy.py#L107
        archive_argmax = aggregate(
            indices, data["objective"], func="argmax", fill_value=-1
        )
        should_insert = archive_argmax[archive_argmax != -1]

        # Select only solutions that will be inserted into the archive.
        indices = indices[should_insert]
        data = {name: arr[should_insert] for name, arr in data.items()}
        data["threshold"] = new_threshold[should_insert]

        # Insert elites into the store.
        self._store.add(indices, data)

        # Compute statistics.
        cur_objective = cur_data["objective"]
        cur_objective[~cur_occupied] = 0.0
        cur_objective = cur_objective[can_insert][should_insert]
        objective_sum = self._objective_sum + np.sum(data["objective"] - cur_objective)
        best_index = indices[np.argmax(data["objective"])]
        self._stats_update(objective_sum, best_index)

        return add_info

    def add_single(self, solution, objective, measures, **fields):
        """Inserts a single solution into the archive.

        The solution is only inserted if it has a higher ``objective`` than the
        threshold of the corresponding cell. For the default values of ``learning_rate``
        and ``threshold_min``, this threshold is simply the objective value of the elite
        previously in the cell. The threshold is also updated if the solution was
        inserted.

        .. note::
            This method is provided as an easier-to-understand implementation
            that has less performance due to inserting only one solution at a
            time. For better performance, see :meth:`add`.

        Args:
            solution (array-like): Parameters of the solution.
            objective (float): Objective function evaluation of the solution.
            measures (array-like): Coordinates in measure space of the solution.
            fields (keyword arguments): Additional data for the solution.

        Returns:
            dict: Information describing the result of the add operation. The
            dict contains ``status`` and ``value`` keys; refer to :meth:`add`
            for the meaning of status and value.

        Raises:
            ValueError: The array arguments do not match their specified shapes.
            ValueError: ``objective`` is non-finite (inf or NaN) or ``measures``
                has non-finite values.
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

        # Information to return about the addition.
        add_info = {}

        # Identify the archive cell.
        index = self.index_of_single(data["measures"])

        # Retrieve current data of the cell.
        cur_occupied, cur_data = self._store.retrieve([index])
        cur_occupied = cur_occupied[0]

        if cur_occupied:
            # If the cell is currently occupied, the threshold comes from the current
            # data of the elite in the cell.
            cur_threshold = cur_data["threshold"][0]
        else:
            # If the cell is not currently occupied, the threshold needs special
            # settings.
            #
            # If threshold_min is -inf, then we want CMA-ME behavior, which computes the
            # improvement value with a threshold of zero for new solutions. Otherwise,
            # we will set cur_threshold to threshold_min.
            cur_threshold = (
                self.dtypes["threshold"](0.0)
                if self.threshold_min == -np.inf
                else self.threshold_min
            )

        # Retrieve candidate objective.
        objective = data["objective"]

        # Compute status and threshold.
        add_info["status"] = np.int32(0)  # NOT_ADDED

        # Now we check whether a solution should be added to the archive. We use the
        # addition rule from MAP-Elites (Fig. 2 of Mouret 2015
        # https://arxiv.org/pdf/1504.04909.pdf), with modifications for CMA-MAE.

        # This checks if a new solution is discovered in the archive. Note that regular
        # MAP-Elites only checks `not cur_occupied`. CMA-MAE has an additional
        # `threshold_min` that the objective must exceed for new cells. If CMA-MAE is
        # not being used, then `threshold_min` is -np.inf, making this check identical
        # to that of MAP-Elites.
        is_new = not cur_occupied and self.threshold_min < objective

        # This checks whether the solution improves an existing cell in the archive,
        # i.e., whether it performs better than the current elite in this cell. Vanilla
        # MAP-Elites compares to the objective of the cell's current elite. CMA-MAE
        # compares to a threshold value that updates over time (i.e., cur_threshold).
        # When learning_rate is set to 1.0 (the default value), we recover the same rule
        # as in MAP-Elites because cur_threshold is equivalent to the objective of the
        # solution in the cell.
        improve_existing = cur_occupied and cur_threshold < objective

        if is_new or improve_existing:
            if improve_existing:
                add_info["status"] = np.int32(1)  # IMPROVE_EXISTING
            else:
                add_info["status"] = np.int32(2)  # NEW

            # This calculation works in the case where threshold_min is -inf because
            # cur_threshold will be set to 0.0 instead.
            data["threshold"] = (
                cur_threshold * (1.0 - self.learning_rate)
                + objective * self.learning_rate
            )

            # Insert elite into the store.
            self._store.add(
                index[None],
                {name: np.expand_dims(arr, axis=0) for name, arr in data.items()},
            )

            # Update stats.
            cur_objective = (
                cur_data["objective"][0]
                if cur_occupied
                else self.dtypes["objective"](0.0)
            )
            self._stats_update(self._objective_sum + objective - cur_objective, index)

        # Value is the improvement over the current threshold (can be negative).
        add_info["value"] = objective - cur_threshold

        return add_info

    def clear(self):
        """Removes all elites in the archive."""
        self._store.clear()
        self._stats_reset()

    ## Methods for reading from the archive ##
    ## Refer to ArchiveBase for documentation of these methods. ##

    def retrieve(self, measures):
        measures = np.asarray(measures, dtype=self.dtypes["measures"])
        check_batch_shape(measures, "measures", self.measure_dim, "measure_dim")

        occupied, data = self._store.retrieve(self.index_of(measures))
        fill_sentinel_values(occupied, data)

        return occupied, data

    def retrieve_single(self, measures):
        measures = np.asarray(measures, dtype=self.dtypes["measures"])
        check_shape(measures, "measures", self.measure_dim, "measure_dim")

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
