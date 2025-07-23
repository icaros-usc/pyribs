"""Contains the CVTArchive."""

import numbers

import numpy as np
from numpy_groupies import aggregate_nb as aggregate
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
from scipy.stats.qmc import Halton, Sobol
from sklearn.cluster import k_means

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
from ribs.archives._utils import (
    fill_sentinel_values,
    parse_dtype,
    validate_cma_mae_settings,
)


class CVTArchive(ArchiveBase):
    # pylint: disable = too-many-public-methods
    """An archive that tessellates the measure space with centroids.

    This archive originates in `Vassiliades 2018
    <https://ieeexplore.ieee.org/document/8000667>`_. It uses Centroidal Voronoi
    Tessellation (CVT) to divide an n-dimensional measure space into k cells. The CVT is
    created by sampling points uniformly from the n-dimensional measure space and using
    k-means clustering to identify k centroids. When items are inserted into the
    archive, we identify their cell by identifying the closest centroid in measure space
    (using Euclidean distance). For k-means clustering, we use
    :func:`sklearn.cluster.k_means`.

    By default, finding the closest centroid is done in roughly O(log(number of cells))
    time using :class:`scipy.spatial.cKDTree`. To switch to brute force, which takes
    O(number of cells) time, pass ``use_kd_tree=False``.

    To compare the performance of using the k-D tree vs brute force, we ran benchmarks
    where we inserted 1k batches of 100 solutions into a 2D archive with varying numbers
    of cells. We took the minimum over 5 runs for each data point, as recommended in the
    docs for :meth:`timeit.Timer.repeat`. Note the logarithmic scales. This plot was
    generated on a reasonably modern laptop.

    .. image:: ../_static/imgs/cvt_add_plot.png
        :alt: Runtime to insert 100k entries into CVTArchive

    Across almost all numbers of cells, using the k-D tree is faster than using brute
    force. Thus, **we recommend always using the k-D tree.** See `benchmarks/cvt_add.py
    <https://github.com/icaros-usc/pyribs/tree/master/benchmarks/cvt_add.py>`_ in the
    project repo for more information about how this plot was generated.

    Finally, if running multiple experiments, it may be beneficial to use the same
    centroids across each experiment. Doing so can keep experiments consistent and
    reduce execution time. To do this, either (1) construct custom centroids and pass
    them in via the ``custom_centroids`` argument, or (2) access the centroids created
    in the first archive with :attr:`centroids` and pass them into ``custom_centroids``
    when constructing archives for subsequent experiments.

    .. note:: The idea of archive thresholds was introduced in `Fontaine 2023
        <https://arxiv.org/abs/2205.10752>`_. For more info on thresholds, including the
        ``learning_rate`` and ``threshold_min`` parameters, refer to our tutorial
        :doc:`/tutorials/cma_mae`.

    .. note:: For more information on our choice of k-D tree implementation, see
        :pr:`38`.

    Args:
        solution_dim (int or tuple of int): Dimensionality of the solution space. Scalar
            or multi-dimensional solution shapes are allowed by passing an empty tuple
            or tuple of integers, respectively.
        cells (int): The number of cells to use in the archive, equivalent to the number
            of centroids/areas in the CVT.
        ranges (array-like of (float, float)): Upper and lower bound of each dimension
            of the measure space, e.g. ``[(-1, 1), (-2, 2)]`` indicates the first
            dimension should have bounds :math:`[-1,1]` (inclusive), and the second
            dimension should have bounds :math:`[-2,2]` (inclusive). ``ranges`` should
            be the same length as ``dims``.
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
        seed (int): Value to seed the random number generator as well as
            :func:`~sklearn.cluster.k_means`. Set to None to avoid a fixed seed.
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
        custom_centroids (array-like): If passed in, this (cells, measure_dim) array
            will be used as the centroids of the CVT instead of generating new ones. In
            this case, ``samples`` will be ignored, and ``archive.samples`` will be
            None. This can be useful when one wishes to use the same CVT across
            experiments for fair comparison.
        centroid_method (str): Pass in the following methods for generating centroids:
            "random", "sobol", "scrambled_sobol", "halton". Default method is "kmeans".
            These methods are derived from Mouret 2023:
            https://dl.acm.org/doi/pdf/10.1145/3583133.3590726. Note: Samples are only
            used when method is "kmeans".
        samples (int or array-like): If it is an int, this specifies the number of
            samples to generate when creating the CVT. Otherwise, this must be a
            (num_samples, measure_dim) array where samples[i] is a sample to use when
            creating the CVT. It can be useful to pass in custom samples when there are
            restrictions on what samples in the measure space are (physically) possible.
        k_means_kwargs (dict): kwargs for :func:`~sklearn.cluster.k_means`. By default,
            we pass in `n_init=1`, `init="random"`, `algorithm="lloyd"`, and
            `random_state=seed`.
        use_kd_tree (bool): If True, use a k-D tree for finding the closest centroid
            when inserting into the archive. If False, brute force will be used instead.
        ckdtree_kwargs (dict): kwargs for :class:`~scipy.spatial.cKDTree`. By default,
            we do not pass in any kwargs.
        chunk_size (int): If passed, brute forcing the closest centroid search will
            chunk the distance calculations to compute chunk_size inputs at a time.
    Raises:
        ValueError: Invalid values for learning_rate and threshold_min.
        ValueError: Invalid names in extra_fields.
        ValueError: The ``samples`` array or the ``custom_centroids`` array has the
            wrong shape.
    """

    def __init__(
        self,
        *,
        solution_dim,
        cells,
        ranges,
        learning_rate=None,
        threshold_min=-np.inf,
        qd_score_offset=0.0,
        seed=None,
        dtype=np.float64,
        extra_fields=None,
        custom_centroids=None,
        centroid_method="kmeans",
        samples=100_000,
        k_means_kwargs=None,
        use_kd_tree=True,
        ckdtree_kwargs=None,
        chunk_size=None,
    ):
        self._rng = np.random.default_rng(seed)

        ArchiveBase.__init__(
            self,
            solution_dim=solution_dim,
            objective_dim=(),
            measure_dim=len(ranges),
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
            capacity=cells,
        )

        # Set up constant properties.
        ranges = list(zip(*ranges))
        self._lower_bounds = np.array(ranges[0], dtype=self.dtypes["measures"])
        self._upper_bounds = np.array(ranges[1], dtype=self.dtypes["measures"])
        self._interval_size = self._upper_bounds - self._lower_bounds
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

        # Apply default args for k-means. Users can easily override these, particularly
        # if they want higher quality clusters.
        self._k_means_kwargs = {} if k_means_kwargs is None else k_means_kwargs.copy()
        self._k_means_kwargs.setdefault(
            # Only run one iter to be fast.
            "n_init",
            1,
        )
        self._k_means_kwargs.setdefault(
            # The default "k-means++" takes very long to init.
            "init",
            "random",
        )
        self._k_means_kwargs.setdefault("algorithm", "lloyd")
        self._k_means_kwargs.setdefault("random_state", seed)

        if custom_centroids is None:
            self._samples = None
            if centroid_method == "kmeans":
                if not isinstance(samples, numbers.Integral):
                    # Validate shape of custom samples.
                    samples = np.asarray(samples, dtype=self.dtypes["measures"])
                    if samples.shape[1] != self._measure_dim:
                        raise ValueError(
                            f"Samples has shape {samples.shape} but must be of "
                            f"shape (n_samples, len(ranges)="
                            f"{self._measure_dim})"
                        )
                    self._samples = samples
                else:
                    self._samples = self._rng.uniform(
                        self._lower_bounds,
                        self._upper_bounds,
                        size=(samples, self._measure_dim),
                    ).astype(self.dtypes["measures"])

                self._centroids = k_means(
                    self._samples, self.cells, **self._k_means_kwargs
                )[0]

                if self._centroids.shape[0] < self.cells:
                    raise RuntimeError(
                        "While generating the CVT, k-means clustering found "
                        f"{self._centroids.shape[0]} centroids, but this "
                        f"archive needs {self.cells} cells. This most "
                        "likely happened because there are too few samples "
                        "and/or too many cells."
                    )
            elif centroid_method == "random":
                # Generates random centroids.
                self._centroids = self._rng.uniform(
                    self._lower_bounds,
                    self._upper_bounds,
                    size=(self.cells, self._measure_dim),
                )
            elif centroid_method == "sobol":
                # Generates centroids as a Sobol sequence.
                sampler = Sobol(d=self._measure_dim, scramble=False)
                sobol_nums = sampler.random(n=self.cells)
                self._centroids = self._lower_bounds + sobol_nums * (
                    self._upper_bounds - self._lower_bounds
                )
            elif centroid_method == "scrambled_sobol":
                # Generates centroids as a scrambled Sobol sequence.
                sampler = Sobol(d=self._measure_dim, scramble=True)
                sobol_nums = sampler.random(n=self.cells)
                self._centroids = self._lower_bounds + sobol_nums * (
                    self._upper_bounds - self._lower_bounds
                )
            elif centroid_method == "halton":
                # Generates centroids with a Halton sequence.
                sampler = Halton(d=self._measure_dim)
                halton_nums = sampler.random(n=self.cells)
                self._centroids = self._lower_bounds + halton_nums * (
                    self._upper_bounds - self._lower_bounds
                )
        else:
            # Validate shape of `custom_centroids` when they are provided.
            custom_centroids = np.asarray(
                custom_centroids, dtype=self.dtypes["measures"]
            )
            if custom_centroids.shape != (cells, self._measure_dim):
                raise ValueError(
                    f"custom_centroids has shape {custom_centroids.shape} but "
                    f"must be of shape (cells={cells}, len(ranges)="
                    f"{self._measure_dim})"
                )
            self._centroids = custom_centroids
            self._samples = None

        self._use_kd_tree = use_kd_tree
        self._centroid_kd_tree = None
        self._ckdtree_kwargs = {} if ckdtree_kwargs is None else ckdtree_kwargs.copy()
        self._chunk_size = chunk_size
        if self._use_kd_tree:
            self._centroid_kd_tree = cKDTree(self._centroids, **self._ckdtree_kwargs)

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

    @property
    def centroids(self):
        """(n_centroids, measure_dim) numpy.ndarray: The centroids used in the
        CVT.
        """
        return self._centroids

    @property
    def samples(self):
        """(num_samples, measure_dim) numpy.ndarray: The samples used in creating the
        CVT.

        Will be None if custom centroids were passed in to the archive.
        """
        return self._samples

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
        """Finds the indices of the centroid closest to the given coordinates in measure
        space.

        If ``index_batch`` is the batch of indices returned by this method, then
        ``archive.centroids[index_batch[i]]`` holds the coordinates of the centroid
        closest to ``measures[i]``. See :attr:`centroids` for more info.

        The centroid indices are located using either the k-D tree or brute force,
        depending on the value of ``use_kd_tree`` in the constructor.

        Args:
            measures (array-like): (batch_size, :attr:`measure_dim`) array of
                coordinates in measure space.
        Returns:
            numpy.ndarray: (batch_size,) array of centroid indices
            corresponding to each measure space coordinate.
        Raises:
            ValueError: ``measures`` is not of shape (batch_size, :attr:`measure_dim`).
            ValueError: ``measures`` has non-finite values (inf or NaN).
        """
        measures = np.asarray(measures)
        check_batch_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")

        if self._use_kd_tree:
            _, indices = self._centroid_kd_tree.query(measures)
            return indices.astype(np.int32)
        else:
            expanded_measures = np.expand_dims(measures, axis=1)
            # Compute indices chunks at a time
            if self._chunk_size is not None and self._chunk_size < measures.shape[0]:
                indices = []
                chunks = np.array_split(
                    expanded_measures,
                    np.ceil(len(expanded_measures) / self._chunk_size),
                )
                for chunk in chunks:
                    distances = chunk - self.centroids
                    distances = np.sum(np.square(distances), axis=2)
                    current_res = np.argmin(distances, axis=1).astype(np.int32)
                    indices.append(current_res)
                return np.concatenate(tuple(indices))
            else:
                # Brute force distance calculation -- start by taking the difference
                # between each measure i and all the centroids.
                distances = expanded_measures - self.centroids
                # Compute the total squared distance -- no need to compute actual
                # distance with a sqrt.
                distances = np.sum(np.square(distances), axis=2)
                return np.argmin(distances, axis=1).astype(np.int32)

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
