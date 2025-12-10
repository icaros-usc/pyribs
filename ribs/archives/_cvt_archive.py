"""Contains the CVTArchive."""

from __future__ import annotations

import numbers
from collections.abc import Collection, Iterator
from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from numpy_groupies import aggregate_nb as aggregate
from scipy.spatial import KDTree
from sklearn.cluster import k_means
from sklearn.neighbors import NearestNeighbors

from ribs._utils import (
    check_batch_shape,
    check_finite,
    check_shape,
    validate_batch,
    validate_single,
)
from ribs.archives._archive_base import ArchiveBase
from ribs.archives._archive_data_frame import ArchiveDataFrame
from ribs.archives._archive_stats import ArchiveStats
from ribs.archives._array_store import ArrayStore
from ribs.archives._utils import (
    fill_sentinel_values,
    parse_all_dtypes,
    validate_cma_mae_settings,
)
from ribs.typing import BatchData, FieldDesc, Float, Int, SingleData


def k_means_centroids(
    *,
    centroids: Int,
    ranges: Collection[tuple[Float, Float]],
    samples: Int | ArrayLike = 100_000,
    dtype: DTypeLike = np.float64,
    seed: Int | None = None,
    k_means_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates archive centroids with k-means clustering.

    Based on `Vassiliades 2018 <https://ieeexplore.ieee.org/document/8000667>`_, this
    function approximately generates a Centroidal Voronoi Tessellation (CVT) with
    uniformly-sized cells. This is accomplished by sampling ``samples`` points uniformly
    across the measure space range determined by ``ranges``, and then clustering the
    points into ``centroids`` clusters using k-means clustering. The set of cluster
    centroids output by k-means is used for the CVT.

    Args:
        centroids: Number of centroids to create during clustering.
        ranges: Upper and lower bound of each dimension of the measure space, e.g.,
            ``[(-1, 1), (-2, 2)]`` indicates the first dimension should have bounds
            :math:`[-1,1]` (inclusive), and the second dimension should have bounds
            :math:`[-2,2]` (inclusive). ``ranges`` should be the same length as
            ``dims``.
        samples: If it is an int, this specifies the number of samples to generate
            before clustering them to create the CVT. These points will be sampled
            uniformly within the ``ranges`` specified above. Alternatively, this
            argument can be a (num_samples, measure_dim) array of measure space points
            to cluster. It can be useful to pass in custom samples when there are
            restrictions on what samples in the measure space are (physically) possible.
        dtype: Data type of the centroids and samples.
        seed: Value to seed the random number generator and
            :func:`sklearn.cluster.k_means`. Pass None to avoid a fixed seed.
        k_means_kwargs: Keyword arguments for :func:`sklearn.cluster.k_means`. By
            default, we pass `n_init=1`, `init="random"`, `algorithm="lloyd"`, and
            `random_state=seed`. Note that these settings are geared towards quickly
            generating centroids that are "good enough." To create centroids that are
            more uniformly distributed, it may be better to use settings like
            `init="k-means++"`, though such settings will require more time to run.

    Returns:
        Two arrays. The first is a ``(centroids, measure_dim)`` array of centroids. The
        second is a ``(samples, measure_dim)`` array of samples that were clustered to
        create the centroids.

    Raises:
        ValueError: ``samples`` was passed in as an array, and the array has the wrong
            shape.
        RuntimeError: The number of centroids found during k-means clustering is not
            equal to the number of centroids passed in.
    """
    measure_dim = len(ranges)
    ranges = list(zip(*ranges, strict=True))
    lower_bounds = np.array(ranges[0], dtype=dtype)
    upper_bounds = np.array(ranges[1], dtype=dtype)

    # Apply default args for k-means. Users can override these,
    # particularly if they want higher quality clusters.
    k_means_kwargs = {} if k_means_kwargs is None else k_means_kwargs.copy()
    # By default, the `k_means` function may run the clustering multiple times and
    # choose the best output. For performance, we just run once.
    k_means_kwargs.setdefault("n_init", 1)
    # The default "k-means++" takes very long to init.
    k_means_kwargs.setdefault("init", "random")
    k_means_kwargs.setdefault("algorithm", "lloyd")
    k_means_kwargs.setdefault("random_state", seed)

    if isinstance(samples, numbers.Integral):
        # Generate random samples in the measure space when `samples` is an integer.
        rng = np.random.default_rng(seed)
        samples = rng.uniform(
            lower_bounds,
            upper_bounds,
            size=(samples, measure_dim),
        ).astype(dtype)
    else:
        # Use custom `samples` when `samples` is an array, but check the shape first.
        samples = np.asarray(samples, dtype=dtype)
        check_batch_shape(
            array=samples,
            array_name="samples",
            dim=measure_dim,
            dim_name="measure_dim",
            batch_name="num_samples",
        )

    centroid_points = k_means(samples, centroids, **k_means_kwargs)[0]

    if centroid_points.shape[0] < centroids:
        raise RuntimeError(
            "While generating the CVT, k-means clustering found "
            f"{centroid_points.shape[0]} centroids, but this "
            f"archive needs {centroids} cells. This most "
            "likely happened because there are too few samples "
            "and/or too many cells."
        )

    return centroid_points, samples


class CVTArchive(ArchiveBase):
    """An archive that tessellates the measure space with centroids.

    This archive originates in `Vassiliades 2018
    <https://ieeexplore.ieee.org/document/8000667>`_. It uses a Centroidal Voronoi
    Tessellation (CVT) to divide an n-dimensional measure space into k cells. Each cell
    is represented by a centroid, and when items are inserted into the archive, we
    identify their cell by finding the closest centroid in measure space.

    Several options are available for creating the centroids used in the CVT. The
    default option in this archive is to sample points uniformly in the measure space
    and then cluster them using k-means clustering; the centroids of the clusters are
    then used as the centroids of the CVT in this archive. This procedure is implemented
    in :func:`ribs.archives.k_means_centroids`, which internally calls
    :func:`sklearn.cluster.k_means` to perform the clustering. For alternative methods
    of centroid generation, refer to the tutorial :doc:`/tutorials/centroid_methods`.

    If running multiple experiments with this archive, it may be useful to maintain the
    same centroids across experiments. To do this, we recommend generating the centroids
    just once, such as by calling :func:`ribs.archives.k_means_centroids`. Then, save
    the centroids to a file (e.g., with :func:`numpy.save`). When constructing the
    archive for new experiments, the centroids can be loaded from the file and passed to
    the archive via the ``centroids`` parameter. More information is available in the
    aforementioned tutorial.

    Several options are also available for finding the closest centroid in measure
    space; these are set via the ``nearest_neighbors`` parameter:

    - ``nearest_neighbors="scipy_kd_tree"`` is the default option. It uses
      :class:`scipy.spatial.KDTree` to find the nearest neighbors in terms of Euclidean
      distance in O(log(number of cells)) time.
    - ``nearest_neighbors="brute_force"`` also uses Euclidean distance but operates in
      O(number of cells) time.
    - ``nearest_neighbors="sklearn_nn"`` uses
      :class:`sklearn.neighbors.NearestNeighbors` to find the nearest neighbors.

    .. note:: To compare the performance of the different nearest neighbor methods, we
        ran benchmarks where we inserted 1k batches of 100 solutions into a 2D archive
        with varying numbers of cells. We took the minimum over 5 runs for each data
        point --- minimum is recommended in the docs for :meth:`timeit.Timer.repeat`.
        Note the logarithmic scales. This plot was generated on a reasonably modern
        laptop; see `benchmarks/cvt_add.py
        <https://github.com/icaros-usc/pyribs/tree/master/benchmarks/cvt_add.py>`_ in
        the project repo for more information.

        .. image:: ../_static/imgs/cvt_add_plot.png
            :alt: Runtime to insert 100k entries into CVTArchive

        We hope that the performance differences in this plot serve as a rough guide for
        choosing nearest neighbor methods, but we note that they are not definitive, as
        each nearest neighbor method has a wide variety of options that can influence
        performance. Furthermore, performance is vastly affected by factors like the
        dimensionality of the measure space and the number of centroids/cells.

    .. note:: The idea of archive thresholds was introduced in `Fontaine 2023
        <https://arxiv.org/abs/2205.10752>`_. For more info on thresholds, including the
        ``learning_rate`` and ``threshold_min`` parameters, refer to our tutorial
        :doc:`/tutorials/cma_mae`.

    Args:
        solution_dim: Dimensionality of the solution space. Scalar or multi-dimensional
            solution shapes are allowed by passing an empty tuple or tuple of integers,
            respectively.
        centroids: This parameter may be an integer, which indicates the number of
            centroids in the CVT. In this case, the centroids will be automatically
            generated with :func:`ribs.archives.k_means_centroids`. Alternatively, this
            parameter can be a (num_centroids, measure_dim) array with the measure space
            coordinates of the centroids.
        ranges: Upper and lower bound of each dimension of the measure space, e.g.
            ``[(-1, 1), (-2, 2)]`` indicates the first dimension should have bounds
            :math:`[-1,1]` (inclusive), and the second dimension should have bounds
            :math:`[-2,2]` (inclusive). ``ranges`` should be the same length as
            ``dims``.
        learning_rate: The learning rate for threshold updates. Defaults to 1.0.
        threshold_min: The initial threshold value for all the cells.
        qd_score_offset: Archives often contain negative objective values, and if the QD
            score were to be computed with these negative objectives, the algorithm
            would be penalized for adding new cells with negative objectives. Thus, a
            standard practice is to normalize all the objectives so that they are
            non-negative by introducing an offset. This QD score offset will be
            *subtracted* from all objectives in the archive, e.g., if your objectives go
            as low as -300, pass in -300 so that each objective will be transformed as
            ``objective - (-300)``.
        seed: Value to seed the random number generator. Set to None to avoid a fixed
            seed.
        solution_dtype: Data type of the solutions. Defaults to float64 (numpy's default
            floating point type).
        objective_dtype: Data type of the objectives. Defaults to float64 (numpy's
            default floating point type).
        measures_dtype: Data type of the measures. Defaults to float64 (numpy's default
            floating point type).
        dtype: Shortcut for providing data type of the solutions, objectives, and
            measures. Defaults to float64 (numpy's default floating point type). This
            parameter sets all the dtypes simultaneously. To set individual dtypes, pass
            ``solution_dtype``, ``objective_dtype``, and ``measures_dtype``. Note that
            ``dtype`` cannot be used at the same time as those parameters.
        extra_fields: Description of extra fields of data that are stored next to elite
            data like solutions and objectives. The description is a dict mapping from a
            field name (str) to a tuple of ``(shape, dtype)``. For instance, ``{"foo":
            ((), np.float32), "bar": ((10,), np.float32)}`` will create a "foo" field
            that contains scalar values and a "bar" field that contains 10D values. Note
            that field names must be valid Python identifiers, and names already used in
            the archive are not allowed.
        samples: For convenience, this argument is passed directly to
            :func:`k_means_centroids` (assuming that function is called).
        k_means_kwargs: For convenience, this argument is passed directly to
            :func:`k_means_centroids` (assuming that function is called).
        nearest_neighbors: Method to use for computing nearest neighbors. See earlier in
            this docstring for more info.
        kdtree_kwargs: kwargs for :class:`scipy.spatial.KDTree`. By default, we do not
            pass in any kwargs. Only applicable when
            ``nearest_neighbors="scipy_kd_tree"``.
        chunk_size: If passed, brute forcing the closest centroid search will chunk the
            distance calculations to compute chunk_size inputs at a time. Only
            applicable when ``nearest_neighbors="brute_force"``.
        sklearn_nn_kwargs: kwargs for :class:`sklearn.neighbors.NearestNeighbors`. By
            default, we do not pass in any kwargs. Only applicable when
            ``nearest_neighbors="sklearn_nn"``.
        cells: DEPRECATED.
        custom_centroids: DEPRECATED.
        centroid_method: DEPRECATED.
        use_kd_tree: DEPRECATED.

    Raises:
        ValueError: Invalid values for learning_rate and threshold_min.
        ValueError: Invalid names in extra_fields.
        ValueError: ``centroids`` has the wrong shape.
        ValueError: nearest_neighbors has an invalid value.
    """

    def __init__(
        self,
        *,
        solution_dim: Int | tuple[Int, ...],
        centroids: Int | ArrayLike,
        ranges: Collection[tuple[Float, Float]],
        learning_rate: Float | None = None,
        threshold_min: Float = -np.inf,
        qd_score_offset: Float = 0.0,
        seed: Int | None = None,
        solution_dtype: DTypeLike = None,
        objective_dtype: DTypeLike = None,
        measures_dtype: DTypeLike = None,
        dtype: DTypeLike = None,
        extra_fields: FieldDesc | None = None,
        samples: Int | ArrayLike = 100_000,
        k_means_kwargs: dict | None = None,
        nearest_neighbors: Literal[
            "scipy_kd_tree", "brute_force", "sklearn_nn"
        ] = "scipy_kd_tree",
        kdtree_kwargs: dict | None = None,
        chunk_size: Int = None,
        sklearn_nn_kwargs: dict | None = None,
        # Deprecated parameters.
        cells: None = None,
        custom_centroids: None = None,
        centroid_method: None = None,
        use_kd_tree: None = None,
        ckdtree_kwargs: None = None,
    ) -> None:
        if cells is not None:
            raise ValueError(
                "`cells` is deprecated in pyribs 0.9.0. "
                "Please pass in `centroids` instead."
            )
        if custom_centroids is not None:
            raise ValueError(
                "`custom_centroids` is deprecated in pyribs 0.9.0. "
                "Please pass in `centroids` instead."
            )
        if centroid_method is not None:
            raise ValueError(
                "`centroid_method` is deprecated in pyribs 0.9.0. "
                "Please generate centroids and pass them in instead."
            )
        if use_kd_tree is not None:
            raise ValueError(
                "`use_kd_tree` is deprecated in pyribs 0.9.0. "
                "Please use `nearest_neighbors` instead."
            )
        if ckdtree_kwargs is not None:
            raise ValueError(
                "`ckdtree_kwargs` is deprecated in pyribs 0.9.0. "
                "Please use `kdtree_kwargs` instead."
            )

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
        solution_dtype, objective_dtype, measures_dtype = parse_all_dtypes(
            dtype, solution_dtype, objective_dtype, measures_dtype, np
        )
        self._store = ArrayStore(
            field_desc={
                "solution": (self.solution_dim, solution_dtype),
                "objective": ((), objective_dtype),
                "measures": (self.measure_dim, measures_dtype),
                # Must be same dtype as the objective since they share calculations.
                "threshold": ((), objective_dtype),
                **extra_fields,
            },
            capacity=(
                centroids if isinstance(centroids, numbers.Integral) else len(centroids)
            ),
        )

        # Set up constant properties.
        new_ranges = list(zip(*ranges, strict=True))
        self._lower_bounds = np.array(new_ranges[0], dtype=self.dtypes["measures"])
        self._upper_bounds = np.array(new_ranges[1], dtype=self.dtypes["measures"])
        self._interval_size = self._upper_bounds - self._lower_bounds
        self._learning_rate, self._threshold_min = validate_cma_mae_settings(
            learning_rate, threshold_min, self.dtypes["threshold"]
        )
        self._qd_score_offset = np.asarray(
            qd_score_offset, dtype=self.dtypes["objective"]
        )

        # Set up statistics -- objective_sum is the sum of all objective values in the
        # archive; it is useful for computing qd_score and obj_mean.
        self._best_elite = None
        self._objective_sum = None
        self._stats = None
        self._stats_reset()

        if isinstance(centroids, numbers.Integral):
            # Generate centroids with k-means. Ignore the samples returned.
            self._centroids, _ = k_means_centroids(
                centroids=centroids,
                ranges=ranges,
                samples=samples,
                dtype=self.dtypes["measures"],
                seed=seed,
                k_means_kwargs=k_means_kwargs,
            )
        else:
            # Validate custom centroids.
            self._centroids = np.asarray(centroids, dtype=self.dtypes["measures"])
            check_batch_shape(
                array=self._centroids,
                array_name="centroids",
                dim=self.measure_dim,
                dim_name="measure_dim",
                batch_name="num_centroids",
            )

        self._nearest_neighbors = nearest_neighbors
        if self._nearest_neighbors == "scipy_kd_tree":
            self._kdtree_kwargs = {} if kdtree_kwargs is None else kdtree_kwargs.copy()
            self._centroid_kd_tree = KDTree(self._centroids, **self._kdtree_kwargs)
        elif self._nearest_neighbors == "brute_force":
            self._chunk_size = chunk_size
        elif self._nearest_neighbors == "sklearn_nn":
            self._sklearn_nn_kwargs = (
                {} if sklearn_nn_kwargs is None else sklearn_nn_kwargs.copy()
            )
            self._sklearn_nn = NearestNeighbors(**self._sklearn_nn_kwargs)
            self._sklearn_nn.fit(self._centroids)
        else:
            raise ValueError(
                f"Unknown value `{self._nearest_neighbors}` for nearest_neighbors."
            )

    ## Properties inherited from ArchiveBase ##

    @property
    def field_list(self) -> list[str]:
        return self._store.field_list_with_index

    @property
    def dtypes(self) -> dict[str, np.dtype]:
        return self._store.dtypes_with_index

    @property
    def stats(self) -> ArchiveStats:
        return self._stats

    @property
    def empty(self) -> bool:
        return len(self._store) == 0

    ## Properties that are not in ArchiveBase ##
    ## Roughly ordered by the parameter list in the constructor. ##

    @property
    def best_elite(self) -> SingleData:
        """The elite with the highest objective in the archive.

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
    def cells(self) -> Int:
        """Total number of cells in the archive."""
        return self._store.capacity

    @property
    def lower_bounds(self) -> np.ndarray:
        """(:attr:`measure_dim`,) array listing the lower bound of each dimension."""
        return self._lower_bounds

    @property
    def upper_bounds(self) -> np.ndarray:
        """(:attr:`measure_dim`,) array listing the upper bound of each dimension."""
        return self._upper_bounds

    @property
    def interval_size(self) -> np.ndarray:
        """(:attr:`measure_dim`,) array listing the size of each dim (upper_bounds - lower_bounds)."""
        return self._interval_size

    @property
    def learning_rate(self) -> float:
        """The learning rate for threshold updates."""
        return self._learning_rate

    @property
    def threshold_min(self) -> float:
        """The initial threshold value for all the cells."""
        return self._threshold_min

    @property
    def qd_score_offset(self) -> float:
        """Subtracted from objective values when computing the QD score."""
        return self._qd_score_offset

    @property
    def centroids(self) -> np.ndarray:
        """(num_centroids, measure_dim) array of centroids used in the CVT."""
        return self._centroids

    @property
    def samples(self) -> None:
        """DEPRECATED."""
        raise ValueError("CVTArchive.samples is deprecated in pyribs 0.9.0")

    ## dunder methods ##

    def __len__(self) -> int:
        return len(self._store)

    def __iter__(self) -> Iterator[SingleData]:
        return iter(self._store)

    ## Utilities ##

    def _stats_reset(self) -> None:
        """Resets the archive stats."""
        self._best_elite = None
        self._objective_sum = np.asarray(0.0, dtype=self.dtypes["objective"])
        self._stats = ArchiveStats(
            num_elites=0,
            coverage=np.asarray(0.0, dtype=self.dtypes["objective"]),
            qd_score=np.asarray(0.0, dtype=self.dtypes["objective"]),
            norm_qd_score=np.asarray(0.0, dtype=self.dtypes["objective"]),
            obj_max=None,
            obj_mean=None,
        )

    def _stats_update(self, new_objective_sum: Float, new_best_index: Float) -> None:
        """Updates statistics.

        Update is based on a new sum of objective values (new_objective_sum) and the
        index of a potential new best elite (new_best_index).
        """
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
            - np.asarray(len(self), dtype=self.dtypes["objective"])
            * self._qd_score_offset
        )
        self._stats = ArchiveStats(
            num_elites=len(self),
            coverage=np.asarray(len(self) / self.cells, dtype=self.dtypes["objective"]),
            qd_score=new_qd_score,
            norm_qd_score=np.asarray(
                new_qd_score / self.cells, dtype=self.dtypes["objective"]
            ),
            obj_max=new_obj_max,
            obj_mean=np.asarray(
                self._objective_sum / len(self), dtype=self.dtypes["objective"]
            ),
        )

    def index_of(self, measures: ArrayLike) -> np.ndarray:
        """Finds indices of the centroids closest to the given measures.

        If ``index_batch`` is the batch of indices returned by this method, then
        ``archive.centroids[index_batch[i]]`` holds the coordinates of the centroid
        closest to ``measures[i]``. See :attr:`centroids` for more info.

        The centroid indices are located using the method specified by
        ``nearest_neighbors`` during initialization.

        Args:
            measures: (batch_size, :attr:`measure_dim`) array of coordinates in measure
                space.

        Returns:
            (batch_size,) array of centroid indices corresponding to each measure space
            coordinate.

        Raises:
            ValueError: ``measures`` is not of shape (batch_size, :attr:`measure_dim`).
            ValueError: ``measures`` has non-finite values (inf or NaN).
        """
        measures = np.asarray(measures, dtype=self.dtypes["measures"])
        check_batch_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")

        if self._nearest_neighbors == "scipy_kd_tree":
            _, indices = self._centroid_kd_tree.query(measures)
            return indices.astype(np.int32)
        elif self._nearest_neighbors == "brute_force":
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
        elif self._nearest_neighbors == "sklearn_nn":
            if len(measures) == 0:
                # sklearn's NearestNeighbors expects at least one item.
                return np.array([], dtype=np.int32)
            indices = self._sklearn_nn.kneighbors(
                measures, n_neighbors=1, return_distance=False
            )
            return indices.astype(np.int32).squeeze(1)
        else:
            raise ValueError(
                f"Unknown value `{self._nearest_neighbors}` for nearest_neighbors."
            )

    def index_of_single(self, measures: ArrayLike) -> Int:
        """Returns the index of the measures for one solution.

        See :meth:`index_of`.

        Args:
            measures: (:attr:`measure_dim`,) array of measures for a single solution.

        Returns:
            Integer index of the measures in the archive's storage arrays.

        Raises:
            ValueError: ``measures`` is not of shape (:attr:`measure_dim`,).
            ValueError: ``measures`` has non-finite values (inf or NaN).
        """
        measures = np.asarray(measures, dtype=self.dtypes["measures"])
        check_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")
        return self.index_of(measures[None])[0]

    ## Methods for writing to the archive ##

    @staticmethod
    def _compute_thresholds(
        indices: np.ndarray,
        objective: np.ndarray,
        cur_threshold: np.ndarray,
        learning_rate: float,
        dtype: np.dtype,
    ) -> np.ndarray:
        """Computes new thresholds with the CMA-MAE batch threshold update rule.

        If entries in `indices` are duplicated, they receive the same threshold.
        """
        if len(indices) == 0:
            return np.empty(0, dtype=dtype)

        # Compute the number of objectives inserted into each cell. Note that we index
        # with `indices` to place the counts at all relevant indices. For instance, if
        # we had an array [1,2,3,1,5], we would end up with [2,1,1,2,1] (there are 2
        # 1's, 1 2, 1 3, 2 1's, and 1 5).
        #
        # All objective_sizes should be > 0 since we only retrieve counts for indices in
        # `indices`.
        objective_sizes = aggregate(indices, 1, func="len", fill_value=0)[indices]  # ty: ignore[call-non-callable]

        # Compute the sum of the objectives inserted into each cell -- again, we index
        # with `indices`.
        objective_sums = aggregate(indices, objective, func="sum", fill_value=np.nan)[  # ty: ignore[call-non-callable]
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
        ratio = np.asarray(1.0 - learning_rate, dtype=dtype) ** objective_sizes
        new_threshold = ratio * cur_threshold + (objective_sums / objective_sizes) * (
            1 - ratio
        )

        return new_threshold

    def add(
        self,
        solution: ArrayLike,
        objective: ArrayLike,
        measures: ArrayLike,
        **fields: ArrayLike,
    ) -> BatchData:
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
            solution: (batch_size, :attr:`solution_dim`) array of solution parameters.
            objective: (batch_size,) array with objective function evaluations of the
                solutions.
            measures: (batch_size, :attr:`measure_dim`) array with measure space
                coordinates of all the solutions.
            fields: Additional data for each solution. Each argument should be an array
                with batch_size as the first dimension.

        Returns:
            Information describing the result of the add operation. The dict contains
            the following keys:

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
            0.0 if self.threshold_min == -np.inf else self.threshold_min
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
        archive_argmax = aggregate(  # ty: ignore[call-non-callable]
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

    def add_single(
        self,
        solution: ArrayLike,
        objective: ArrayLike,
        measures: ArrayLike,
        **fields: ArrayLike,
    ) -> SingleData:
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
            solution: Parameters of the solution.
            objective: Objective function evaluation of the solution.
            measures: Coordinates in measure space of the solution.
            fields: Additional data for the solution.

        Returns:
            Information describing the result of the add operation. The dict contains
            ``status`` and ``value`` keys; refer to :meth:`add` for the meaning of
            status and value.

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
                np.asarray(0.0, self.dtypes["threshold"])
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
                else np.asarray(0.0, dtype=self.dtypes["objective"])
            )
            self._stats_update(self._objective_sum + objective - cur_objective, index)

        # Value is the improvement over the current threshold (can be negative).
        add_info["value"] = objective - cur_threshold

        return add_info

    def clear(self) -> None:
        """Removes all elites in the archive."""
        self._store.clear()
        self._stats_reset()

    ## Methods for reading from the archive ##
    ## Refer to ArchiveBase for documentation of these methods. ##

    def retrieve(self, measures: ArrayLike) -> tuple[np.ndarray, BatchData]:
        measures = np.asarray(measures, dtype=self.dtypes["measures"])
        check_batch_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")

        occupied, data = self._store.retrieve(self.index_of(measures))
        fill_sentinel_values(occupied, data)

        return occupied, data

    def retrieve_single(self, measures: ArrayLike) -> tuple[bool, SingleData]:
        measures = np.asarray(measures, dtype=self.dtypes["measures"])
        check_shape(measures, "measures", self.measure_dim, "measure_dim")
        check_finite(measures, "measures")

        occupied, data = self.retrieve(measures[None])

        return occupied[0], {field: arr[0] for field, arr in data.items()}

    @overload
    def data(
        self,
        fields: str,
        return_type: Literal["dict", "tuple", "pandas"] = "dict",
    ) -> np.ndarray: ...

    @overload
    def data(
        self,
        fields: None | Collection[str] = None,
        return_type: Literal["dict"] = "dict",
    ) -> BatchData: ...

    @overload
    def data(
        self,
        fields: None | Collection[str] = None,
        return_type: Literal["tuple"] = "tuple",
    ) -> tuple[np.ndarray]: ...

    @overload
    def data(
        self,
        fields: None | Collection[str] = None,
        return_type: Literal["pandas"] = "pandas",
    ) -> ArchiveDataFrame: ...

    def data(
        self,
        fields: None | Collection[str] | str = None,
        return_type: Literal["dict", "tuple", "pandas"] = "dict",
    ) -> np.ndarray | BatchData | tuple[np.ndarray] | ArchiveDataFrame:
        return self._store.data(fields, return_type)

    def sample_elites(self, n: Int) -> BatchData:
        if self.empty:
            raise IndexError("No elements in archive.")

        random_indices = self._rng.integers(len(self._store), size=n)
        selected_indices = self._store.occupied_list[random_indices]
        _, elites = self._store.retrieve(selected_indices)
        return elites
