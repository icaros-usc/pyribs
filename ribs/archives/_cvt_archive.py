"""Contains the CVTArchive class."""
import numbers

import numpy as np
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
from scipy.stats.qmc import Halton, Sobol
from sklearn.cluster import k_means

from ribs._utils import check_batch_shape, check_finite
from ribs.archives._archive_base import ArchiveBase


class CVTArchive(ArchiveBase):
    """An archive that divides the entire measure space into a fixed number of
    cells.

    This archive originates in `Vassiliades 2018
    <https://ieeexplore.ieee.org/document/8000667>`_. It uses Centroidal Voronoi
    Tessellation (CVT) to divide an n-dimensional measure space into k cells.
    The CVT is created by sampling points uniformly from the n-dimensional
    measure space and using k-means clustering to identify k centroids. When
    items are inserted into the archive, we identify their cell by identifying
    the closest centroid in measure space (using Euclidean distance). For
    k-means clustering, we use :func:`sklearn.cluster.k_means`.

    By default, finding the closest centroid is done in roughly
    O(log(number of cells)) time using :class:`scipy.spatial.cKDTree`. To switch
    to brute force, which takes O(number of cells) time, pass
    ``use_kd_tree=False``.

    To compare the performance of using the k-D tree vs brute force, we ran
    benchmarks where we inserted 1k batches of 100 solutions into a 2D archive
    with varying numbers of cells. We took the minimum over 5 runs for each data
    point, as recommended in the docs for :meth:`timeit.Timer.repeat`.  Note the
    logarithmic scales. This plot was generated on a reasonably modern laptop.

    .. image:: ../_static/imgs/cvt_add_plot.png
        :alt: Runtime to insert 100k entries into CVTArchive

    Across almost all numbers of cells, using the k-D tree is faster than using
    brute force. Thus, **we recommend always using the k-D tree.** See
    `benchmarks/cvt_add.py
    <https://github.com/icaros-usc/pyribs/tree/master/benchmarks/cvt_add.py>`_
    in the project repo for more information about how this plot was generated.

    Finally, if running multiple experiments, it may be beneficial to use the
    same centroids across each experiment. Doing so can keep experiments
    consistent and reduce execution time. To do this, either (1) construct
    custom centroids and pass them in via the ``custom_centroids`` argument, or
    (2) access the centroids created in the first archive with :attr:`centroids`
    and pass them into ``custom_centroids`` when constructing archives for
    subsequent experiments.

    .. note:: The idea of archive thresholds was introduced in `Fontaine 2022
        <https://arxiv.org/abs/2205.10752>`_. For more info on thresholds,
        including the ``learning_rate`` and ``threshold_min`` parameters, refer
        to our tutorial :doc:`/tutorials/cma_mae`.

    .. note:: For more information on our choice of k-D tree implementation, see
        :pr:`38`.

    Args:
        solution_dim (int): Dimension of the solution space.
        cells (int): The number of cells to use in the archive, equivalent to
            the number of centroids/areas in the CVT.
        ranges (array-like of (float, float)): Upper and lower bound of each
            dimension of the measure space, e.g. ``[(-1, 1), (-2, 2)]``
            indicates the first dimension should have bounds :math:`[-1,1]`
            (inclusive), and the second dimension should have bounds
            :math:`[-2,2]` (inclusive). ``ranges`` should be the same length as
            ``dims``.
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
        seed (int): Value to seed the random number generator as well as
            :func:`~sklearn.cluster.k_means`. Set to None to avoid a fixed seed.
        dtype (str or data-type): Data type of the solutions, objectives,
            and measures. We only support ``"f"`` / ``np.float32`` and ``"d"`` /
            ``np.float64``.
        custom_centroids (array-like): If passed in, this (cells, measure_dim)
            array will be used as the centroids of the CVT instead of generating
            new ones. In this case, ``samples`` will be ignored, and
            ``archive.samples`` will be None. This can be useful when one wishes
            to use the same CVT across experiments for fair comparison.
        centroid_method (str): Pass in the following methods for
            generating centroids: "random", "sobol", "scrambled sobol",
            "halton". Default method is "kmeans". These methods are derived from
            Mouret 2023: https://dl.acm.org/doi/pdf/10.1145/3583133.3590726.
            Note: Samples are only used when method is "kmeans".
        samples (int or array-like): If it is an int, this specifies the number
            of samples to generate when creating the CVT. Otherwise, this must
            be a (num_samples, measure_dim) array where samples[i] is a sample
            to use when creating the CVT. It can be useful to pass in custom
            samples when there are restrictions on what samples in the measure
            space are (physically) possible.
        k_means_kwargs (dict): kwargs for :func:`~sklearn.cluster.k_means`. By
            default, we pass in `n_init=1`, `init="random"`,
            `algorithm="lloyd"`, and `random_state=seed`.
        use_kd_tree (bool): If True, use a k-D tree for finding the closest
            centroid when inserting into the archive. If False, brute force will
            be used instead.
        ckdtree_kwargs (dict): kwargs for :class:`~scipy.spatial.cKDTree`. By
            default, we do not pass in any kwargs.
        chunk_size (int): If passed, brute forcing the closest centroid search
            will chunk the distance calculations to compute chunk_size inputs at
            a time.
    Raises:
        ValueError: The ``samples`` array or the ``custom_centroids`` array has
            the wrong shape.
    """

    def __init__(self,
                 *,
                 solution_dim,
                 cells,
                 ranges,
                 learning_rate=1.0,
                 threshold_min=-np.inf,
                 qd_score_offset=0.0,
                 seed=None,
                 dtype=np.float64,
                 custom_centroids=None,
                 centroid_method="kmeans",
                 samples=100_000,
                 k_means_kwargs=None,
                 use_kd_tree=True,
                 ckdtree_kwargs=None,
                 chunk_size=None):

        ArchiveBase.__init__(
            self,
            solution_dim=solution_dim,
            cells=cells,
            measure_dim=len(ranges),
            learning_rate=learning_rate,
            threshold_min=threshold_min,
            qd_score_offset=qd_score_offset,
            seed=seed,
            dtype=dtype,
        )

        ranges = list(zip(*ranges))
        self._lower_bounds = np.array(ranges[0], dtype=self.dtype)
        self._upper_bounds = np.array(ranges[1], dtype=self.dtype)

        # Apply default args for k-means. Users can easily override these,
        # particularly if they want higher quality clusters.
        self._k_means_kwargs = ({} if k_means_kwargs is None else
                                k_means_kwargs.copy())
        self._k_means_kwargs.setdefault(
            # Only run one iter to be fast.
            "n_init",
            1)
        self._k_means_kwargs.setdefault(
            # The default "k-means++" takes very long to init.
            "init",
            "random")
        self._k_means_kwargs.setdefault("algorithm", "lloyd")
        self._k_means_kwargs.setdefault("random_state", seed)

        if custom_centroids is None:
            self._samples = None
            if centroid_method == "kmeans":
                if not isinstance(samples, numbers.Integral):
                    # Validate shape of custom samples.
                    samples = np.asarray(samples, dtype=self.dtype)
                    if samples.shape[1] != self._measure_dim:
                        raise ValueError(
                            f"Samples has shape {samples.shape} but must be of "
                            f"shape (n_samples, len(ranges)="
                            f"{self._measure_dim})")
                    self._samples = samples
                else:
                    self._samples = self._rng.uniform(
                        self._lower_bounds,
                        self._upper_bounds,
                        size=(samples, self._measure_dim),
                    ).astype(self.dtype)

                self._centroids = k_means(self._samples, self._cells,
                                          **self._k_means_kwargs)[0]

                if self._centroids.shape[0] < self._cells:
                    raise RuntimeError(
                        "While generating the CVT, k-means clustering found "
                        f"{self._centroids.shape[0]} centroids, but this "
                        f"archive needs {self._cells} cells. This most "
                        "likely happened because there are too few samples "
                        "and/or too many cells.")
            elif centroid_method == "random":
                # Generate random centroids for the archive.
                self._centroids = self._rng.uniform(self._lower_bounds,
                                                    self._upper_bounds,
                                                    size=(self._cells,
                                                          self._measure_dim))
            elif centroid_method == "sobol":
                # Generate self._cells number of centroids as a Sobol sequence.
                sampler = Sobol(d=self._measure_dim, scramble=False)
                num_points = np.log2(self._cells).astype(int)
                self._centroids = sampler.random_base2(num_points)
            elif centroid_method == "scrambled_sobol":
                # Generates centroids as a scrambled Sobol sequence.
                sampler = Sobol(d=self._measure_dim, scramble=True)
                num_points = np.log2(self._cells).astype(int)
                self._centroids = sampler.random_base2(num_points)
            elif centroid_method == "halton":
                # Generates centroids using a Halton sequence.
                sampler = Halton(d=self._measure_dim)
                self._centroids = sampler.random(n=self._cells)
        else:
            # Validate shape of `custom_centroids` when they are provided.
            custom_centroids = np.asarray(custom_centroids, dtype=self.dtype)
            if custom_centroids.shape != (cells, self._measure_dim):
                raise ValueError(
                    f"custom_centroids has shape {custom_centroids.shape} but "
                    f"must be of shape (cells={cells}, len(ranges)="
                    f"{self._measure_dim})")
            self._centroids = custom_centroids
            self._samples = None

        self._use_kd_tree = use_kd_tree
        self._centroid_kd_tree = None
        self._ckdtree_kwargs = ({} if ckdtree_kwargs is None else
                                ckdtree_kwargs.copy())
        self._chunk_size = chunk_size
        if self._use_kd_tree:
            self._centroid_kd_tree = cKDTree(self._centroids,
                                             **self._ckdtree_kwargs)

    @property
    def lower_bounds(self):
        """(measure_dim,) numpy.ndarray: Lower bound of each dimension."""
        return self._lower_bounds

    @property
    def upper_bounds(self):
        """(measure_dim,) numpy.ndarray: Upper bound of each dimension."""
        return self._upper_bounds

    @property
    def samples(self):
        """(num_samples, measure_dim) numpy.ndarray: The samples used in
        creating the CVT.

        Will be None if custom centroids were passed in to the archive.
        """
        return self._samples

    @property
    def centroids(self):
        """(n_centroids, measure_dim) numpy.ndarray: The centroids used in the
        CVT.
        """
        return self._centroids

    def index_of(self, measures_batch):
        """Finds the indices of the centroid closest to the given coordinates in
        measure space.

        If ``index_batch`` is the batch of indices returned by this method, then
        ``archive.centroids[index_batch[i]]`` holds the coordinates of the
        centroid closest to ``measures_batch[i]``. See :attr:`centroids` for
        more info.

        The centroid indices are located using either the k-D tree or brute
        force, depending on the value of ``use_kd_tree`` in the constructor.

        Args:
            measures_batch (array-like): (batch_size, :attr:`measure_dim`)
                array of coordinates in measure space.
        Returns:
            numpy.ndarray: (batch_size,) array of centroid indices
            corresponding to each measure space coordinate.
        Raises:
            ValueError: ``measures_batch`` is not of shape (batch_size,
                :attr:`measure_dim`).
            ValueError: ``measures_batch`` has non-finite values (inf or NaN).
        """
        measures_batch = np.asarray(measures_batch)
        check_batch_shape(measures_batch, "measures_batch", self.measure_dim,
                          "measure_dim")
        check_finite(measures_batch, "measures_batch")

        if self._use_kd_tree:
            _, indices = self._centroid_kd_tree.query(measures_batch)
            return indices.astype(np.int32)
        else:
            expanded_measures = np.expand_dims(measures_batch, axis=1)
            # Compute indices chunks at a time
            if self._chunk_size is not None and \
                    self._chunk_size < measures_batch.shape[0]:
                indices = []
                chunks = np.array_split(
                    expanded_measures,
                    np.ceil(len(expanded_measures) / self._chunk_size))
                for chunk in chunks:
                    distances = chunk - self.centroids
                    distances = np.sum(np.square(distances), axis=2)
                    current_res = np.argmin(distances, axis=1).astype(np.int32)
                    indices.append(current_res)
                return np.concatenate(tuple(indices))
            else:
                # Brute force distance calculation -- start by taking the
                # difference between each measure i and all the centroids.
                distances = expanded_measures - self.centroids
                # Compute the total squared distance -- no need to compute
                # actual distance with a sqrt.
                distances = np.sum(np.square(distances), axis=2)
                return np.argmin(distances, axis=1).astype(np.int32)
