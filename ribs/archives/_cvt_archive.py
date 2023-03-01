"""Contains the CVTArchive class."""
import numpy as np
import semantic_version
import sklearn
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
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
    brute force. Thus, **we recommend always using he k-D tree.** See
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
        <https://arxiv.org/abs/2205.10752>`_. Refer to our `CMA-MAE tutorial
        <../../tutorials/cma_mae.html>`_ for more info on thresholds, including
        the ``learning_rate`` and ``threshold_min`` parameters.

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
        samples (int or array-like): If it is an int, this specifies the number
            of samples to generate when creating the CVT. Otherwise, this must
            be a (num_samples, measure_dim) array where samples[i] is a sample
            to use when creating the CVT. It can be useful to pass in custom
            samples when there are restrictions on what samples in the measure
            space are (physically) possible.
        custom_centroids (array-like): If passed in, this (cells, measure_dim)
            array will be used as the centroids of the CVT instead of generating
            new ones. In this case, ``samples`` will be ignored, and
            ``archive.samples`` will be None. This can be useful when one wishes
            to use the same CVT across experiments for fair comparison.
        k_means_kwargs (dict): kwargs for :func:`~sklearn.cluster.k_means`. By
            default, we pass in `n_init=1`, `init="random"`,
            `algorithm="lloyd"`, and `random_state=seed`.
        use_kd_tree (bool): If True, use a k-D tree for finding the closest
            centroid when inserting into the archive. If False, brute force will
            be used instead.
        ckdtree_kwargs (dict): kwargs for :class:`~scipy.spatial.cKDTree`. By
            default, we do not pass in any kwargs.
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
                 samples=100_000,
                 custom_centroids=None,
                 k_means_kwargs=None,
                 use_kd_tree=True,
                 ckdtree_kwargs=None):

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
        if "n_init" not in self._k_means_kwargs:
            # Only run one iter to be fast.
            self._k_means_kwargs["n_init"] = 1
        if "init" not in self._k_means_kwargs:
            # The default, "k-means++", takes very long to init.
            self._k_means_kwargs["init"] = "random"
        if "algorithm" not in self._k_means_kwargs:
            if semantic_version.Version(
                    sklearn.__version__) >= semantic_version.Version("1.1.0"):
                # In the newer versions, "full" has been deprecated in favor of
                # "lloyd".
                self._k_means_kwargs["algorithm"] = "lloyd"
            else:
                # The default, "auto"/"elkan", allocates a huge array.
                self._k_means_kwargs["algorithm"] = "full"
        if "random_state" not in self._k_means_kwargs:
            self._k_means_kwargs["random_state"] = seed

        self._use_kd_tree = use_kd_tree
        self._centroid_kd_tree = None
        self._ckdtree_kwargs = ({} if ckdtree_kwargs is None else
                                ckdtree_kwargs.copy())

        if custom_centroids is None:
            if not isinstance(samples, int):
                # Validate shape of custom samples. These are ignored when
                # `custom_centroids` is provided.
                samples = np.asarray(samples, dtype=self.dtype)
                if samples.shape[1] != self._measure_dim:
                    raise ValueError(
                        f"Samples has shape {samples.shape} but must be of "
                        f"shape (n_samples, len(ranges)={self._measure_dim})")
            self._samples = samples
            self._centroids = None
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
        if self._centroids is None:
            self._samples = self._rng.uniform(
                self._lower_bounds,
                self._upper_bounds,
                size=(self._samples, self._measure_dim),
            ).astype(self.dtype) if isinstance(self._samples,
                                               int) else self._samples

            self._centroids = k_means(self._samples, self._cells,
                                      **self._k_means_kwargs)[0]

            if self._centroids.shape[0] < self._cells:
                raise RuntimeError(
                    "While generating the CVT, k-means clustering found "
                    f"{self._centroids.shape[0]} centroids, but this archive "
                    f"needs {self._cells} cells. This most likely happened "
                    "because there are too few samples and/or too many cells.")

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

        May be None until :meth:`initialize` is called.
        """
        return self._samples

    @property
    def centroids(self):
        """(n_centroids, measure_dim) numpy.ndarray: The centroids used in the
        CVT.

        None until :meth:`initialize` is called.
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
            return np.asarray(
                self._centroid_kd_tree.query(measures_batch))[1].astype(
                    np.int32)

        # Brute force distance calculation -- start by taking the difference
        # between each measure i and all the centroids.
        distances = np.expand_dims(measures_batch, axis=1) - self.centroids

        # Compute the total squared distance -- no need to compute actual
        # distance with a sqrt.
        distances = np.sum(np.square(distances), axis=2)

        return np.argmin(distances, axis=1).astype(np.int32)
