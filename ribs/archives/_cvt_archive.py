"""Contains the CVTArchive class."""
import numpy as np
from numba import jit
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module
from sklearn.cluster import k_means

from ribs._utils import check_measures_batch_shape
from ribs.archives._archive_base import ArchiveBase


class CVTArchive(ArchiveBase):
    """An archive that divides the entire behavior space into a fixed number of
    cells.

    This archive originates in `Vassiliades 2018
    <https://ieeexplore.ieee.org/document/8000667>`_. It uses Centroidal Voronoi
    Tesselation (CVT) to divide an n-dimensional behavior space into k cell. The
    CVT is created by sampling points uniformly from the n-dimensional behavior
    space and using k-means clustering to identify k centroids. When items are
    inserted into the archive, we identify their cell by identifying the closest
    centroid in behavior space (using Euclidean distance). For k-means
    clustering, we use :func:`sklearn.cluster.k_means`.

    Finding the closest centroid is done in O(cells) time (i.e. brute force) by
    default. If ``use_kd_tree`` is True, it can be done in roughly O(log cells)
    time using :class:`scipy.spatial.cKDTree`. However, using the k-D tree
    lowers performance for small numbers of cells. The following plot compares
    the runtime of brute force vs k-D tree when inserting 100k samples into a 2D
    archive with varying numbers of cells (we took the minimum over 5 runs for
    each data point, as recommended in the docs for :meth:`timeit.Timer.repeat`.
    Note the logarithmic scales. This plot was generated on a reasonably modern
    laptop.

    .. image:: ../_static/imgs/cvt_add_plot.png
        :alt: Runtime to insert 100k entries into CVTArchive

    Archives with at least 5k cells seem to have faster insertion when using a
    k-D tree than when using brute force, so **we recommend setting**
    ``use_kd_tree`` **if the** ``CVTArchive`` **has at least 5k cells**. See
    `benchmarks/cvt_add.py
    <https://github.com/icaros-usc/pyribs/tree/master/benchmarks/cvt_add.py>`_
    in the project repo for more information about how this plot was generated.

    Finally, if running multiple experiments, it may be beneficial to use the
    same centroids across each experiment. Doing so can keep experiments
    consistent and reduce execution time. To do this, either 1) construct custom
    centroids and pass them in via the ``custom_centroids`` argument, or 2)
    access the centroids created in the first archive with :attr:`centroids` and
    pass them into ``custom_centroids`` when constructing archives for
    subsequent experiments.

    Args:
        solution_dim (int): Dimension of the solution space.
        cells (int): The number of cells to use in the archive, equivalent to
            the number of centroids/areas in the CVT.
        ranges (array-like of (float, float)): Upper and lower bound of each
            dimension of the behavior space, e.g. ``[(-1, 1), (-2, 2)]``
            indicates the first dimension should have bounds :math:`[-1,1]`
            (inclusive), and the second dimension should have bounds
            :math:`[-2,2]` (inclusive). ``ranges`` should be the same length as
            ``dims``.
        seed (int): Value to seed the random number generator as well as
            :func:`~sklearn.cluster.k_means`. Set to None to avoid a fixed seed.
        dtype (str or data-type): Data type of the solutions, objective values,
            and behavior values. We only support ``"f"`` / :class:`np.float32`
            and ``"d"`` / :class:`np.float64`.
        samples (int or array-like): If it is an int, this specifies the number
            of samples to generate when creating the CVT. Otherwise, this must
            be a (num_samples, behavior_dim) array where samples[i] is a sample
            to use when creating the CVT. It can be useful to pass in custom
            samples when there are restrictions on what samples in the behavior
            space are (physically) possible.
        custom_centroids (array-like): If passed in, this (cells, behavior_dim)
            array will be used as the centroids of the CVT instead of generating
            new ones. In this case, ``samples`` will be ignored, and
            ``archive.samples`` will be None. This can be useful when one wishes
            to use the same CVT across experiments for fair comparison.
        k_means_kwargs (dict): kwargs for :func:`~sklearn.cluster.k_means`. By
            default, we pass in `n_init=1`, `init="random"`, `algorithm="full"`,
            and `random_state=seed`.
        use_kd_tree (bool): If True, use a k-D tree for finding the closest
            centroid when inserting into the archive. This may result in a
            speedup for larger dimensions.
        ckdtree_kwargs (dict): kwargs for :class:`~scipy.spatial.cKDTree`. By
            default, we do not pass in any kwargs.
    Raises:
        ValueError: The ``samples`` array or the ``custom_centroids`` array has
            the wrong shape.
    """

    def __init__(self,
                 solution_dim,
                 cells,
                 ranges,
                 seed=None,
                 dtype=np.float64,
                 samples=100_000,
                 custom_centroids=None,
                 k_means_kwargs=None,
                 use_kd_tree=False,
                 ckdtree_kwargs=None):

        ArchiveBase.__init__(
            self,
            solution_dim=solution_dim,
            cells=cells,
            behavior_dim=len(ranges),
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
                if samples.shape[1] != self._behavior_dim:
                    raise ValueError(
                        f"Samples has shape {samples.shape} but must be of "
                        f"shape (n_samples, len(ranges)={self._behavior_dim})")
            self._samples = samples
            self._centroids = None
        else:
            # Validate shape of `custom_centroids` when they are provided.
            custom_centroids = np.asarray(custom_centroids, dtype=self.dtype)
            if custom_centroids.shape != (cells, self._behavior_dim):
                raise ValueError(
                    f"custom_centroids has shape {custom_centroids.shape} but "
                    f"must be of shape (cells={cells}, len(ranges)="
                    f"{self._behavior_dim})")
            self._centroids = custom_centroids
            self._samples = None
        if self._centroids is None:
            self._samples = self._rng.uniform(
                self._lower_bounds,
                self._upper_bounds,
                size=(self._samples, self._behavior_dim),
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
        """(behavior_dim,) numpy.ndarray: Lower bound of each dimension."""
        return self._lower_bounds

    @property
    def upper_bounds(self):
        """(behavior_dim,) numpy.ndarray: Upper bound of each dimension."""
        return self._upper_bounds

    @property
    def samples(self):
        """(num_samples, behavior_dim) numpy.ndarray: The samples used in
        creating the CVT.

        May be None until :meth:`initialize` is called.
        """
        return self._samples

    @property
    def centroids(self):
        """(n_centroids, behavior_dim) numpy.ndarray: The centroids used in the
        CVT.

        None until :meth:`initialize` is called.
        """
        return self._centroids

    @staticmethod
    @jit(nopython=True)
    def _brute_force_nn_numba(measures_batch, centroids):
        """Calculates the nearest centroids to the given batch of measures with
        brute force."""

        # We want to take the difference between each measure i and all the
        # centroids.
        distances = np.expand_dims(measures_batch, axis=1) - centroids

        # Compute the total squared distance -- there is no need to compute the
        # actual distance with a sqrt.
        distances = np.sum(np.square(distances), axis=2)

        return distances

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
            measures_batch (array-like): (batch_size, :attr:`behavior_dim`)
                array of coordinates in measure space.
        Returns:
            numpy.ndarray: (batch_size,) array of centroid indices
            corresponding to each measure space coordinate.
        Raises:
            ValueError: ``measures_batch`` is not of shape (batch_size,
                :attr:`behavior_dim`).
        """
        measures_batch = np.asarray(measures_batch)
        check_measures_batch_shape(measures_batch, self.behavior_dim)

        if self._use_kd_tree:
            return np.asarray(
                self._centroid_kd_tree.query(measures_batch))[1].astype(
                    np.int32)

        # Older versions of Numba do not support the axis argument for argmin,
        # so the argmin is done here instead of in `_brute_force_nn_numba`.
        return np.argmin(
            self._brute_force_nn_numba(measures_batch, self.centroids),
            axis=1,
        ).astype(np.int32)
