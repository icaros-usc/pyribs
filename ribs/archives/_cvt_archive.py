"""Contains the CVTArchive class."""
import numpy as np
import pandas as pd
from numba import jit
from scipy.cluster.vq import kmeans
from scipy.spatial import cKDTree  # pylint: disable=no-name-in-module

from ribs.archives._archive_base import ArchiveBase, require_init


class CVTArchive(ArchiveBase):
    """Divides the entire behavior space into a fixed number of bins.

    This archive originates in the `CVT-MAP-Elites paper
    <https://ieeexplore.ieee.org/document/8000667>`_. It uses Centroidal Voronoi
    Tesselation (CVT) to divide an n-dimensional behavior space into k bins. The
    CVT is created by sampling points uniformly from the n-dimensional behavior
    space and using k-means clustering to identify k centroids. When items are
    inserted into the archive, we identify their bin by identifying the closest
    centroid in behavior space (using Euclidean distance). For k-means
    clustering, we use `scipy.cluster.vq
    <https://docs.scipy.org/doc/scipy/reference/cluster.vq.html>`_.

    Finding the closest centroid is done in O(bins) time (i.e. brute force) by
    default. If you set ``use_kd_tree``, it can be done in roughly
    O(log bins) time using `scipy.spatial.cKDTree
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html>`_.
    However, using this k-D tree lowers performance for small numbers of bins.
    The following plot compares the runtime of brute force vs k-D tree when
    inserting 100k samples into a 2D archive with varying numbers of bins (we
    took the minimum over 5 runs for each data point, as recommended `here
    <https://docs.python.org/3/library/timeit.html#timeit.Timer.repeat>`__).
    Note the logarithmic scales. This plot was generated on a reasonably modern
    laptop.

    .. image:: _static/imgs/cvt_add_plot.png
        :alt: Runtime to insert 100k entries into CVTArchive

    As we can see, archives with at least 5k bins seem to have faster insertion
    when using a k-D tree than when using brute force, so **we recommend
    setting** ``use_kd_tree`` **if you have at least 5k bins in
    your** ``CVTArchive``. See `benchmarks/cvt_add.py
    <https://github.com/icaros-usc/pyribs/tree/master/benchmarks/cvt_add.py>`_
    in the project repo for more information about how this plot was generated.

    .. note:: If you are attempting to use CVTArchive deterministically, you
        will also need to seed numpy's global random number generator with::

            np.random.seed(seed)

        This will ensure that scipy's k-means clustering runs deterministically.

    Args:
        ranges (array-like of (float, float)): Upper and lower bound of each
            dimension of the behavior space, e.g. ``[(-1, 1), (-2, 2)]``
            indicates the first dimension should have bounds ``(-1, 1)``, and
            the second dimension should have bounds ``(-2, 2)``. Note that the
            length of this array defines the dimensionality of the behavior
            space.
        bins (int): The number of bins to use in the archive, equivalent to the
            number of areas in the CVT.
        samples (int or array-like): If it is an int, this specifies the number
            of samples to generate when creating the CVT. Otherwise, this must
            be a (num_samples, behavior_dim) array where samples[i] is a sample
            to use when creating the CVT. It can be useful to pass in custom
            samples when there are restrictions on what samples in the behavior
            space are possible.
        k_means_threshold (float): When finding the centroids at the beginning,
            k-means will terminate when the difference in distortion between
            iterations goes below this threshold (see `here
            <https://docs.scipy.org/doc/scipy/reference/cluster.vq.html>`__ for
            more info).
        use_kd_tree (bool): If True, use a k-D tree for finding the closest
            centroid when inserting into the archive. This may result in a
            speedup for larger dimensions; refer to
            :class:`~ribs.archives.CVTArchive` for more info.
        seed (float or int): Value to seed the random number generator. Set to
            None to avoid any seeding.
    """

    def __init__(self,
                 ranges,
                 bins,
                 samples=100_000,
                 k_means_threshold=1e-6,
                 use_kd_tree=False,
                 seed=None):
        ArchiveBase.__init__(
            self,
            storage_dims=(bins,),
            behavior_dim=len(ranges),
            seed=seed,
        )

        ranges = list(zip(*ranges))
        self._lower_bounds = np.array(ranges[0])
        self._upper_bounds = np.array(ranges[1])

        self._bins = bins
        self._k_means_threshold = k_means_threshold
        self._use_kd_tree = use_kd_tree
        self._samples = samples
        self._centroids = None
        self._centroid_kd_tree = None

    @property
    def lower_bounds(self):
        """(behavior_dim,) np.ndarray: Lower bound of each dimension."""
        return self._lower_bounds

    @property
    def upper_bounds(self):
        """(behavior_dim,) np.ndarray: Upper bound of each dimension."""
        return self._upper_bounds

    @property
    def samples(self):
        """(num_samples, behavior_dim) np.ndarray: The samples used in creating
        the CVT. May be None until :meth:`initialize` is called."""
        return self._samples

    @property
    def centroids(self):
        """(num_centroids, behavior_dim) np.ndarray: The centroids used in the
        CVT. None until :meth:`initialize` is called."""
        return self._centroids

    def initialize(self, solution_dim):
        """Initializes the archive.

        This method may take a while to run. In addition to allocating storage
        space, it runs k-means to create an approximate CVT, and it may create a
        k-D tree containing the centroids found by k-means.

        Args:
            solution_dim (int): The dimension of the solution space. The array
                for storing solutions is created with shape
                ``(*self._storage_dims, solution_dim)``.
        """
        ArchiveBase.initialize(self, solution_dim)

        self._samples = self._rng.uniform(
            self._lower_bounds,
            self._upper_bounds,
            size=(self._samples, self._behavior_dim),
        ) if isinstance(self._samples, int) else np.array(self._samples)

        self._centroids = kmeans(
            self._samples,
            self._bins,
            iter=1,
            thresh=self._k_means_threshold,
        )[0]

        if self._use_kd_tree:
            self._centroid_kd_tree = cKDTree(self._centroids)

    @staticmethod
    @jit(nopython=True)
    def _brute_force_nn_numba(behavior_values, centroids):
        """Calculates the nearest centroid to the given behavior values.

        Technically, we calculate squared distance, but we only care about
        finding the neighbor and not the distance itself.
        """
        distances = np.expand_dims(behavior_values, axis=0) - centroids
        distances = np.sum(np.square(distances), axis=1)
        return np.argmin(distances)

    def _get_index(self, behavior_values):
        """Finds the centroid index using either the k-D tree or brute force.

        :meta private:
        """
        if self._use_kd_tree:
            return self._centroid_kd_tree.query(behavior_values)[1]

        return self._brute_force_nn_numba(behavior_values, self._centroids)

    @require_init
    def as_pandas(self):
        """Converts the archive into a Pandas dataframe.

        Returns:
            A dataframe where each row is an elite in the archive. The dataframe
            consists of 1 ``index`` column indicating the index of the centroid
            in ``self._centroids``, ``behavior_dim`` columns called
            ``behavior-{i}`` for the behavior values, 1 column for the objective
            function value called ``objective``, and ``solution_dim`` columns
            called ``solution-{i}`` for the solution values.
        """
        data = {"index": self._occupied_indices}

        behavior_values = self._behavior_values[self._occupied_indices]
        for i in range(self._behavior_dim):
            data[f"behavior-{i}"] = behavior_values[:, i]

        data["objective"] = self._objective_values[self._occupied_indices]

        solutions = self._solutions[self._occupied_indices]
        for i in range(self._solution_dim):
            data[f"solution-{i}"] = solutions[:, i]

        return pd.DataFrame(data)
