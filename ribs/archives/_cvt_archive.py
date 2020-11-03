"""Contains the CVTArchive class."""
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans
from scipy.spatial import KDTree

from ribs.archives._archive_base import ArchiveBase
from ribs.config import create_config


class CVTArchiveConfig:
    """Configuration for the CVTArchive.

    Args:
        seed (float or int): Value to seed the random number generator. Set to
            None to avoid any seeding. Default: None
        samples (int): Number of samples to generate before creating the
            archive.  If ``samples`` is passed into ``CVTArchive``, this option
            is ignored.
        k_means_threshold (float): When finding the centroids at the beginning,
            k-means will terminate when the difference in distortion between
            iterations goes below this threshold (see `here
            <https://docs.scipy.org/doc/scipy/reference/cluster.vq.html>`_) for
            more info.
        use_kd_tree (bool): If True, use a KDTree for finding the closest
            centroid when inserting into the archive. This may result in a
            speedup for larger dimensions; refer to
            :class:`~ribs.archives.CVTArchive` for more info.
    """

    def __init__(
        self,
        seed=None,
        samples=100_000,
        k_means_threshold=1e-6,
        use_kd_tree=False,
    ):
        self.seed = seed
        self.samples = samples
        self.k_means_threshold = k_means_threshold
        self.use_kd_tree = use_kd_tree


class CVTArchive(ArchiveBase):
    """An archive that divides the space into a fixed number of bins.

    This archive originates in the CVT-MAP-Elites paper
    https://ieeexplore.ieee.org/document/8000667. It uses Centroidal Voronoi
    Tesselation (CVT) to divide an n-dimensional behavior space into k bins. The
    CVT is created by sampling points uniformly from the n-dimensional behavior
    space and using k-means clustering to identify k centroids. When items are
    inserted into the archive, we identify their bin by identifying the closest
    centroid in behavior space (using Euclidean distance). For k-means
    clustering, note that we use `scipy.cluster.vq
    <https://docs.scipy.org/doc/scipy/reference/cluster.vq.html>`_.

    Finding the closest centroid is done in O(bins) time (i.e. brute force) by
    default. If the config has ``use_kd_tree`` set, it can be done in roughly
    O(log bins) time using `scipy.spatial.KDTree
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html>`_.
    However, using KDTree actually lowers performance for small numbers of bins.
    The following plot compares the runtime of brute force and KDTree when
    inserting 100k samples into a 2D archive with varying numbers of bins (we
    took the minimum over 5 runs for each data point, as recommended `here
    <https://docs.python.org/3/library/timeit.html#timeit.Timer.repeat>`_). Note
    the logarithmic x-axis. This plot was generated on a reasonably modern
    laptop.

    .. image:: _static/imgs/cvt_add_plot.png
        :alt: Runtime to insert 100k entries into CVTArchive

    As we can see, archives with at least 10k bins seem to have faster insertion
    when using KDTree than when using brute force, so **we recommend setting**
    ``use_kd_tree`` **in your config only if you have at least 10k bins in
    your** ``CVTArchive``. See `examples/performance/cvt_add.py
    <https://github.com/icaros-usc/pyribs/tree/master/examples/performance/cvt_add.py>`_
    in the project repo for more information about how this plot was generated.

    Args:
        ranges (array-like of (float, float)): Upper and lower bound of each
            dimension of the behavior space, e.g. ``[(-1, 1), (-2, 2)]``
            indicates the first dimension should have bounds ``(-1, 1)``, and
            the second dimension should have bounds ``(-2, 2)``. Note that the
            length of this array defines the dimensionality of the behavior
            space.
        bins (int): The number of bins to use in the archive, equivalent to the
            number of areas in the CVT.
        samples (array-like): A (num_samples, n_dims) array where samples[i] is
            a sample to use when creating the CVT. These samples may be passed
            in instead of generating the samples uniformly at random (u.a.r.) --
            this can be useful when, for instance, samples generated u.a.r. in
            the behavior space are not physically possible, such as in the case
            of trajectories represented by a series of points.
        config (None or dict or CVTArchiveConfig): Configuration object. If
            None, a default CVTArchiveConfig is constructed. A dict may also be
            passed in, in which case its arguments will be passed into
            CVTArchiveConfig.
    Attributes:
        config (CVTArchiveConfig): Configuration object.
        n_dims (int): Number of dimensions of the archive behavior space.
        lower_bounds (np.ndarray): Lower bound of each dimension.
        upper_bounds (np.ndarray): Upper bound of each dimension.
        samples: The samples used in creating the CVT.
        centroids: The centroids used in the CVT.
    """

    def __init__(self, ranges, bins, samples=None, config=None):
        self.config = create_config(config, CVTArchiveConfig)
        self.n_dims = len(ranges)
        ArchiveBase.__init__(
            self,
            bins,  # objective_value_dim
            (bins, self.n_dims),  # behavior_value_dim
            bins,  # solution_dim
            self.config.seed,
        )

        ranges = list(zip(*ranges))
        self.lower_bounds = np.array(ranges[0])
        self.upper_bounds = np.array(ranges[1])

        self.samples = (np.array(samples)
                        if samples is not None else self._rng.uniform(
                            self.lower_bounds,
                            self.upper_bounds,
                            size=(self.config.samples, self.n_dims),
                        ))
        self.centroids = kmeans(
            self.samples,
            bins,
            iter=1,
            thresh=self.config.k_means_threshold,
        )[0]

        if self.config.use_kd_tree:
            self._centroid_kd_tree = KDTree(self.centroids)

    def _get_index(self, behavior_values):
        if self.config.use_kd_tree:
            return self._centroid_kd_tree.query(behavior_values)[1]

        # Default: calculate nearest neighbor with brute force.
        distances = np.expand_dims(behavior_values, axis=0) - self.centroids
        distances = np.sum(np.square(distances), axis=1)
        return np.argmin(distances)

    def as_pandas(self):
        """Converts the archive into a Pandas dataframe.

        Returns:
            A dataframe where each row is an elite in the archive. The dataframe
            consists of 1 ``index`` column indicating the index of the centroid
            in ``self.centroids``, ``n_dims`` columns called ``centroid-{i}``
            for the coordinates of the centroid, ``n_dims`` columns called
            ``behavior-{i}`` for the behavior values, 1 column for the objective
            function value called ``objective``, and 1 column for solution
            objects called ``solution``.
        """
        column_titles = [
            "index",
            *[f"centroid-{i}" for i in range(self.n_dims)],
            *[f"behavior-{i}" for i in range(self.n_dims)],
            "objective",
            "solution",
        ]

        rows = []
        for index in self._occupied_indices:
            row = [
                index,
                *self.centroids[index],
                *self._behavior_values[index],
                self._objective_values[index],
                self._solutions[index],
            ]
            rows.append(row)

        return pd.DataFrame(rows, columns=column_titles)
