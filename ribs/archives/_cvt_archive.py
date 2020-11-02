"""Contains the CVTArchive class."""
import numpy as np
import pandas as pd
from scipy.cluster import vq

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
    """

    def __init__(
        self,
        seed=None,
        samples=100_000,
        k_means_threshold=1e-6,
    ):
        self.seed = seed
        self.samples = samples
        self.k_means_threshold = k_means_threshold


class CVTArchive(ArchiveBase):
    """An archive that divides the space into a fixed number of bins.

    This archive originates in the CVT-MAP-Elites paper
    https://ieeexplore.ieee.org/document/8000667. It uses Centroidal Voronoi
    Tesselation (CVT) to divide an n-dimensional behavior space into k bins. The
    CVT is created by sampling points uniformly from the n-dimensional behavior
    space and using k-means clustering to identify k centroids. When items are
    inserted into the archive, we identify their bin by identifying the closest
    centroid in behavior space (using Euclidean distance).

    Currently, finding the closes centroid is implemented as an O(k) search,
    though we are considering implementing an O(log k) search in the future.

    For k-means clustering, note that we use `scipy.cluster.vq
    <https://docs.scipy.org/doc/scipy/reference/cluster.vq.html>`_

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
        n_dims (int): Number of dimensions in the archive.
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
        initial_centroids = self._rng.uniform(
            self.lower_bounds,
            self.upper_bounds,
            size=(bins, self.n_dims),
        )
        self.centroids = vq.kmeans(
            self.samples,
            initial_centroids,
            iter=1,
            thresh=self.config.k_means_threshold,
        )[0]

    def _get_index(self, behavior_values):
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
