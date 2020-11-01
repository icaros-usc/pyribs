"""Contains the CVTArchive class."""
from collections import namedtuple

import numpy as np
import pandas as pd

from ribs.archives._archive_base import ArchiveBase
from ribs.config import create_config

#: Configuration for the CVTArchive.
#:
#: Attributes:
#:     seed (float or int): Value to seed the random number generator. Set to
#:         None to avoid any seeding. Default: None
#:     samples (int): Number of samples to generate before creating the archive.
#:         If ``samples`` is passed into ``CVTArchive``, this option is ignored.
#:     k_means_threshold (float): When running k-means to find the centroids
#:         during initialization, we will calculate the cost of the clustering
#:         at each iteration, where the cost is the sum of the distances from
#:         each point to its centroid. When the change in cost between
#:         iterations goes below this threshold, k-means will terminate. By
#:         default, this is set to 1e-9 to facilitate convergence (i.e. the
#:         centroids no longer move between iterations.
#:     k_means_itrs (int): If you prefer to run k-means for a set number of
#:         iterations at the beginning, set this value. By default, it is None,
#:         which means that ``k_means_threshold`` is used instead. If provided,
#:         this value will have precedence over ``k_means_threshold``.
CVTArchiveConfig = namedtuple("CVTArchiveConfig", [
    "seed",
    "samples",
    "k_means_threshold",
    "k_means_itrs",
])
CVTArchiveConfig.__new__.__defaults__ = (
    None,
    100000,
    1e-9,
    None,
)


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
        config (CVTArchiveConfig): Configuration object. If None, a default
            CVTArchiveConfig is constructed. A dict may also be passed in, in
            which case its arguments will be passed into CVTArchiveConfig.
    Attributes:
        lower_bounds (np.ndarray): Lower bound of each dimension.
        upper_bounds (np.ndarray): Upper bound of each dimension.
        samples: The samples used in creating the CVT.
        centroids: The centroids used in the CVT.
    """

    def __init__(self, ranges, bins, samples=None, config=None):
        self.config = create_config(config, CVTArchiveConfig)
        dims = len(ranges)
        ArchiveBase.__init__(
            self,
            dims,  # n_dims
            bins,  # objective_value_dim
            (bins, dims),  # behavior_value_dim
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
                            size=(self.config.samples, dims),
                        ))
        initial_centroids = self._rng.uniform(
            self.lower_bounds,
            self.upper_bounds,
            size=(bins, dims),
        )
        self.centroids = self._k_means_cluster(
            initial_centroids,
            self.samples,
            self.config.k_means_threshold,
            self.config.k_means_itrs,
        )

    @staticmethod
    def _k_means_cluster(centroids, points, threshold, total_itrs):
        """Clusters the given points, uses the centroids to initialize.

        Args:
            centroids ((n_centroids, n_dims) array): Initial centroids.
            points ((n_points, n_dims) array): Points to cluster.
            threshold: See ``CVTArchiveConfig.k_means_threshold``.
            total_itrs: See ``CVTArchiveConfig.k_means_itrs``.
        """
        previous_cost = None
        itrs = 0

        while True:
            # Check termination with itrs.
            if total_itrs is not None and itrs >= total_itrs:
                break

            # Calculate distance between centroids and points. Start by making
            # `distances` an (n_points, n_dims, n_clusters) array -- we want to
            # subtract every point in centroids from every point in points.
            distances = np.expand_dims(points, axis=2) - np.expand_dims(
                centroids.T, axis=0)
            # Now we square distances and sum it along axis 1 to get an
            # (n_points, n_centroids) array where (i,j) is the distance from
            # point i to centroid j.
            distances = np.sum(np.square(distances), axis=1)

            # Check termination with threshold.
            current_cost = np.sum(distances)
            if (total_itrs is None and previous_cost is not None and
                    abs(previous_cost - current_cost) < threshold):
                break
            previous_cost = current_cost

            # Find the closest centroid for each point.
            closest = np.argmin(distances, axis=1)

            # Indices of the points (in `points`) that belong to each centroid.
            centroid_pts = [[] for _ in range(len(centroids))]
            for pt_idx, centroid_idx in enumerate(closest):
                centroid_pts[centroid_idx].append(pt_idx)

            # Reassign the centroids.
            for centroid_idx, pts_idx in enumerate(centroid_pts):
                if len(pts_idx) != 0:  # No reassignment if no points assigned.
                    centroids[centroid_idx] = np.mean(points[pts_idx], axis=0)

            itrs += 1

        return centroids

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
            *[f"centroid-{i}" for i in range(self._n_dims)],
            *[f"behavior-{i}" for i in range(self._n_dims)],
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
