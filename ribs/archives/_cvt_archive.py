"""Contains the CVTArchive class."""

from collections import namedtuple

import numpy as np

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


class CVTArchive:

    def __init__(self, ranges, clusters, samples=None, config=None):
        self.config = create_config(config, CVTArchiveConfig)

        ranges = list(zip(*ranges))
        self.lower_bounds = np.array(ranges[0])
        self.upper_bounds = np.array(ranges[1])

        self._rng = np.random.default_rng(self.config.seed)

        dim = self.lower_bounds.shape[0]
        self.samples = samples if samples is not None else self._rng.uniform(
            self.lower_bounds,
            self.upper_bounds,
            size=(self.config.samples, dim),
        )

        initial_centroids = self._rng.uniform(
            self.lower_bounds,
            self.upper_bounds,
            size=(clusters, dim),
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
            centroids ((n_centroids, dim) array): Initial centroids.
            points ((n_points, dim) array): Points to cluster.
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
            # `distances` an (n_points, dim, n_clusters) array -- we want to
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

    def add(self):
        pass

    def is_empty(self):
        pass

    def get_random_elite(self):
        pass

    def as_pandas(self):
        pass

    # TODO: Add voronoi method
