"""Contains the CVTArchive class."""
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from scipy.spatial import Voronoi  # pylint: disable=no-name-in-module

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

    # TODO: figure out how to pass in ax parameter.
    # TODO: move somewhere else.
    def heatmap(self,
                filename=None,
                plot_samples=False,
                figsize=(8, 6),
                colormap="magma"):
        """Plots heatmap of the 2D archive and saves it to a file.

        Essentially, we create a Voronoi diagram and shade in each cell with a
        color corresponding to the value of that cell's elite.

        Raises:
            RuntimeError: The archive is not 2D.
        """
        if self._n_dims != 2:
            raise RuntimeError("Cannot plot heatmap for non-2D archive.")

        colormap = plt.get_cmap(colormap)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")
        ax.set_xlim(self.lower_bounds[0], self.upper_bounds[0])
        ax.set_ylim(self.lower_bounds[1], self.upper_bounds[1])

        # Add faraway points so that the edge regions of the Voronoi diagram are
        # filled in. Refer to
        # https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram
        # for more info.
        interval = self.upper_bounds - self.lower_bounds
        scale = 1000
        faraway_pts = [
            self.upper_bounds + interval * scale,  # Far upper right.
            self.upper_bounds + interval * [-1, 1] * scale,  # Far upper left.
            self.lower_bounds + interval * [-1, -1] * scale,  # Far bottom left.
            self.lower_bounds + interval * [1, -1] * scale,  # Far bottom right.
        ]
        vor = Voronoi(np.append(self.centroids, faraway_pts, axis=0))

        # Calculate objective value for each region. `vor.point_region` contains
        # the region index of each point.
        region_obj = [None] * len(vor.regions)
        min_obj, max_obj = np.inf, np.NINF
        for index, region_idx in enumerate(
                vor.point_region[:-4]):  # Exclude faraway_pts.
            if region_idx != -1 and self._solutions[index] is not None:
                obj = self._objective_values[index]
                min_obj = min(min_obj, obj)
                max_obj = max(max_obj, obj)
                region_obj[region_idx] = obj

        # Shade the regions.
        for region, objective in zip(vor.regions, region_obj):
            # This check is O(n), but n is typically small, and creating
            # `polygon` is also O(n) anyway.
            if -1 not in region:
                if objective is None:
                    color = "white"
                else:
                    normalized_obj = (objective - min_obj) / (max_obj - min_obj)
                    color = colormap(normalized_obj)
                polygon = [vor.vertices[i] for i in region]
                ax.fill(*zip(*polygon), color=color, ec="k", lw=0.5)
        mappable = ScalarMappable(cmap=colormap)
        mappable.set_clim(min_obj, max_obj)
        fig.colorbar(mappable, ax=ax, pad=0.1)

        # Plot the sample points and centroids.
        if plot_samples:
            ax.plot(self.samples[:, 0], self.samples[:, 1], "o", c="gray", ms=1)
        ax.plot(self.centroids[:, 0], self.centroids[:, 1], "ko")

        if filename is not None:
            fig.savefig(filename)

        return fig, ax
