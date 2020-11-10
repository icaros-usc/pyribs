"""Contains the SlidingBoundaryArchive and corresponding
SlidingBoundaryArchiveConfig."""

import numpy as np
import pandas as pd

from ribs.archives._archive_base import ArchiveBase
from ribs.config import create_config


class SlidingBoundaryArchiveConfig:
    """Configuration for the SlidingBoundaryArchive.

    Args:
        seed (float or int): Value to seed the random number generator. Set to
            None to avoid any seeding. Default: None
        remap_frequency (int): Frequency of remapping. Archive will remap once
            after ``remap_frequency`` number of solutions has been found.
    """

    def __init__(
        self,
        seed=None,
        remap_frequency=100,
    ):
        self.seed = seed
        self.remap_frequency = remap_frequency


class SlidingBoundaryArchive(ArchiveBase):
    """An archive that divides each dimension into a fixed number of bins with
    sliding boundaries that are placed at percentage marks of the behavior
    characteristics

    This archive is the container described in the Hearthstone Deck Space paper:
    https://arxiv.org/pdf/1904.10656.pdf. Same as the GridArchive, it can bes
    visualized as an n-dimensional grid in the behavior space that is divided
    into a certain number of bins in each dimension. However, it places the
    boundaries at the percentage marks of the behavior characteristics along
    each dimension. At a certain frequency, the archive will remap the boundary
    in accordance with all of the solutions found (Note: not only those already
    in the archive, but ALL of the solutions found by CMA-ME).

    This archive attempts to enable the distribution of the space illuminated
    by the archive to more accurately matches the true distribution if the
    behavior characteristics are not uniformly distributed.

    Args:
        dims (array-like): Number of bins in each dimension of the behavior
            space, e.g. ``[20, 30, 40]`` indicates there should be 3 dimensions
            with 20, 30, and 40 bins. (The number of dimensions is implicitly
            defined in the length of this argument).
        ranges (array-like of (float, float)): Upper and lower bound of each
            dimension of the behavior space, e.g. ``[(-1, 1), (-2, 2)]``
            indicates the first dimension should have bounds ``(-1, 1)``, and
            the second dimension should have bounds ``(-2, 2)``.
        config (None or dict or GridArchiveConfig): Configuration object. If
            None, a default GridArchiveConfig is constructed. A dict may also be
            passed in, in which case its arguments will be passed into
            GridArchiveConfig.
    Attributes:
        config (GridArchiveConfig): Configuration object.
        dims (np.ndarray): Number of bins in each dimension.
        n_dims (int): Number of dimensions.
        lower_bounds (np.ndarray): Lower bound of each dimension.
        upper_bounds (np.ndarray): Upper bound of each dimension.
        interval_size (np.ndarray): The size of each dimension (``upper_bounds -
            lower_bounds``).
        boundaries (list of np.ndarray): The dynamic boundaries of each
            dimension of the behavior space. The number of boundaries is
            determined by ``dims``.
        remap_frequency (int): Frequency of remapping. Archive will remap once
            after ``remap_frequency`` number of solutions has been found.
        all_solutions (list of np.ndarray): All solutions found by CMA-ME.
        all_behavior_values (list of np.ndarray): All behavior values found by
            CMA-ME.
        all_objective_values (list of np.ndarray): All objective values found
            by CMA-ME.
    """

    def __init__(self, dims, ranges, config=None):
        self.config = create_config(config, SlidingBoundaryArchiveConfig)
        self.dims = np.array(dims)
        self.n_dims = len(self.dims)
        ArchiveBase.__init__(
            self,
            n_dims=self.n_dims,
            objective_value_dim=self.dims,
            behavior_value_dim=(*self.dims, self.n_dims),
            solution_dim=self.dims,
            seed=self.config.seed,
        )

        ranges = list(zip(*ranges))
        self.lower_bounds = np.array(ranges[0])
        self.upper_bounds = np.array(ranges[1])
        self.interval_size = self.upper_bounds - self.lower_bounds

        # Sliding boundary specifics
        self.remap_frequency = self.config.remap_frequency
        self.boundaries = [
            np.full(self.dims[i], None, dtype=float) for i in range(self.n_dims)
        ]
        self.all_solutions = []
        self.all_behavior_values = []
        self.all_objective_values = []

    def _get_index(self, behavior_values):
        """Index is determined based on sliding boundaries
        """
        epsilon = 1e-9
        behavior_values = np.clip(behavior_values + epsilon, self.lower_bounds,
                                  self.upper_bounds - epsilon)

        index = []
        for i, behavior_value in enumerate(behavior_values):
            idx = 0
            while idx < self.dims[i] and \
                        self.boundaries[i][idx] < behavior_value:
                idx += 1
            index.append(idx)
        return np.max([0, index-1])

    def add(self, solution, objective_value, behavior_values):
        """ Remap the archive or attempt to insert the solution into the archive

        Args:
            solution (np.ndarray): Parameters for the solution.
            objective_value (float): Objective function evaluation of this
                solution.
            behavior_values (np.ndarray): Coordinates in behavior space of this
                solution.
        Returns:
            bool: Whether the value was inserted into the archive.
        """
        self.all_solutions.append(solution)
        self.all_behavior_values.append(behavior_values)
        self.all_objective_values.append(objective_value)

        if len(self.all_solutions) % self.remap_frequency == 1:
            self.re_map()
        else:
            ArchiveBase.add(self, solution, objective_value, behavior_values)

    def re_map(self,):
        """Remap the archive so that the boundaries locate at the percentage
        marks of the solutions stored in the archive.

        Re-add all of the solutions, (Note: not just those in the current
        archive, but ALL of the solutions encountered).
        """

        # sort all behavior values along the axis of each bc
        sorted_bc = np.sort(self.all_behavior_values, axis=0)

        for i in range(self.n_dims):
            for j in range(self.dims[i]):
                sample_idx = int(j * len(self.all_solutions) / self.dims[i])
                self.boundaries[i][j] = sorted_bc[i][sample_idx]

        # add all solutions to the new empty archive
        self._occupied_indices = []
        for solution, objective_value, behavior_value in zip(
                self.all_solutions, self.all_objective_values,
                self.all_behavior_values):
            ArchiveBase.add(self, solution, objective_value, behavior_value)

    def as_pandas(self):
        pass
