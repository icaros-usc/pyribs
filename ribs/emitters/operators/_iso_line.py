"""Provides IsoLineOperator."""

import numpy as np

from ribs.emitters.operators._operator_base import OperatorBase


class IsoLineOperator(OperatorBase):
    """Adds Isotropic Gaussian noise and directional noise to parents.

    This operator was introduced in `Vassiliades 2018
    <https://arxiv.org/abs/1804.03906>`_.

    Args:
        iso_sigma (float): Scale factor for the isotropic distribution used to generate
            solutions.
        line_sigma (float): Scale factor for the line distribution used when generating
            solutions.
        seed (int): Value to seed the random number generator. Set to None to avoid a
            fixed seed.
    """

    def __init__(self, iso_sigma, line_sigma, seed=None):
        self._iso_sigma = iso_sigma
        self._line_sigma = line_sigma
        self._rng = np.random.default_rng(seed)

    @property
    def parent_type(self):
        """int: Parent Type to be used by selector."""
        return 2

    def ask(self, parents):
        """Adds Isotropic Guassian noise and directional noise to parents.

        Args:
            parents (array-like): (2, batch_size, solution_dim) parents[0] array of
                solutions selected by emitter parents[1] array of second batch of
                solutions passed by emitter. Used for calculating directional
                correlation.

        Returns:
            numpy.ndarray: ``(batch_size, solution_dim)`` array that contains
            ``batch_size`` mutated solutions.
        """
        parents = np.asarray(parents)

        elites = parents[0]
        directions = parents[1] - parents[0]

        iso_gaussian = self._rng.normal(
            scale=self._iso_sigma,
            size=(elites.shape[0], elites.shape[1]),
        ).astype(elites.dtype)

        line_gaussian = self._rng.normal(
            scale=self._line_sigma,
            size=(elites.shape[0], 1),
        ).astype(elites.dtype)
        solution_batch = elites + iso_gaussian + line_gaussian * directions

        return solution_batch
