"""Iso Line Operator"""
import numpy as np

from ribs.emitters.operators._operator_base import OperatorBase


class IsoLineOperator(OperatorBase):
    """Adds Isotropic Gaussian noise and directional noise to parents.

    This operator was introduced in `Vassiliades 2018
    <https://arxiv.org/abs/1804.03906>`_.

    Args:
        iso_sigma (float): Scale factor for the isotropic distribution used to
            generate solutions.
        line_sigma (float): Scale factor for the line distribution used when
            generating solutions.
        sigma (float or array-like): Standard deviation of the Gaussian
            distribution. Note we assume the Gaussian is diagonal, so if this
            argument is an array, it must be 1D.
        lower_bounds (array-like): Upper bounds of the solution space. Passed in
            by emitter
        upper_bounds (array-like): Upper bounds of the solution space. Passed in
            by emitter
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
    """

    def __init__(self,
                 lower_bounds,
                 upper_bounds,
                 iso_sigma,
                 line_sigma,
                 seed=None):
        self._iso_sigma = iso_sigma
        self._line_sigma = line_sigma
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

        self._rng = np.random.default_rng(seed)

    @property
    def parent_type(self):
        """int: Parent Type to be used by selector."""
        return 2

    def ask(self, parents):
        """ Adds Isotropic Guassian noise and directional noise to parents.

        Args:
            parents (array-like): (2, batch_size, solution_dim) parents[0] array
                of solutions selected by emitter parents[1] array of second
                batch of solutions passed by emitter. Used for calculating
                directional correlation.

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

        return np.clip(solution_batch, self._lower_bounds, self._upper_bounds)
