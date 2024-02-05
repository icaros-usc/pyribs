"""Gaussian Operator"""
import numpy as np

from ribs.emitters.operators._operator_base import OperatorBase


class GaussianOperator(OperatorBase):
    """Adds Gaussian noise to solutions.

    Args:
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

    def __init__(self, sigma, lower_bounds, upper_bounds, seed=None):
        self._sigma = sigma
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

        self._rng = np.random.default_rng(seed)

    @property
    def parent_type(self):
        """int: Parent Type to be used by selector."""
        return 1

    def ask(self, parents):
        """Adds Gaussian noise to parents.

        Args:
            parents (array-like): (batch_size, solution_dim) array of
                solutions to be mutated.

        Returns:
            numpy.ndarray: ``(batch_size, solution_dim)`` array that contains
            ``batch_size`` mutated solutions.
        """
        parents = np.asarray(parents)

        noise = self._rng.normal(
            scale=self._sigma,
            size=(parents.shape[0], parents.shape[1]),
        ).astype(parents.dtype)

        return np.clip(parents + noise, self._lower_bounds, self._upper_bounds)
