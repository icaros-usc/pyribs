"""Gaussian Operator"""
import numpy as np

from ribs.emitters.operators._operator_base import OperatorBase


class GaussianOperator(OperatorBase):
    """Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
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

    def __init__(self, sigma, seed, lower_bounds, upper_bounds):

        self._sigma = sigma
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

        self._rng = np.random.default_rng(seed)

    def ask(self, parents):
        """Adds Gaussian noise to solution

         Args:
            parents (array-like): (batch_size, solution_dim) array of
                solutions to be mutated.

        Returns:
            numpy.ndarray: ``(batch_size, solution_dim)`` array that contains
            ``batch_size`` mutated solutions.
        """

        noise = self._rng.normal(
            scale=self._sigma,
            size=(parents.shape[0], parents.shape[1]),
        ).astype(float)

        return np.clip(parents + noise, self._lower_bounds, self._upper_bounds)
