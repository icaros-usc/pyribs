"""Iso Line Operator"""
import numpy as np

from ribs.emitters.operators._operator_base import OperatorBase


class IsoLineOperator(OperatorBase):
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

    def __init__(self,
                 iso_sigma,
                 line_sigma,
                 lower_bounds,
                 upper_bounds,
                 seed=None):
        self._iso_sigma = iso_sigma
        self._line_sigma = line_sigma
        self._rng = np.random.default_rng(seed)
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

    def operate(self, parents, directions):
        """ Adds isotropic Guassian noise and directional noise to an elite

         Args:
            parents (np-array): (batch_size, :attr:`solution_dim`)
                array of solutions selected by emitter
            directions (np-array): (batch_size, :attr:`solution_dim`)
                array of directions to random elites selected by emitter

        Returns:
            ``(batch_size, solution_dim)`` array
            -- contains ``batch_size`` mutated solutions
        """

        iso_gaussian = self._rng.normal(
            scale=self._iso_sigma,
            size=(parents.shape[0], parents.shape[1]),
        ).astype(float)

        line_gaussian = self._rng.normal(
            scale=self._line_sigma,
            size=(parents.shape[0], 1),
        ).astype(float)
        solution_batch = parents + iso_gaussian + line_gaussian * directions

        return np.clip(solution_batch, self._lower_bounds, self._upper_bounds)
