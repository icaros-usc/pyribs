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

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        class_args = [
            'iso_sigma', 'line_sigma', 'lower_bounds', 'upper_bounds', 'seed'
        ]

        if not all(arg in kwargs for arg in class_args):
            raise ValueError(
                "IsoLine Operator initialization arguments must be provided.")

        self._rng = np.random.default_rng(self.seed)

    def ask(self, **kwargs):
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

        if ('parents' not in kwargs or 'directions' not in kwargs):
            raise ValueError("Parents and directions must be provided.")

        parents = kwargs['parents']
        directions = kwargs['directions']

        iso_gaussian = self._rng.normal(
            scale=self.iso_sigma,
            size=(parents.shape[0], parents.shape[1]),
        ).astype(float)

        line_gaussian = self._rng.normal(
            scale=self.line_sigma,
            size=(parents.shape[0], 1),
        ).astype(float)
        solution_batch = parents + iso_gaussian + line_gaussian * directions

        return np.clip(solution_batch, self.lower_bounds, self.upper_bounds)
