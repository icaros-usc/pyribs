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

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        class_args = ['sigma', 'seed', 'lower_bounds', 'upper_bounds']

        if not all(arg in kwargs for arg in class_args):
            raise ValueError(
                "Gaussian Operator initialization arguments must be provided.")

        self._rng = np.random.default_rng(self.seed)

    def ask(self, **kwargs):
        """Adds Gaussian noise to solution

         Args:
            parents (np-array): (batch_size, :attr:`solution_dim`)
                array of solutions selected by emitter

        Returns:
            ``(batch_size, solution_dim)`` array
            -- contains ``batch_size`` mutated solutions
        """
        if 'parents' not in kwargs:
            raise ValueError("Parents must be provided.")

        parents = kwargs['parents']

        noise = self._rng.normal(
            scale=self.sigma,
            size=(parents.shape[0], parents.shape[1]),
        ).astype(float)

        return np.clip(parents + noise, self.lower_bounds, self.upper_bounds)
