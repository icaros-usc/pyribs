"""Provides the GaussianEmitter and corresponding GaussianEmitterConfig."""
import numpy as np

from ribs.config import create_config
from ribs.emitters._emitter_base import EmitterBase


class GaussianEmitterConfig:
    """Configuration for the GaussianEmitter.

    Args:
        seed (float or int): Value to seed the random number generator. Set to
            None to avoid seeding. Default: None
        batch_size (int): Number of solutions to send back in the ask() method.
            Default: 64
    """

    def __init__(
        self,
        seed=None,
        batch_size=64,
    ):
        self.seed = seed
        self.batch_size = batch_size


class GaussianEmitter(EmitterBase):
    """Emits solutions by adding Gaussian noise to existing archive solutions.

    If the archive is empty, calls to ask() will generate solutions from a
    user-specified Gaussian distribution with mean ``x0`` and standard deviation
    ``sigma0``. Otherwise, this emitter selects solutions from the archive and
    generates solutions from a Gaussian distribution centered around each
    solution with standard deviation ``sigma0``.

    Args:
        x0 (array-like): Center of the Gaussian distribution from which to
            sample solutions when the archive is empty.
        sigma0 (float or array-like): Standard deviation of the Gaussian
            distribution, both when the archive is empty and afterwards. Note we
            assume the Gaussian is diagonal, so if this argument is an array, it
            must be 1D.
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        config (None or dict or GaussianEmitterConfig): Configuration object. If
            None, a default GaussianEmitterConfig is constructed. A dict may
            also be passed in, in which case its arguments will be passed into
            GaussianEmitterConfig.
    """

    def __init__(self, x0, sigma0, archive, config=None):
        self._config = create_config(config, GaussianEmitterConfig)
        self._x0 = np.array(x0)
        self._sigma0 = sigma0 if isinstance(sigma0, float) else np.array(sigma0)

        EmitterBase.__init__(self, len(self._x0), self._config.batch_size,
                             archive, self._config.seed)

    @property
    def config(self):
        """GaussianEmitterConfig: Configuration object."""
        return self._config

    @property
    def x0(self):
        """np.ndarray: Center of the Gaussian distribution from which to sample
        solutions when the archive is empty."""
        return self._x0

    @property
    def sigma0(self):
        """float or np.ndarray: Standard deviation of the (diagonal) Gaussian
        distribution."""
        return self._sigma0

    def ask(self):
        """Creates solutions by adding Gaussian noise to elites in the archive.

        If the archive is empty, solutions are drawn from a (diagonal) Gaussian
        distribution centered at ``self.x0``. Otherwise, each solution is drawn
        from a distribution centered at a randomly chosen elite. In either case,
        the standard deviation is ``self.sigma0``.

        Returns:
            ``(self.batch_size, self.solution_dim)`` array -- contains
            ``batch_size`` new solutions to evaluate.
        """
        if self._archive.is_empty():
            parents = np.expand_dims(self._x0, axis=0)
        else:
            parents = [
                self._archive.get_random_elite()[0]
                for _ in range(self.batch_size)
            ]

        return parents + self._rng.normal(
            scale=self._sigma0, size=(self.batch_size, self.solution_dim))
