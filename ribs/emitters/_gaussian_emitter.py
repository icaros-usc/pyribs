"""Provides the GaussianEmitter."""
import numpy as np
from numba import jit

from ribs.emitters._emitter_base import EmitterBase


class GaussianEmitter(EmitterBase):
    """Emits solutions by adding Gaussian noise to existing archive solutions.

    If the archive is empty, calls to :meth:`ask` will generate solutions from a
    user-specified Gaussian distribution with mean ``x0`` and standard deviation
    ``sigma0``. Otherwise, this emitter selects solutions from the archive and
    generates solutions from a Gaussian distribution centered around each
    solution with standard deviation ``sigma0``.

    This is the classic variation operator presented in `Mouret 2015
    <https://arxiv.org/pdf/1504.04909.pdf>`_.

    Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        x0 (array-like): Center of the Gaussian distribution from which to
            sample solutions when the archive is empty.
        sigma0 (float or array-like): Standard deviation of the Gaussian
            distribution, both when the archive is empty and afterwards. Note we
            assume the Gaussian is diagonal, so if this argument is an array, it
            must be 1D.
        bounds (None or array-like): Bounds of the solution space. Solutions are
            clipped to these bounds. Pass None to indicate there are no bounds.

            Pass an array-like to specify the bounds for each dim. Each element
            in this array-like can be None to indicate no bound, or a tuple of
            ``(lower_bound, upper_bound)``, where ``lower_bound`` or
            ``upper_bound`` may be None to indicate no bound.
        batch_size (int): Number of solutions to send back in :meth:`ask`.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
    Raises:
        ValueError: There is an error in the bounds configuration.
    """

    def __init__(self,
                 archive,
                 x0,
                 sigma0,
                 bounds=None,
                 batch_size=64,
                 seed=None):
        self._x0 = np.array(x0, dtype=archive.dtype)
        self._sigma0 = archive.dtype(sigma0) if isinstance(
            sigma0, (float, np.floating)) else np.array(sigma0)

        EmitterBase.__init__(
            self,
            archive,
            len(self._x0),
            bounds,
            batch_size,
            seed,
        )

    @property
    def x0(self):
        """numpy.ndarray: Center of the Gaussian distribution from which to
        sample solutions when the archive is empty."""
        return self._x0

    @property
    def sigma0(self):
        """float or numpy.ndarray: Standard deviation of the (diagonal) Gaussian
        distribution."""
        return self._sigma0

    @staticmethod
    @jit(nopython=True)
    def _ask_clip_helper(parents, noise, lower_bounds, upper_bounds):
        """Numba equivalent of np.clip."""
        return np.minimum(np.maximum(parents + noise, lower_bounds),
                          upper_bounds)

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
        if self._archive.empty:
            parents = np.expand_dims(self._x0, axis=0)
        else:
            parents = [
                self._archive.get_random_elite()[0]
                for _ in range(self.batch_size)
            ]

        noise = self._rng.normal(
            scale=self._sigma0,
            size=(self.batch_size, self.solution_dim),
        ).astype(self._archive.dtype)

        return self._ask_clip_helper(np.asarray(parents), noise,
                                     self._lower_bounds, self._upper_bounds)
