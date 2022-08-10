"""Provides the GaussianEmitter."""
import numpy as np

from ribs.emitters._emitter_base import EmitterBase


class GaussianEmitter(EmitterBase):
    """Emits solutions by adding Gaussian noise to existing archive solutions.

    If the archive is empty, calls to :meth:`ask` will generate solutions from a
    user-specified Gaussian distribution with mean ``x0`` and standard deviation
    ``sigma0``. Otherwise, this emitter selects solutions from the archive and
    generates solutions from a Gaussian distribution centered around each
    solution with standard deviation ``sigma``.

    This is the classic variation operator presented in `Mouret 2015
    <https://arxiv.org/pdf/1504.04909.pdf>`_.

    Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        x0 (array-like): Center of the Gaussian distribution from which to
            sample solutions when the archive is empty. Must be 1-dimensional.
        sigma (float or array-like): Standard deviation of the Gaussian
            distribution when the archive is not empty. Note we assume
            the Gaussian is diagonal, so if this argument is an array, it
            must be 1D.
        sigma0 (float or array-like): Standard deviation of the Gaussian
            distribution when the archive is empty. If this argument is
            None, then it defaults to sigma. Note we assume the Gaussian is
            diagonal, so if this argument is an array, it must be 1D.
        bounds (None or array-like): Bounds of the solution space. Solutions are
            clipped to these bounds. Pass None to indicate there are no bounds.
            Alternatively, pass an array-like to specify the bounds for each
            dim. Each element in this array-like can be None to indicate no
            bound, or a tuple of ``(lower_bound, upper_bound)``, where
            ``lower_bound`` or ``upper_bound`` may be None to indicate no bound.
        batch_size (int): Number of solutions to return in :meth:`ask`.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
    Raises:
        ValueError: There is an error in the bounds configuration.
    """

    def __init__(self,
                 archive,
                 x0,
                 sigma,
                 sigma0=None,
                 bounds=None,
                 batch_size=64,
                 seed=None):
        self._rng = np.random.default_rng(seed)
        self._batch_size = batch_size

        self._x0 = np.array(x0, dtype=archive.dtype)
        if self._x0.ndim != 1:
            raise ValueError(
                f"x0 has shape {self._x0.shape}, should be 1-dimensional.")

        self._sigma = archive.dtype(sigma) if isinstance(
            sigma,
            (float, np.floating)) else np.array(sigma, dtype=archive.dtype)
        self._sigma0 = self._sigma if sigma0 is None else (
            archive.dtype(sigma0) if isinstance(sigma0, (
                float, np.floating)) else np.array(sigma0, dtype=archive.dtype))

        EmitterBase.__init__(
            self,
            archive,
            len(self._x0),
            bounds,
        )

    @property
    def x0(self):
        """numpy.ndarray: Center of the Gaussian distribution from which to
        sample solutions when the archive is empty."""
        return self._x0

    @property
    def sigma(self):
        """float or numpy.ndarray: Standard deviation of the (diagonal) Gaussian
        distribution when the archive is not empty."""
        return self._sigma

    @property
    def sigma0(self):
        """float or numpy.ndarray: Standard deviation of the (diagonal) Gaussian
        distribution when the archive is empty."""
        return self._sigma0

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    def ask(self):
        """Creates solutions by adding Gaussian noise to elites in the archive.

        If the archive is empty, solutions are drawn from a (diagonal) Gaussian
        distribution centered at ``self.x0`` with standard deviation
        ``self.sigma0``. Otherwise, each solution is drawn from a distribution
        centered at a randomly chosen elite with standard deviation
        ``self.sigma``.

        Returns:
            ``(batch_size, solution_dim)`` array -- contains ``batch_size`` new
            solutions to evaluate.
        """
        if self.archive.empty:
            sigma = self._sigma0
            parents = np.expand_dims(self._x0, axis=0)
        else:
            sigma = self._sigma
            parents = self.archive.sample_elites(
                self._batch_size).solution_batch

        noise = self._rng.normal(
            scale=sigma,
            size=(self._batch_size, self.solution_dim),
        ).astype(self.archive.dtype)

        return np.clip(parents + noise, self.lower_bounds, self.upper_bounds)

    def tell(self,
             solution_batch,
             objective_batch,
             measures_batch,
             status_batch,
             value_batch,
             metadata_batch=None):
        """Inserts entries into the archive.

        Args:
            solution_batch (numpy.ndarray): Array of solutions generated by this
                emitter's :meth:`ask()` method.
            objective_batch (numpy.ndarray): 1D array containing the objective
                function value of each solution.
            behavior_batch (numpy.ndarray): ``(n, <behavior space dimension>)``
                array with the behavior space coordinates of each solution.
            status_batch (numpy.ndarray): An array of integer statuses
                returned by a series of calls to archive's :meth:`add_single()`
                method or by a single call to archive's :meth:`add()`.
            value_batch  (numpy.ndarray): 1D array of floats returned by a
                series of calls to archive's :meth:`add_single()` method or by a
                single call to archive's :meth:`add()`. For what these floats
                represent, refer to :meth:`ribs.archives.add()`.
            metadata_batch (numpy.ndarray): 1D object array containing a
                metadata object for each solution.
        """
