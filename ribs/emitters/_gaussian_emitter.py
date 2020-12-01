"""Provides the GaussianEmitter."""
import numpy as np
from numba import jit

from ribs.emitters._emitter_base import EmitterBase


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
        bounds (None or tuple or array-like): Bounds of the solution space.
            Solutions are clipped to these bounds. Passing None indicates there
            are no bounds (i.e. bounds are set to +/-inf). A two-element tuple
            of ``(lower_bound, upper_bound)`` specifies bounds that apply to all
            dims. ``lower_bound`` or ``upper_bound`` may be None to indicate
            there is no bound.

            Finally, you can pass an array-like (but not a tuple) of elements
            that specify the bounds for each dim. Each element in this iterable
            can be None to indicate no bound, or a tuple of ``(lower_bound,
            upper_bound)`` as described above.
        batch_size (int): Number of solutions to send back in the ask() method.
        seed (float or int): Value to seed the random number generator. Set to
            None to avoid seeding.
    Raises:
        ValueError: There is an error in the bounds configuration.
    """

    def __init__(self,
                 x0,
                 sigma0,
                 archive,
                 bounds=None,
                 batch_size=64,
                 seed=None):
        self._x0 = np.array(x0)
        self._sigma0 = sigma0 if isinstance(sigma0, float) else np.array(sigma0)
        (self._lower_bounds,
         self._upper_bounds) = self._process_bounds(bounds, len(self._x0))
        EmitterBase.__init__(self, len(self._x0), batch_size, archive, seed)

    @staticmethod
    def _process_bounds(bounds, solution_dim):
        """Processes the input bounds.

        Returns:
            tuple: Either two integers for the lower and upper bounds, or two
                arrays containing all the lower bounds and all the upper bounds.
        Raises:
            ValueError: There is an error in the bounds configuration.
        """
        if bounds is None:
            return -np.inf, np.inf
        if isinstance(bounds, tuple):
            if len(bounds) != 2:
                raise ValueError("If it is a tuple, bounds must be length 2")
            return (-np.inf if bounds[0] is None else bounds[0],
                    np.inf if bounds[1] is None else bounds[1])

        if len(bounds) != solution_dim:
            raise ValueError("If it is an array-like, bounds must have the "
                             "same length as x0")
        lower_bounds = np.full(solution_dim, -np.inf)
        upper_bounds = np.full(solution_dim, np.inf)
        for idx, bnd in enumerate(bounds):
            if bnd is None:
                continue  # Bounds already default to -inf and inf.
            if len(bnd) != 2:
                raise ValueError("All entries of bounds must be length 2")
            lower_bounds[idx] = -np.inf if bnd[0] is None else bnd[0]
            upper_bounds[idx] = np.inf if bnd[1] is None else bnd[1]
        return lower_bounds, upper_bounds

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

    @property
    def lower_bounds(self):
        """float or np.ndarray: Lower bounds of the solution space."""
        return self._lower_bounds

    @property
    def upper_bounds(self):
        """float or np.ndarray: Upper bounds of the solution space."""
        return self._upper_bounds

    @staticmethod
    @jit(nopython=True)
    def _ask_expand_dims_helper(x0):
        return np.expand_dims(x0, axis=0)

    @staticmethod
    @jit(nopython=True)
    def _ask_clip_helper(parents, noise, lower_bounds, upper_bounds):
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
        if self._archive.is_empty():
            parents = self._ask_expand_dims_helper(x0=self._x0)
        else:
            parents = [
                self._archive.get_random_elite()[0]
                for _ in range(self.batch_size)
            ]

        noise = self._rng.normal(scale=self._sigma0,
                                 size=(self.batch_size, self.solution_dim))
        return self._ask_clip_helper(np.array(parents), noise, self._lower_bounds,
                                     self._upper_bounds)
