"""Provides the GaussianEmitter."""

import numpy as np

from ribs._utils import check_batch_shape, check_shape, deprecate_bounds
from ribs.emitters._emitter_base import EmitterBase


class GaussianEmitter(EmitterBase):
    """Emits solutions by adding Gaussian noise to existing elites.

    If the archive is empty and ``initial_solutions`` is set, a call to :meth:`ask` will
    return ``initial_solutions``. If ``initial_solutions`` is not set, we draw solutions
    from a Gaussian distribution centered at ``x0`` with standard deviation ``sigma``.
    Otherwise, each solution is drawn from a distribution centered at a randomly chosen
    elite with standard deviation ``sigma``.

    Args:
        archive (ribs.archives.ArchiveBase): Archive of solutions, e.g.,
            :class:`ribs.archives.GridArchive`.
        sigma (float or array-like): Standard deviation of the Gaussian distribution.
            Note we assume the Gaussian is diagonal, so if this argument is an array, it
            must be 1D.
        x0 (array-like): Center of the Gaussian distribution from which to sample
            solutions when the archive is empty. Must be 1-dimensional. This argument is
            ignored if ``initial_solutions`` is set.
        initial_solutions (array-like): An (n, solution_dim) array of solutions to be
            used when the archive is empty. If this argument is None, then solutions
            will be sampled from a Gaussian distribution centered at ``x0`` with
            standard deviation ``sigma``.
        lower_bounds (None or array-like): Lower bounds of the solution space. Pass None
            to indicate there are no bounds (i.e., bounds are set to -inf).
        upper_bounds (None or array-like): Upper bounds of the solution space. Pass None
            to indicate there are no bounds (i.e., bounds are set to inf).
        bounds: DEPRECATED.
        batch_size (int): Number of solutions to return in :meth:`ask`.
        seed (int): Value to seed the random number generator. Set to None to avoid a
            fixed seed.

    Raises:
        ValueError: There is an error in x0 or initial_solutions.
        ValueError: There is an error in the bounds configuration.
    """

    def __init__(
        self,
        archive,
        *,
        sigma,
        x0=None,
        initial_solutions=None,
        lower_bounds=None,
        upper_bounds=None,
        bounds=None,
        batch_size=64,
        seed=None,
    ):
        deprecate_bounds(bounds)

        EmitterBase.__init__(
            self,
            archive,
            solution_dim=archive.solution_dim,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        self._rng = np.random.default_rng(seed)
        self._batch_size = batch_size
        self._sigma = np.asarray(sigma, dtype=archive.dtypes["solution"])
        self._x0 = None
        self._initial_solutions = None

        if x0 is None and initial_solutions is None:
            raise ValueError("Either x0 or initial_solutions must be provided.")
        if x0 is not None and initial_solutions is not None:
            raise ValueError("x0 and initial_solutions cannot both be provided.")

        if x0 is not None:
            self._x0 = np.array(x0, dtype=archive.dtypes["solution"])
            check_shape(self._x0, "x0", archive.solution_dim, "archive.solution_dim")
        elif initial_solutions is not None:
            self._initial_solutions = np.asarray(
                initial_solutions, dtype=archive.dtypes["solution"]
            )
            check_batch_shape(
                self._initial_solutions,
                "initial_solutions",
                archive.solution_dim,
                "archive.solution_dim",
            )

    @property
    def x0(self):
        """numpy.ndarray: Initial Gaussian distribution center.

        Solutions are sampled from this distribution when the archive is empty (if
        :attr:`initial_solutions` is not set).
        """
        return self._x0

    @property
    def initial_solutions(self):
        """numpy.ndarray: Returned when the archive is empty (if :attr:`x0` is not set)."""
        return self._initial_solutions

    @property
    def sigma(self):
        """float or numpy.ndarray: Standard deviation of the (diagonal) Gaussian distribution."""  # noqa: D403
        return self._sigma

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    def _clip(self, solutions):
        """Clips solutions to the bounds of the solution space."""
        return np.clip(solutions, self.lower_bounds, self.upper_bounds)

    def ask(self):
        """Creates solutions by adding Gaussian noise to elites in the archive.

        If the archive is empty and ``initial_solutions`` is set, a call to :meth:`ask`
        will return ``initial_solutions``. If ``initial_solutions`` is not set, we draw
        solutions from a Gaussian distribution centered at ``x0`` with standard
        deviation ``sigma``. Otherwise, each solution is drawn from a distribution
        centered at a randomly chosen elite with standard deviation ``sigma``.

        Returns:
            If the archive is not empty, ``(batch_size, solution_dim)`` array --
            contains ``batch_size`` new solutions to evaluate. If the archive is empty,
            we return ``initial_solutions``, which might not have ``batch_size``
            solutions.
        """
        if self.archive.empty and self.initial_solutions is not None:
            return self._clip(self.initial_solutions)

        if self.archive.empty:
            parents = np.repeat(self.x0[None], repeats=self.batch_size, axis=0)
        else:
            parents = self.archive.sample_elites(self.batch_size)["solution"]

        noise = self._rng.normal(
            scale=self.sigma,
            size=(self.batch_size, self.solution_dim),
        ).astype(self.archive.dtypes["solution"])

        return self._clip(parents + noise)
