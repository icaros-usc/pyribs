"""Provides EmitterBase."""
from abc import ABC, abstractmethod

import numpy as np


class EmitterBase(ABC):
    """Base class for emitters.

    Every emitter has an :meth:`ask` method that generates a batch of solutions,
    and a :meth:`tell` method that inserts solutions into the emitter's archive.
    Users are only required to override :meth:`ask`.

    .. note:: Members beginning with an underscore are only intended to be
        accessed by child classes.

    Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        solution_dim (int): The dimension of solutions produced by this emitter.
        bounds (None or array-like): Bounds of the solution space. Each emitter
            decides how to handle these bounds (if at all). Unbounded upper
            bounds are set to +inf, and unbounded lower bounds are set to -inf.

            Pass None to indicate there are no bounds.

            Pass an array-like to specify the bounds for each dim. Each element
            in this array-like can be None to indicate no bound, or a tuple of
            ``(lower_bound, upper_bound)``, where ``lower_bound`` or
            ``upper_bound`` may be None to indicate no bound.
        batch_size (int): Number of solutions to generate on each call to
            :meth:`ask`.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
    Attributes:
        _rng (numpy.random.Generator): Random number generator.
        _archive (ribs.archives.ArchiveBase): See ``archive`` arg.
    """

    def __init__(self, archive, solution_dim, bounds, batch_size, seed=None):
        self._rng = np.random.default_rng(seed)
        self._archive = archive
        self._solution_dim = solution_dim
        (self._lower_bounds,
         self._upper_bounds) = self._process_bounds(bounds, self._solution_dim,
                                                    archive.dtype)
        self._batch_size = batch_size

    @staticmethod
    def _process_bounds(bounds, solution_dim, dtype):
        """Processes the input bounds.

        Returns:
            tuple: Two arrays containing all the lower bounds and all the upper
                bounds.
        Raises:
            ValueError: There is an error in the bounds configuration.
        """
        lower_bounds = np.full(solution_dim, -np.inf, dtype=dtype)
        upper_bounds = np.full(solution_dim, np.inf, dtype=dtype)

        if bounds is None:
            return lower_bounds, upper_bounds

        # Handle array-like bounds.
        if len(bounds) != solution_dim:
            raise ValueError("If it is an array-like, bounds must have the "
                             "same length as x0")
        for idx, bnd in enumerate(bounds):
            if bnd is None:
                continue  # Bounds already default to -inf and inf.
            if len(bnd) != 2:
                raise ValueError("All entries of bounds must be length 2")
            lower_bounds[idx] = -np.inf if bnd[0] is None else bnd[0]
            upper_bounds[idx] = np.inf if bnd[1] is None else bnd[1]
        return lower_bounds, upper_bounds

    @property
    def solution_dim(self):
        """int: The dimension of solutions produced by this emitter."""
        return self._solution_dim

    @property
    def lower_bounds(self):
        """numpy.ndarray: Lower bound of each dim of the solution space."""
        return self._lower_bounds

    @property
    def upper_bounds(self):
        """numpy.ndarray: Upper bound of each dim of the solution space."""
        return self._upper_bounds

    @property
    def batch_size(self):
        """int: Number of solutions to generate on each call to :meth:`ask`."""
        return self._batch_size

    @abstractmethod
    def ask(self):
        """Generates ``self.batch_size`` solutions."""

    def tell(self, solutions, objective_values, behavior_values):
        """Gives the emitter results from evaluating several solutions.

        These solutions are then inserted into the archive.

        Args:
            solutions (numpy.ndarray): Array of solutions generated by this
                emitter's :meth:`ask()` method.
            objective_values (numpy.ndarray): 1D array containing the objective
                function value of each solution.
            behavior_values (numpy.ndarray): ``(n, <behavior space dimension>)``
                array with the behavior space coordinates of each solution.
        """
        for sol, obj, beh in zip(solutions, objective_values, behavior_values):
            self._archive.add(sol, obj, beh)
