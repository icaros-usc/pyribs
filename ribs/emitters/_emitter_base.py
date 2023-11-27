"""Provides EmitterBase."""
from abc import ABC

import numpy as np


class EmitterBase(ABC):
    """Base class for emitters.

    Every emitter has an :meth:`ask` method that generates a batch of solutions,
    and a :meth:`tell` method that inserts solutions into the emitter's archive.
    Child classes are only required to override :meth:`ask`.

    DQD emitters should override :meth:`ask_dqd` and :meth:`tell_dqd` methods.

    Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        solution_dim (int): The dimension of solutions produced by this emitter.
        bounds (None or array-like): Bounds of the solution space. Pass None to
            indicate there are no bounds. Alternatively, pass an array-like to
            specify the bounds for each dim. Each element in this array-like can
            be None to indicate no bound, or a tuple of
            ``(lower_bound, upper_bound)``, where ``lower_bound`` or
            ``upper_bound`` may be None to indicate no bound.

            Unbounded upper bounds are set to +inf, and unbounded lower bounds
            are set to -inf.
    """

    def __init__(self, archive, *, solution_dim, bounds):
        self._archive = archive
        self._solution_dim = solution_dim
        (self._lower_bounds,
         self._upper_bounds) = self._process_bounds(bounds, self._solution_dim,
                                                    archive.dtype)

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
    def archive(self):
        """ribs.archives.ArchiveBase: The archive which stores solutions
        generated by this emitter."""
        return self._archive

    @property
    def solution_dim(self):
        """int: The dimension of solutions produced by this emitter."""
        return self._solution_dim

    @property
    def lower_bounds(self):
        """numpy.ndarray: ``(solution_dim,)`` array with lower bounds of
        solution space.

        For instance, ``[-1, -1, -1]`` indicates that every dimension of the
        solution space has a lower bound of -1.
        """
        return self._lower_bounds

    @property
    def upper_bounds(self):
        """numpy.ndarray: ``(solution_dim,)`` array with upper bounds of
        solution space.

        For instance, ``[1, 1, 1]`` indicates that every dimension of the
        solution space has an upper bound of 1.
        """
        return self._upper_bounds

    def ask(self):
        """Generates a ``(batch_size, solution_dim)`` array of solutions.

        Returns an empty array by default.
        """
        return np.empty((0, self.solution_dim), dtype=self.archive.dtype)

    def tell(self, solution, objective, measures, status_batch, value_batch,
             **fields):
        """Gives the emitter results from evaluating solutions.

        This base class implementation (in :class:`~ribs.emitters.EmitterBase`)
        needs to be overriden.

        Args:
            solution (numpy.ndarray): Array of solutions generated by this
                emitter's :meth:`ask()` method.
            objective (numpy.ndarray): 1D array containing the objective
                function value of each solution.
            measures (numpy.ndarray): ``(n, <measure space dimension>)`` array
                with the measure space coordinates of each solution.
            status_batch (numpy.ndarray): An array of integer statuses
                returned by a series of calls to archive's :meth:`add_single()`
                method or by a single call to archive's :meth:`add()`.
            value_batch (numpy.ndarray): 1D array of floats returned by a
                series of calls to archive's :meth:`add_single()` method or by a
                single call to archive's :meth:`add()`. For what these floats
                represent, refer to :meth:`ribs.archives.add()`.
            fields (keyword arguments): Additional data for each solution. Each
                argument should be an array with batch_size as the first
                dimension.
        """

    def ask_dqd(self):
        """Generates a ``(batch_size, solution_dim)`` array of solutions for
        which gradient information must be computed.

        This method only needs to be implemented by emitters used in DQD. The
        method returns an empty array by default.
        """
        return np.empty((0, self.solution_dim), dtype=self.archive.dtype)

    def tell_dqd(self, solution, objective, measures, jacobian, status_batch,
                 value_batch, **fields):
        """Gives the emitter results from evaluating the gradient of the
        solutions, only used for DQD emitters.

        Args:
            solution (numpy.ndarray): ``(batch_size, :attr:`solution_dim`)``
                array of solutions generated by this emitter's :meth:`ask()`
                method.
            objective (numpy.ndarray): 1-dimensional array containing the
                objective function value of each solution.
            measures (numpy.ndarray): ``(batch_size, measure space dimension)``
                array with the measure space coordinates of each solution.
            jacobian (numpy.ndarray): ``(batch_size, 1 + measure_dim,
                solution_dim)`` array consisting of Jacobian matrices of the
                solutions obtained from :meth:`ask_dqd`. Each matrix should
                consist of the objective gradient of the solution followed by
                the measure gradients.
            status_batch (numpy.ndarray): 1d array of
                :class:`ribs.archive.addstatus` returned by a series of calls
                to archive's :meth:`add()` method.
            value_batch (numpy.ndarray): 1d array of floats returned by a series
                of calls to archive's :meth:`add()` method. for what these
                floats represent, refer to :meth:`ribs.archives.add()`.
            fields (keyword arguments): Additional data for each solution. Each
                argument should be an array with batch_size as the first
                dimension.
        """
