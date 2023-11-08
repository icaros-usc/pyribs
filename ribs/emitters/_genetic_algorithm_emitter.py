"""Provides the GaussianEmitter."""
import numpy as np

from ribs._utils import check_1d_shape, check_batch_shape
from ribs.emitters._emitter_base import EmitterBase
from ribs.emitters.operators import _get_op


class GeneticAlgorithmEmitter(EmitterBase):
    """Emits solutions by using operator provided

    If the archive is empty and ``self._initial_solutions`` is set, a call to
    :meth:`ask` will return ``self._initial_solutions``. If
    ``self._initial_solutions`` is not set, we operate on self.x0.


    Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        x0 (np.ndarray): Initial solution.
        operator (external class): Operator Class from pymoo or pygad
        os (string): External Library identifier
        initial_solutions (array-like): An (n, solution_dim) array of solutions
            to be used when the archive is empty. If this argument is None, then
            solutions will be sampled from a Gaussian distribution centered at
            ``x0`` with standard deviation ``sigma``.
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
        ValueError: There is an error in x0 or initial_solutions.
        ValueError: There is an error in the bounds configuration.
    """

    def __init__(self,
                 archive,
                 *,
                 operator=None,
                 x0=None,
                 initial_solutions=None,
                 bounds=None,
                 batch_size=64,
                 os=None,
                 seed=None,
                 iso_sigma=0.01,
                 line_sigma=0.2,
                 sigma=0.1):
        self._batch_size = batch_size
        self._os = os
        self._x0 = x0
        self._initial_solutions = None
        self._seed = seed
        self._sigma = sigma
        self._iso_sigma = iso_sigma
        self._line_sigma = line_sigma

        if x0 is None and initial_solutions is None:
            raise ValueError("Either x0 or initial_solutions must be provided.")
        if x0 is not None and initial_solutions is not None:
            raise ValueError(
                "x0 and initial_solutions cannot both be provided.")

        if x0 is not None:
            self._x0 = np.array(x0, dtype=archive.dtype)
            check_1d_shape(self._x0, "x0", archive.solution_dim,
                           "archive.solution_dim")
        elif initial_solutions is not None:
            self._initial_solutions = np.asarray(initial_solutions,
                                                 dtype=archive.dtype)
            check_batch_shape(self._initial_solutions, "initial_solutions",
                              archive.solution_dim, "archive.solution_dim")

        EmitterBase.__init__(
            self,
            archive,
            solution_dim=archive.solution_dim,
            bounds=bounds,
        )
        self._operator = _get_op(os)(operator=operator,
                                     lower_bounds=self._lower_bounds,
                                     upper_bounds=self._upper_bounds,
                                     seed=self._seed,
                                     sigma=self._sigma,
                                     iso_sigma=self._iso_sigma,
                                     line_sigma=self._line_sigma)

    @property
    def x0(self):
        """numpy.ndarray: Initial Solution (if initial_solutions is not
        set)."""
        return self._x0

    @property
    def iso_sigma(self):
        """float: Scale factor for the isotropic distribution used to
        generate solutions when the archive is not empty."""
        return self._iso_sigma

    @property
    def line_sigma(self):
        """float: Scale factor for the line distribution used when generating
        solutions."""
        return self._line_sigma

    @property
    def sigma(self):
        """float or numpy.ndarray: Standard deviation of the (diagonal) Gaussian
        distribution when the archive is not empty."""
        return self._sigma

    @property
    def initial_solutions(self):
        """numpy.ndarray: The initial solutions which are returned when the
        archive is empty (if x0 is not set)."""
        return self._initial_solutions

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    def ask(self):
        """Creates solutions with operator provided.


        If the archive is empty and ``self._initial_solutions`` is set, we
        return ``self._initial_solutions``. If ``self._initial_solutions`` is
        not set and the archive is still empty, we operate on the initial
        solution (x0) provided. Otherwise, we sample parents from the archive
        to be used as input to our operator

        Returns:
            If the archive is not empty, ``(batch_size, solution_dim)`` array
            -- contains ``batch_size`` new solutions to evaluate. If the
            archive is empty, we return ``self._initial_solutions``, which
            might not have ``batch_size`` solutions.
        """
        if self.archive.empty:
            if self._initial_solutions is not None:
                return np.clip(self._initial_solutions, self.lower_bounds,
                               self.upper_bounds)
            parents = np.expand_dims(self._x0, axis=0)

            directions = np.full((parents.shape[0], parents.shape[1]), 0)
        else:
            parents = self.archive.sample_elites(
                self._batch_size).solution_batch
            directions = (
                self.archive.sample_elites(self._batch_size).solution_batch -
                parents)

        solution = self._operator.operate(parents=parents,
                                          bounds=(self._lower_bounds[0],
                                                  self._upper_bounds[0]),
                                          n_var=len(parents[0]),
                                          directions=directions)
        return solution
