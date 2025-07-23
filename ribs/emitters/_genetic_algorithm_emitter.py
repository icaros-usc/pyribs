"""Provides the GeneticAlgorithmEmitter."""

import numpy as np

from ribs._utils import check_batch_shape, check_shape
from ribs.emitters._emitter_base import EmitterBase
from ribs.emitters.operators import _get_op


class GeneticAlgorithmEmitter(EmitterBase):
    """Creates solutions with a genetic algorithm.

    If the archive is empty and ``initial_solutions`` is set, a call to :meth:`ask` will
    return ``initial_solutions``. If ``initial_solutions`` is not set, we pass ``x0``
    through the operator.

    Args:
        archive (ribs.archives.ArchiveBase): Archive of solutions, e.g.,
            :class:`ribs.archives.GridArchive`.
        operator (str): Internal operator for mutating solutions. See
            :mod:`ribs.emitters.operators` for the list of allowed names.
        operator_kwargs (dict): Additional arguments to pass to the operator. See
            :mod:`ribs.emitters.operators` for the arguments allowed by each operator.
        x0 (numpy.ndarray): Initial solution.
        initial_solutions (array-like): An (n, solution_dim) array of solutions to be
            used when the archive is empty, in lieu of ``x0``.
        bounds (None or array-like): Bounds of the solution space. Solutions are clipped
            to these bounds. Pass None to indicate there are no bounds. Alternatively,
            pass an array-like to specify the bounds for each dim. Each element in this
            array-like can be None to indicate no bound, or a tuple of ``(lower_bound,
            upper_bound)``, where ``lower_bound`` or ``upper_bound`` may be None to
            indicate no bound.
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
        operator,
        operator_kwargs=None,
        x0=None,
        initial_solutions=None,
        bounds=None,
        batch_size=64,
        seed=None,
    ):
        EmitterBase.__init__(
            self,
            archive,
            solution_dim=archive.solution_dim,
            bounds=bounds,
        )

        self._batch_size = batch_size
        self._x0 = x0
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

        operator_class = _get_op(operator)
        self._operator = operator_class(
            **(operator_kwargs if operator_kwargs is not None else {}),
            seed=seed,
        )

    @property
    def x0(self):
        """numpy.ndarray: Initial Solution (if ``initial_solutions`` is not
        set)."""
        return self._x0

    @property
    def initial_solutions(self):
        """numpy.ndarray: The initial solutions which are returned when the
        archive is empty (if x0 is not set)."""
        return self._initial_solutions

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    def _clip(self, solutions):
        """Clips solutions to the bounds of the solution space."""
        return np.clip(solutions, self.lower_bounds, self.upper_bounds)

    def ask(self):
        """Creates solutions with the provided operator.

        If the archive is empty and ``initial_solutions`` is set, a call to :meth:`ask`
        will return ``initial_solutions``. If ``initial_solutions`` is not set, we pass
        ``x0`` through the operator. Otherwise, we sample parents from the archive to be
        passed to the operator.

        Returns:
            numpy.ndarray: If the archive is not empty, ``(batch_size, solution_dim)``
            array -- contains ``batch_size`` new solutions to evaluate. If the archive
            is empty, we return ``initial_solutions``, which might not have
            ``batch_size`` solutions.
        Raises:
            ValueError: The ``parent_type`` of the operator is unknown.
        """
        if self.archive.empty and self.initial_solutions is not None:
            return self._clip(self.initial_solutions)

        if self._operator.parent_type == 1:
            if self.archive.empty:
                parents = np.repeat(self.x0[None], repeats=self.batch_size, axis=0)
            else:
                parents = self.archive.sample_elites(self.batch_size)["solution"]
            return self._clip(self._operator.ask(parents))

        elif self._operator.parent_type == 2:
            if self.archive.empty:
                parents = np.repeat(self.x0[None], repeats=2 * self.batch_size, axis=0)
            else:
                parents = self.archive.sample_elites(2 * self.batch_size)["solution"]
            return self._clip(
                self._operator.ask(parents.reshape(2, self.batch_size, -1))
            )

        else:
            raise ValueError(
                f"Unknown operator `parent_type` {self._operator.parent_type}"
            )
