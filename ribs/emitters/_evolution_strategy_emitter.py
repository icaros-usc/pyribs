"""Provides the EvolutionStrategyEmitter."""
import itertools

import numpy as np

from ribs.emitters._emitter_base import EmitterBase


class EvolutionStrategyEmitter(EmitterBase):
    """Adapts a evolution strategy towards the objective.

    This emitter originates in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_. Initially, it starts at ``x0`` and
    uses CMA-ES to optimize for objective values. After CMA-ES converges, the
    emitter restarts the optimizer. It picks a random elite in the archive and
    begins optimizing from there.

    Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        x0 (np.ndarray): Initial solution.
        sigma0 (float): Initial step size.
        ranker (RankerBase):
        selector (Selector): Method for selecting solutions in
            CMA-ES. With "mu" selection, the first half of the solutions will be
            selected, while in "filter", any solutions that were added to the
            archive will be selected.
        evolution_strategy (EvolutionStrategy): The evolution strategy to use
            :class:`ribs.emitter.opt.CMAEvolutionStrategy`
        restart_rule ("no_improvement" or "basic"): Method to use when checking
            for restart. With "basic", only the default CMA-ES convergence rules
            will be used, while with "no_improvement", the emitter will restart
            when none of the proposed solutions were added to the archive.
        bounds (None or array-like): Bounds of the solution space. Solutions are
            clipped to these bounds. Pass None to indicate there are no bounds.
            Alternatively, pass an array-like to specify the bounds for each
            dim. Each element in this array-like can be None to indicate no
            bound, or a tuple of ``(lower_bound, upper_bound)``, where
            ``lower_bound`` or ``upper_bound`` may be None to indicate no bound.
        batch_size (int): Number of solutions to return in :meth:`ask`. If not
            passed in, a batch size will automatically be calculated.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
    Raises:
        ValueError: If ``restart_rule`` is invalid.
    """

    def __init__(self,
                 archive,
                 x0,
                 sigma0,
                 ranker,
                 selector,
                 evolution_strategy,
                 restart_rule="no_improvement",
                 bounds=None,
                 batch_size=None,
                 seed=None):
        self._rng = np.random.default_rng(seed)
        self._x0 = np.array(x0, dtype=archive.dtype)
        self._sigma0 = sigma0
        EmitterBase.__init__(
            self,
            archive,
            len(self._x0),
            bounds,
        )

        self.selector = selector

        if restart_rule not in ["basic", "no_improvement"]:
            raise ValueError(f"Invalid restart_rule {restart_rule}")
        self._restart_rule = restart_rule

        self.opt = evolution_strategy
        self.opt.reset(self._x0)

        self._ranker = ranker
        self._ranker.reset(archive, self)

        self._selector = selector
        # self._num_parents = (self.batch_size //
        #                      2 if selection_rule == "mu" else None)
        self._batch_size = batch_size
        self._restarts = 0  # Currently not exposed publicly.

    @property
    def x0(self):
        """numpy.ndarray: Initial solution for the optimizer."""
        return self._x0

    @property
    def sigma0(self):
        """float: Initial step size for the CMA-ES optimizer."""
        return self._sigma0

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    def ask(self):
        """Samples new solutions from a multivariate Gaussian.

        The multivariate Gaussian is parameterized by the evolution strategy
        optimizer ``self.opt``.

        Returns:
            ``(batch_size, solution_dim)`` array -- contains ``batch_size`` new
            solutions to evaluate.
        """
        return self.opt.ask(self.lower_bounds, self.upper_bounds)

    def _generate_random_direction(self):
        """Generates a new random direction in the behavior space.

        The direction is sampled from a standard Gaussian -- since the standard
        Gaussian is isotropic, there is equal probability for any direction. The
        direction is then scaled to the behavior space bounds.
        """
        ranges = self.archive.upper_bounds - self.archive.lower_bounds
        behavior_dim = len(ranges)
        unscaled_dir = self._rng.standard_normal(behavior_dim)
        return unscaled_dir * ranges

    def _check_restart(self, num_parents):
        """Emitter-side checks for restarting the optimizer.

        The optimizer also has its own checks.
        """
        if self._restart_rule == "no_improvement":
            return num_parents == 0
        return False

    def tell(self, solutions, objective_values, behavior_values, metadata=None):
        """Gives the emitter results from evaluating solutions.

        As we insert solutions into the archive, we record the solutions'
        projection onto the random direction in behavior space, as well as
        whether the solution was added to the archive. When using "filter"
        selection, we rank the solutions first by whether they were added, and
        second by the projection, and when using "mu" selection, we rank solely
        by projection. We then pass the ranked solutions to the underlying
        CMA-ES optimizer to update the search parameters.

        Args:
            solutions (numpy.ndarray): Array of solutions generated by this
                emitter's :meth:`ask()` method.
            objective_values (numpy.ndarray): 1D array containing the objective
                function value of each solution.
            behavior_values (numpy.ndarray): ``(n, <behavior space dimension>)``
                array with the behavior space coordinates of each solution.
            metadata (numpy.ndarray): 1D object array containing a metadata
                object for each solution.
        """
        # Tuples of (solution was added, projection onto random direction,
        # index).
        ranking_data = []
        new_sols = 0

        metadata = itertools.repeat(None) if metadata is None else metadata

        # Tupe of (add status, add value)
        add_data = []

        # Add solutions to the archive.
        for i, (sol, obj, beh, meta) in enumerate(
                zip(solutions, objective_values, behavior_values, metadata)):
            add_data.append(self.archive.add(sol, obj, beh, meta))

        indices = self._ranker.rank(self, self._archive, solutions,
                                    objective_values, behavior_values, metadata,
                                    add_data[0], add_data[1])

        num_parents = self._selector.select(self, self._archive, solutions,
                                            objective_values, behavior_values,
                                            metadata, add_data[0], add_data[1])
        # (new_sols if self._selection_rule == "filter" else self._num_parents)

        self.opt.tell(solutions[indices], num_parents)

        # Check for reset.
        if (self.opt.check_stop(
            [projection for status, projection, i in ranking_data]) or
                self._check_restart(new_sols)):
            new_x0 = self.archive.sample_elites(1).solution_batch[0]
            self.opt.reset(new_x0)
            self._ranker.reset(self.archive, self)
            self._restarts += 1
