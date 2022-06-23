"""Provides the EvolutionStrategyEmitter."""
import itertools

import numpy as np

from ribs.emitters._emitter_base import EmitterBase
from ribs.emitters.opt import CMAEvolutionStrategy
from ribs.emitters.rankers import RankerBase, get_ranker


class EvolutionStrategyEmitter(EmitterBase):
    """Adapts a evolution strategy optimizer towards the objective.

    This emitter originates in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_. Initially, it starts at ``x0`` and
    uses some evolution strategy (i.e. CMA-ES) to optimize for objective values.
    After the evolution strategy converges, the emitter restarts the optimizer.

    Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        x0 (np.ndarray): Initial solution.
        ranker (ribs.emitters.rankers.RankerBase or str): The Ranker object
            defines how the generated solutions are ranked and what to do on
            restart. If passing in the full or abbreviated name of the ranker,
            the corresponding ranker will be created in the constructor.
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
                 selection_rule="filter",
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

        if selection_rule not in ["mu", "filter"]:
            raise ValueError(f"Invalid selection_rule {selection_rule}")
        self._selection_rule = selection_rule

        if restart_rule not in ["basic", "no_improvement"]:
            raise ValueError(f"Invalid restart_rule {restart_rule}")
        self._restart_rule = restart_rule

        opt_seed = None if seed is None else self._rng.integers(10_000)
        self.opt = CMAEvolutionStrategy(sigma0, batch_size, self._solution_dim,
                                        "truncation", opt_seed,
                                        self.archive.dtype)
        self.opt.reset(self._x0)

        self._ranker = ranker if isinstance(ranker,
                                            RankerBase) else get_ranker(ranker)
        self._ranker.reset(self, archive, self._rng)

        self._batch_size = self.opt.batch_size
        self._restarts = 0  # Currently not exposed publicly.

    @property
    def x0(self):
        """numpy.ndarray: Initial solution for the optimizer."""
        return self._x0

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
        impact on the fitness of the archive. For example, if the added
        solution makes an improvement on an existing elite, then we
        will record ``(AddStatus.IMPROVED_EXISTING, imporvement_value)``

        The solutions are ranked based on the `rank()` function defined by
        `self._ranker`.

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
        add_statues = []
        add_values = []

        metadata = itertools.repeat(None) if metadata is None else metadata

        # Add solutions to the archive.
        new_sols = 0
        for (sol, obj, beh, meta) in zip(solutions, objective_values,
                                         behavior_values, metadata):
            status, value = self.archive.add(sol, obj, beh, meta)
            add_statues.append(status)
            add_values.append(value)
            if bool(status):
                new_sols += 1

        # Sort the solutions using ranker
        indices = self._ranker.rank(self, self.archive, self._rng, solutions,
                                    objective_values, behavior_values, metadata,
                                    add_statues, add_values)

        # Select the number of parents
        num_parents = (new_sols if self._selection_rule == "filter" else
                       self.batch_size // 2)

        # Update Evolution Strategy
        self.opt.tell(solutions[indices], num_parents)

        # Check for reset.
        if (self.opt.check_stop(list(objective_values)) or
                self._check_restart(new_sols)):
            new_x0 = self.archive.sample_elites(1).solution_batch[0]
            self.opt.reset(new_x0)
            self._ranker.reset(self, self.archive, self._rng)
            self._restarts += 1
