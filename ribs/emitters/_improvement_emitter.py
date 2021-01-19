"""Provides the ImprovementEmitter."""
import numpy as np
from numba import jit

from ribs.archives import AddStatus
from ribs.emitters._emitter_base import EmitterBase
from ribs.emitters.opt import CMAEvolutionStrategy


class ImprovementEmitter(EmitterBase):
    """CMA-ME improvement emitter.

    Args:
        selection_rule: "mu" or "filter"
    Raises:
        ValueError: If the selection_rule is invalid.
    """

    def __init__(self,
                 x0,
                 sigma0,
                 archive,
                 selection_rule="filter",
                 bounds=None,
                 batch_size=64,
                 seed=None):
        self._x0 = x0
        self._sigma0 = sigma0
        EmitterBase.__init__(
            self,
            len(self._x0),
            bounds,
            batch_size,
            archive,
            seed,
        )

        if selection_rule not in ["mu", "filter"]:
            raise ValueError(f"Invalid selection_rule {selection_rule}")
        self._selection_rule = selection_rule
        self._num_parents = ((batch_size + 1) //
                             2 if selection_rule == "mu" else None)

        self.opt = CMAEvolutionStrategy(sigma0, batch_size, self._solution_dim)

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

    def ask(self):
        """TODO."""
        return self.opt.ask(self.lower_bounds, self.upper_bounds)

    def tell(self, solutions, objective_values, behavior_values):
        """TODO."""
        deltas = []
        new_sols = 0
        for i, sol in enumerate(solutions):
            obj = objective_values[i]
            beh = behavior_values[i]
            status, value = self._archive.add(sol, obj, beh)
            if status == AddStatus.NEW:
                deltas.append(
                    (1, value, i))  # New solutions sort ahead of improved ones.
                new_sols += 1
            elif status == AddStatus.IMPROVE_EXISTING:
                deltas.append((0, value, i))
        deltas.sort(reverse=True)
        indices = [d[2] for d in deltas]

        num_parents = (self._num_parents
                       if self._selection_rule == "mu" else new_sols)

        self.opt.tell(solutions[indices], num_parents)

        # Update archive
        # Handle restart
        # Rank solutions
        # Pass ranking to cma_es
