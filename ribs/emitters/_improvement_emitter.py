"""Provides the ImprovementEmitter."""
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

        self.opt = CMAEvolutionStrategy(sigma0, batch_size, self._solution_dim,
                                        self._archive.dtype)
        self.opt.reset(self._x0)

    @property
    def x0(self):
        """numpy.ndarray: Center of the Gaussian distribution from which to
        sample solutions when the archive is empty."""
        # TODO
        return self._x0

    @property
    def sigma0(self):
        """float or numpy.ndarray: Standard deviation of the (diagonal) Gaussian
        distribution."""
        # TODO
        return self._sigma0

    def ask(self):
        """TODO."""
        return self.opt.ask(self.lower_bounds, self.upper_bounds)

    def tell(self, solutions, objective_values, behavior_values):
        """TODO."""
        deltas = []
        new_sols = 0
        for i, (sol, obj, beh) in enumerate(
                zip(solutions, objective_values, behavior_values)):
            status, value = self._archive.add(sol, obj, beh)
            # New solutions sort ahead of improved ones, which sort ahead of
            # ones that were not added.
            deltas.append((status, value, i))
            if status in (AddStatus.NEW, AddStatus.IMPROVE_EXISTING):
                new_sols += 1
        deltas.sort(reverse=True)
        indices = [d[2] for d in deltas]

        num_parents = (new_sols if self._selection_rule == "filter" else
                       self._num_parents)

        self.opt.tell(solutions[indices], num_parents)

        # Check for reset.
        if self.opt.check_stop(ranking_values=[d[1] for d in deltas]):
            new_x0 = self._archive.get_random_elite()[0]
            self.opt.reset(new_x0)
