"""Provides the RandomDirectionEmitter."""
import numpy as np

from ribs.archives import AddStatus
from ribs.emitters._emitter_base import EmitterBase
from ribs.emitters.opt import CMAEvolutionStrategy


class RandomDirectionEmitter(EmitterBase):
    """CMA-ME random direction emitter.

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
                 restart_rule="no_improvement",
                 bounds=None,
                 batch_size=None,
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

        if restart_rule not in ["basic", "no_improvement"]:
            raise ValueError(f"Invalid restart_rule {restart_rule}")
        self._restart_rule = restart_rule

        self.opt = CMAEvolutionStrategy(sigma0, batch_size, self._solution_dim,
                                        self._archive.dtype)
        self.opt.reset(self._x0)
        self._num_parents = (self.opt.batch_size //
                             2 if selection_rule == "mu" else None)
        self._target_behavior_dir = self._generate_random_direction()

        # TODO: remove this
        self.restarts = 0

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

    def _generate_random_direction(self):
        """TODO."""
        # Note: Behavior space should always be bounded.
        # TODO: move upper bounds and lower bounds to ArchiveBase?
        ranges = self._archive.upper_bounds - self._archive.lower_bounds
        behavior_dim = len(ranges)
        unscaled_dir = self._rng.standard_normal(behavior_dim)
        return unscaled_dir * ranges

    def _check_restart(self, num_parents):
        if self._restart_rule == "no_improvement":
            # TODO: remove
            #  if num_parents == 0:
            #      print("Check restart: num_parents:", num_parents)
            return num_parents == 0
        return False

    def tell(self, solutions, objective_values, behavior_values):
        """TODO."""
        ranking_data = []
        new_sols = 0
        for i, (sol, obj, beh) in enumerate(
                zip(solutions, objective_values, behavior_values)):
            status, _ = self._archive.add(sol, obj, beh)
            projection = np.dot(beh, self._target_behavior_dir)
            ranking_data.append((status, projection, i))
            if status in (AddStatus.NEW, AddStatus.IMPROVE_EXISTING):
                new_sols += 1

        if self._selection_rule == "filter":
            # Sort by whether the solution was added into the archive, followed
            # by projection.
            key = lambda x: (bool(x[0]), x[1])
        elif self._selection_rule == "mu":
            # Sort only by projection.
            key = lambda x: x[1]
        ranking_data.sort(reverse=True, key=key)
        indices = [d[2] for d in ranking_data]

        num_parents = (new_sols if self._selection_rule == "filter" else
                       self._num_parents)

        self.opt.tell(solutions[indices], num_parents)

        # Check for reset.
        if (self.opt.check_stop(
            [projection for status, projection, i in ranking_data]) or
                self._check_restart(new_sols)):
            new_x0 = self._archive.get_random_elite()[0]
            self.opt.reset(new_x0)
            self._target_behavior_dir = self._generate_random_direction()
            self.restarts += 1
