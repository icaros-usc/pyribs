import numpy as np

from ribs.emitters.rankers._ranker_base import RankerBase


class RandomDirectionRanker(RankerBase):

    def __init__(self):
        self._target_behavior_dir = self._generate_random_direction()

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

    def rank(self, emitter, archive, solutions, objective_values,
             behavior_values, metadata, statuses, add_values):
        ranking_data = []
        for i, (beh, status) in enumerate(zip(behavior_values, statuses)):
            projection = np.dot(beh, self._target_behavior_dir)
            added = bool(status)

            ranking_data.append((added, projection, i))
            if added:
                new_sols += 1

        # elif self._selection_rule == "mu":
        #     # Sort only by projection.
        #     def key(x):
        #         return x[1]

        # Sort by whether the solution was added into the archive, followed by projection.
        ranking_data.sort(reverse=True, key=lambda x: (x[0], x[1]))
        return [d[2] for d in ranking_data]

    def reset(self, target_behavior_dir):
        self._target_behavior_dir = target_behavior_dir
