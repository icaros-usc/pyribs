import numpy as np

from _ranker import Ranker

class RandomDirectionRanker(Ranker):

    def rank(self, emitter, archive, solutions, objective_values,
             behavior_values, metadata, statuses, add_values):
        ranking_data = []
        for i, (beh, status) in enumerate(
                zip(behavior_values, statuses)):
            projection = np.dot(beh, self._target_behavior_dir)
            added = bool(status)

            ranking_data.append((added, projection, i))
            if added:
                new_sols += 1

        # Sort by whether the solution was added into the archive, followed
        # by projection.
        def key(x):
            return (x[0], x[1])

        ranking_data.sort(reverse=True, key=key)
        indices = [d[2] for d in ranking_data]
        return indices

    def reset(self, target_behavior_dir):
        self._target_behavior_dir = target_behavior_dir
