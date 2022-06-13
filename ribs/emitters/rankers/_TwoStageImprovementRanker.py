import numpy as np

from ribs.archives import AddStatus
from ribs.emitters.rankers._ranker_base import RankerBase


class TwoStageImprovementRanker(RankerBase):

    def rank(self, emitter, archive, solutions, objective_values,
             behavior_values, metadata, statuses, add_values):
        ranking_data = []
        for i, (status, add_value) in enumerate(zip(statuses, add_value)):
            added = bool(status)
            ranking_data.append((added, add_value, i))
            if status in (AddStatus.NEW, AddStatus.IMPROVE_EXISTING):
                new_sols += 1

        # New solutions sort ahead of improved ones, which sort ahead of ones
        # that were not added.
        ranking_data.sort(reverse=True)
        return [d[2] for d in ranking_data]
