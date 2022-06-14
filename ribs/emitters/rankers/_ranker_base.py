"""Provides the RankerBase."""
from abc import ABC, abstractmethod


class RankerBase(ABC):
    """Base class for rankers.

    Every ranker has an :meth:`rank` method that returns a list of indices
    that indicate how the solutions should be ranked and an :meth:`reset` method
    that resets the internal state of the ranker
    (i.e. in :class:`ribs.emitters.rankers._random_direction_ranker`).

    Child classes are only required to override :meth:`rank`.
    """

    @abstractmethod
    def rank(self, emitter, archive, solutions, objective_values,
             behavior_values, metadata, add_statuses, add_values):
        # TODO add comment
        pass

    def reset(self, archive, emitter):
        # TODO add comment
        return
