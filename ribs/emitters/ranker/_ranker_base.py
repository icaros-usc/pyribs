from abc import ABC, abstractmethod


class RankerBase(ABC):

    @abstractmethod
    def rank(self, emitter, archive, solutions, objective_values,
             behavior_values, metadata, statuses, add_values):
        # TODO add comment

    @abstractmethod
    def reset():
        # TODO add comment
