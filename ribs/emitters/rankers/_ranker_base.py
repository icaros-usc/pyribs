"""Provides the RankerBase."""
from abc import ABC, abstractmethod
from ribs._docstrings import _core_docs

_args = _core_docs["args"]

class RankerBase(ABC):
    """Base class for rankers.

    Every ranker has a :meth:`rank` method that returns a list of indices
    that indicate how the solutions should be ranked and a :meth:`reset` method
    that resets the internal state of the ranker
    (e.g. in :class:`ribs.emitters.rankers._random_direction_ranker`).

    Child classes are only required to override :meth:`rank`.
    """

    @abstractmethod
    def rank(self, emitter, archive, solutions, objective_values,
             behavior_values, metadata, add_statuses, add_values):
        pass

    # Generates the docstring for rank
    rank.__doc__ = f"""
Generates a list of indices that represents an ordering of solutions.

Args:
    {_args.emitter}
    {_args.archive}
    {_args.solutions}
    {_args.objective_values}
    {_args.behavior_values}
    {_args.metadata}
    {_args.add_statuses}
    {_args.add_values}

Returns:
    indices: which represent the descending order of the solutions
    """

    def reset(self, emitter, archive):
        pass


    # Generates the docstring for rank
    reset.__doc__ = f"""
Resets the internal state of the ranker.

    Args:
        {_args.emitter}
        {_args.archive}
   """
