"""Package with rankers for use across emitters.

TODO [Add instructions on how to use]

.. autosummary::
    :toctree:

    ribs.emitters.rankers.RandomDirectionRanker
    ribs.emitters.rankers.TwoStageRandomDirectionRanker
    ribs.emitters.rankers.ObjectiveRanker
    ribs.emitters.rankers.TwoStageObjectiveRanker
    ribs.emitters.rankers.ImprovementRanker
    ribs.emitters.rankers.TwoStageImprovementRanker
    ribs.emitters.rankers.RankerBase
"""

from abc import ABC, abstractmethod
import numpy as np

from ribs.archives import AddStatus
from ribs._docstrings import DocstringComponents, _core_docs

__all__ = [
    "RandomDirectionRanker",
    "TwoStageRandomDirectionRanker",
    "ObjectiveRanker",
    "TwoStageObjectiveRanker",
    "ImprovementRanker",
    "TwoStageImprovementRanker",
    "RankerBase",
]

# Define common docstrings
_args = _core_docs["args"]
_returns = DocstringComponents(
    dict(indices="""
    1D array containing the index of each solution representing
    the ranking of the solutions
    """,))


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
        # pylint: disable=missing-function-docstring
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
{_returns.indices}
    """

    def reset(self, emitter, archive):
        # pylint: disable=missing-function-docstring
        pass

    # Generates the docstring for reset
    reset.__doc__ = f"""
Resets the internal state of the ranker.

Args:
{_args.emitter}
{_args.archive}
   """


class ImprovementRanker(RankerBase):
    # TODO implement
    pass


class TwoStageImprovementRanker(RankerBase):
    """Ranks the solutions based on the improvement in the objective.

    This ranker originates in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_ in which it was referred to as
    "Improvement Emitter".
    This ranker ranks solutions that improves the archive more higher.
    Moreover, new solutions are ranked above improved ones, which ranks above
    ones that were not added to the archive.
    """

    def rank(self, emitter, archive, solutions, objective_values,
             behavior_values, metadata, add_statuses, add_values):
        ranking_data = []  # TODO make this numpy
        for i, (status, add_value) in enumerate(zip(add_statuses, add_value)):
            added = bool(status)
            ranking_data.append((added, add_value, i))
            if status in (AddStatus.NEW, AddStatus.IMPROVE_EXISTING):
                new_sols += 1

        # New solutions sort ahead of improved ones, which sort ahead of ones
        # that were not added.
        ranking_data.sort(reverse=True)
        return [d[2] for d in ranking_data]

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
{_returns.indices}
    """


class RandomDirectionRanker(RankerBase):
    """Ranks the solutions based on projection onto a direction in
    behavior space.

    This ranker originates in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_ as RandomDirectionEmitter.
    The solutions are ranked solely based on their projection onto a random
    direction in behavior space.

    To rank the solutions first by whether they were added, and then by
    the projection, refer to
    :class:`ribs.emitters.rankers.TwoStageRandomDirectionRanker`.
    """

    def __init__(self, seed=None):
        self._target_behavior_dir = None
        self._rng = np.random.default_rng(seed)

    def rank(self, emitter, archive, solutions, objective_values,
             behavior_values, metadata, add_statuses, add_values):
        ranking_data = []
        for i, (beh, status) in enumerate(zip(behavior_values, add_statuses)):
            projection = np.dot(beh, self._target_behavior_dir)
            added = bool(status)

            ranking_data.append((added, projection, i))
            if added:
                new_sols += 1

        # Sort only by projection.
        ranking_data.sort(reverse=True, key=lambda x: x[1])
        return [d[2] for d in ranking_data]

    # Generates the docstring for rank
    rank.__doc__ = f"""
Ranks the soutions based on projection onto a direction in behavior space.

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
{_returns.indices}
    """

    def reset(self, emitter, archive):
        ranges = archive.upper_bounds - archive.lower_bounds
        behavior_dim = len(ranges)
        unscaled_dir = self._rng.standard_normal(behavior_dim)
        self._target_behavior_dir = unscaled_dir * ranges

    # Generates the docstring for reset
    reset.__doc__ = f"""
Generates a new random direction in the behavior space.

The direction is sampled from a standard Gaussian -- since the standard
Gaussian is isotropic, there is equal probability for any direction. The
direction is then scaled to the behavior space bounds.

Args:
{_args.emitter}
{_args.archive}
   """


class TwoStageRandomDirectionRanker(RankerBase):
    """Ranks the solutions based on projection onto a direction in
    behavior space, while prioritizing newly explored cells.

    This ranker originates from `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_ as RandomDirectionEmitter.
    The solutions are first ranked by whether they are added, then by
    their projection on to a random direction.

    To rank the solutions solely based on their projection onto a random
    direction in behavior space refer to
    :class:`ribs.emitters.rankers.TwoStageRandomDirectionRanker`.
    """

    def __init__(self, seed=None):
        self._target_behavior_dir = None
        self._rng = np.random.default_rng(seed)

    def rank(self, emitter, archive, solutions, objective_values,
             behavior_values, metadata, add_statuses, add_values):
        """Ranks the soutions based on TwoStageRandomDirectionRanker

        The solutions are first ranked by whether they are added, then by
        their projection on to a random direction.

        Args:
            emitter (ribs.emitters.EmitterBase):
            archive (ribs.archives.ArchiveBase): An archive to use when creating
                and inserting solutions. For instance, this can be
                :class:`ribs.archives.GridArchive`.
            solutions (numpy.ndarray): Array of solutions generated by this
                emitter's :meth:`ask()` method.
            objective_values (numpy.ndarray): 1D array containing the objective
                function value of each solution.
            behavior_values (numpy.ndarray): ``(n, <behavior space dimension>)``
                array with the behavior space coordinates of each solution.
            metadata (numpy.ndarray): 1D object array containing a metadata
                object for each solution.
            add_statuses ():
            add_values ():

        Returns:
            indices: which represent the descending order of the solutions
        """
        ranking_data = []
        for i, (beh, status) in enumerate(zip(behavior_values, add_statuses)):
            projection = np.dot(beh, self._target_behavior_dir)
            added = bool(status)

            ranking_data.append((added, projection, i))
            if added:
                new_sols += 1

        # Sort by whether the solution was added into the archive,
        # followed by projection.
        ranking_data.sort(reverse=True, key=lambda x: (x[0], x[1]))
        return [d[2] for d in ranking_data]

    # Generates the docstring for rank
    rank.__doc__ = f"""
Ranks the soutions first by whether they are added, then by their projection on
a random direction.

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
{_returns.indices}
    """

    def reset(self, emitter, archive):
        ranges = archive.upper_bounds - archive.lower_bounds
        behavior_dim = len(ranges)
        unscaled_dir = self._rng.standard_normal(behavior_dim)
        self._target_behavior_dir = unscaled_dir * ranges

    # Generates the docstring for reset
    reset.__doc__ = f"""
Generates a new random direction in the behavior space.

The direction is sampled from a standard Gaussian -- since the standard
Gaussian is isotropic, there is equal probability for any direction. The
direction is then scaled to the behavior space bounds.

Args:
{_args.emitter}
{_args.archive}
   """


class ObjectiveRanker(RankerBase):
    """Ranks the solutions based on the objective values

    This ranker originates in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_ in which it was referred to as
    the "optimizing emitter". This ranker ranks the solutions solely based
    on their objective values.
    """

    def rank(self, emitter, archive, solutions, objective_values,
             behavior_values, metadata, add_statuses, add_values):
        ranking_data = []
        for i, (obj, status) in enumerate(zip(objective_values, add_statuses)):
            added = bool(status)
            ranking_data.append((added, obj, i))
            if added:
                new_sols += 1

        # Sort only by objective value.
        ranking_data.sort(reverse=True, key=lambda x: x[1])
        return [d[2] for d in ranking_data]

    # Generates the docstring for rank
    rank.__doc__ = f"""
Ranks the soutions based on their objective values.

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
{_returns.indices}
    """


class TwoStageObjectiveRanker(RankerBase):
    """Ranks the solutions based on the objective values

    This ranker originates in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_ as OptimizingEmitter.
    We rank the solutions solely based on their objective values.
    """

    def rank(self, emitter, archive, solutions, objective_values,
             behavior_values, metadata, add_statuses, add_values):
        ranking_data = []
        for i, (obj, status) in enumerate(zip(objective_values, add_statuses)):
            added = bool(status)
            ranking_data.append((added, obj, i))
            if added:
                new_sols += 1

        # Sort by whether the solution was added into the archive, followed
        # by objective value.
        ranking_data.sort(reverse=True, key=lambda x: (x[0], x[1]))
        return [d[2] for d in ranking_data]

    # Generates the docstring for rank
    rank.__doc__ = f"""
Ranks the soutions based on their objective values, while prioritizing newly added solutions.

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
{_returns.indices}
    """
