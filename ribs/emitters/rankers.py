"""Package with rankers for use across emitters.

The rankers implemented in this file is intended to be used with emitters.
Specifically, a ranker object should be initialized or passed in the emitters.
The ``Ranker`` object will define the :meth:`rank` method which returns the
result of an descending argsort of the solutions. It will also define a
:meth:`reset` method which resets the internal states of the object.

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
    dict(index_batch="""
    A batch of indicies representing a ranking of the solutions"""))


class RankerBase(ABC):
    """Base class for rankers.

    Every ranker has a :meth:`rank` method that returns a list of indices
    that indicate how the solutions should be ranked and a :meth:`reset` method
    that resets the internal state of the ranker
    (e.g. in :class:`ribs.emitters.rankers._random_direction_ranker`).

    Child classes are only required to override :meth:`rank`.
    """

    @abstractmethod
    def rank(self, emitter, archive, solution_batch, objective_batch,
             measure_batch, metadata, add_statuses, add_values):
        # pylint: disable=missing-function-docstring
        pass

    # Generates the docstring for rank
    rank.__doc__ = f"""
Generates a batch of indicies that represents an ordering of ``solution_batch``.

Args:
{_args.emitter}
{_args.archive}
{_args.solution_batch}
{_args.objective_batch}
{_args.measure_batch}
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

    def rank(self, emitter, archive, solution_batch, objective_batch,
             measure_batch, metadata, add_statuses, add_values):
        # New solutions sort ahead of improved ones, which sort ahead of ones
        # that were not added. Note that lexsort sorts the values in ascending
        # order, so we use numpy fancy indexing to reverse the sorted array.
        return np.lexsort((add_values, add_statuses))[::-1]

    # Generates the docstring for rank
    rank.__doc__ = f"""
Generates a list of indices that represents an ordering of solutions.

Args:
{_args.emitter}
{_args.archive}
{_args.solution_batch}
{_args.objective_batch}
{_args.measure_batch}
{_args.metadata}
{_args.add_statuses}
{_args.add_values}

Returns:
{_returns.indices}
    """


class RandomDirectionRanker(RankerBase):
    """Ranks the solutions based on projection onto a direction in
    measure space.

    This ranker originates in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_ as RandomDirectionEmitter.
    The solutions are ranked solely based on their projection onto a random
    direction in measure space.

    To rank the solutions first by whether they were added, and then by
    the projection, refer to
    :class:`ribs.emitters.rankers.TwoStageRandomDirectionRanker`.
    """

    def __init__(self, seed=None):
        self._target_measure_dir = None
        self._rng = np.random.default_rng(seed)

    @property
    def target_measure_dir(self):
        """numpy.ndarray: ``(measure_dim,)`` array with the target measure
        direction vector."""
        return self._target_measure_dir

    @target_measure_dir.setter
    def target_measure_dir(self, value):
        self._target_measure_dir = value

    def rank(self, emitter, archive, solution_batch, objective_batch,
             measure_batch, metadata, add_statuses, add_values):
        projections = np.dot(measure_batch, self._target_measure_dir)
        # Sort only by projection; use fancy indexing to reverse the order
        return np.lexsort((projections,))[::-1]

    # Generates the docstring for rank
    rank.__doc__ = f"""
Ranks the soutions based on projection onto a direction in behavior space.

Args:
{_args.emitter}
{_args.archive}
{_args.solution_batch}
{_args.objective_batch}
{_args.measure_batch}
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
        self._target_measure_dir = unscaled_dir * ranges

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

    @property
    def target_behavior_dir(self):
        """numpy.ndarray: ``(behavior_dim,)`` array with the target behavior
        direction vector."""
        return self._target_behavior_dir

    @target_behavior_dir.setter
    def target_behavior_dir(self, value):
        self._target_behavior_dir = value

    def rank(self, emitter, archive, solution_batch, objective_batch,
             measure_batch, metadata, add_statuses, add_values):
        projections = np.dot(measure_batch, self._target_behavior_dir)
        # Sort by whether the solution was added into the archive,
        # followed by projection.
        return np.lexsort((add_statuses, projections))[::-1]

    # Generates the docstring for rank
    rank.__doc__ = f"""
Ranks the soutions first by whether they are added, then by their projection on
a random direction in measure space.

Args:
{_args.emitter}
{_args.archive}
{_args.solution_batch}
{_args.objective_batch}
{_args.measure_batch}
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
Generates a new random direction in the measure space.

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

    def rank(self, emitter, archive, solution_batch, objective_batch,
             measure_batch, metadata, add_statuses, add_values):
        # Sort only by objective value.
        return np.argsort(objective_batch)[::-1]

    # Generates the docstring for rank
    rank.__doc__ = f"""
Ranks the soutions based on their objective values.

Args:
{_args.emitter}
{_args.archive}
{_args.solution_batch}
{_args.objective_batch}
{_args.measure_batch}
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

    def rank(self, emitter, archive, solution_batch, objective_batch,
             measure_batch, metadata, add_statuses, add_values):
        # Sort by whether the solution was added into the archive, followed
        # by the objective values.
        return np.lexsort((objective_values, add_statuses))[::-1]

    # Generates the docstring for rank
    rank.__doc__ = f"""
Ranks the soutions based on their objective values, while prioritizing newly added solutions.

Args:
{_args.emitter}
{_args.archive}
{_args.solution_batch}
{_args.objective_batch}
{_args.measure_batch}
{_args.metadata}
{_args.add_statuses}
{_args.add_values}

Returns:
{_returns.indices}
    """


name_to_ranker_map = {
    "RandomDirectionRanker": RandomDirectionRanker,
    "TwoStageRandomDirectionRanker": TwoStageRandomDirectionRanker,
    "ObjectiveRanker": ObjectiveRanker,
    "TwoStageObjectiveRanker": TwoStageObjectiveRanker,
    "ImprovementRanker": ImprovementRanker,
    "TwoStageImprovementRanker": TwoStageImprovementRanker,
}


def get_ranker(key):
    """Constructs and returns a ranker object

    Args:
        key (str): Full or abbreviated name of ranker.

    Returns:
        a ranker object
    """
    try:
        return name_to_ranker_map[key]()
    except KeyError as key_error:
        raise RuntimeError("Cannot find ranker with name " + key) from key_error
