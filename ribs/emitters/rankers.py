"""Rankers for use accross emitters.

The rankers implemented in this file are intended to be used with emitters.
Specifically, a ranker object should be initialized or passed in the emitters.
The ``Ranker`` object will define the :meth:`rank` method which returns the
result of a descending argsort of the solutions. It will also define a
:meth:`reset` method which resets the internal state of the object.

.. autosummary::
    :toctree:

    ribs.emitters.rankers.ImprovementRanker
    ribs.emitters.rankers.TwoStageImprovementRanker
    ribs.emitters.rankers.RandomDirectionRanker
    ribs.emitters.rankers.TwoStageRandomDirectionRanker
    ribs.emitters.rankers.ObjectiveRanker
    ribs.emitters.rankers.TwoStageObjectiveRanker
    ribs.emitters.rankers.RankerBase
"""

from abc import ABC, abstractmethod
import numpy as np

from ribs._docstrings import DocstringComponents, core_args

__all__ = [
    "ImprovementRanker",
    "TwoStageImprovementRanker",
    "RandomDirectionRanker",
    "TwoStageRandomDirectionRanker",
    "ObjectiveRanker",
    "TwoStageObjectiveRanker",
    "RankerBase",
]

# Define common docstrings
core_args["emitter"] = """
    emitter (ribs.emitters.EmitterBase): Emitter that this ``ranker``
        object belongs to.
    """
core_args["archive"] = """
    archive (ribs.archives.ArchiveBase): Archive used by ``emitter``
        when creating and inserting solutions.
    """
_args = DocstringComponents(core_args)
_returns = DocstringComponents(
    dict(indices="""
    Indices representing a ranking of the solutions"""))


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
             measures_batch, metadata, add_statuses, add_values):
        # pylint: disable=missing-function-docstring
        pass

    # Generates the docstring for rank
    rank.__doc__ = f"""
Generates a batch of indices that represents an ordering of ``solution_batch``.

Args:
{_args.emitter}
{_args.archive}
{_args.solution_batch}
{_args.objective_batch}
{_args.measures_batch}
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
    # TODO Implement this (in another PR)
    pass


class TwoStageImprovementRanker(RankerBase):
    """Ranks the solutions based on the improvement in the objective.

    This ranker originates in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_ in which it was referred to as
    "Improvement Emitter".
    This ranker ranks solutions in two stages. First, solutions are ranked by
    "status" -- those that found a new cell in the archive rank above those that
    improved an existing cell, which rank above those that were not added to the
    archive. Second, solutions are ranked by the "value" described in
    :meth:`ArchiveBase.add`.
    """

    def rank(self, emitter, archive, solution_batch, objective_batch,
             measures_batch, metadata, add_statuses, add_values):
        # New solutions sort ahead of improved ones, which sort ahead of ones
        # that were not added. Note that lexsort sorts the values in ascending
        # order, so we use np.flip to reverse the sorted array.
        return np.flip(np.lexsort((add_values, add_statuses)))

    rank.__doc__ = f"""
Generates a list of indices that represents an ordering of solutions.

Args:
{_args.emitter}
{_args.archive}
{_args.solution_batch}
{_args.objective_batch}
{_args.measures_batch}
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
    <https://arxiv.org/abs/1912.02400>`_ as random direction emitter.
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
             measures_batch, metadata, add_statuses, add_values):
        if not self.target_measure_dir:
            raise RuntimeError("target measure direction not set")
        projections = np.dot(measures_batch, self._target_measure_dir)
        # Sort only by projection; use np.flip to reverse the order
        return np.flip(np.argsort(projections))

    rank.__doc__ = f"""
Ranks the soutions based on projection onto a direction in measure space.

Args:
{_args.emitter}
{_args.archive}
{_args.solution_batch}
{_args.objective_batch}
{_args.measures_batch}
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
Generates a random direction in the archive space.

A random direction is sampled from a standard Gaussian -- since the standard
Gaussian is isotropic, there is equal probability for any direction. The
direction is then scaled to the archive bounds so that it is a random archive
direction.

Args:
{_args.emitter}
{_args.archive}
   """


class TwoStageRandomDirectionRanker(RankerBase):
    """Similar to :class:`ribs.emitters.rankers.RandomDirectionRanker`,
    but the solutions are first ranked by whether they are added, then
    by their projection onto a random direction in the archive space.

    This ranker originates from `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_ as RandomDirectionEmitter.
    """

    def __init__(self, seed=None):
        self._target_measure_dir = None
        self._rng = np.random.default_rng(seed)

    @property
    def target_measure_dir(self):
        """numpy.ndarray: ``(behavior_dim,)`` array with the target behavior
        direction vector."""
        return self._target_measure_dir

    @target_measure_dir.setter
    def target_measure_dir(self, value):
        self._target_measure_dir = value

    def rank(self, emitter, archive, solution_batch, objective_batch,
             measures_batch, metadata, add_statuses, add_values):
        if not self.target_measure_dir:
            raise RuntimeError("target measure direction not set")
        projections = np.dot(measures_batch, self._target_measure_dir)
        # Sort by whether the solution was added into the archive,
        # followed by projection.
        return np.flip(np.lexsort((add_statuses, projections)))

    rank.__doc__ = f"""
Ranks the soutions first by whether they are added, then by their projection on
a random direction in measure space.

Args:
{_args.emitter}
{_args.archive}
{_args.solution_batch}
{_args.objective_batch}
{_args.measures_batch}
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

    reset.__doc__ = RandomDirectionRanker.reset.__doc__


class ObjectiveRanker(RankerBase):
    """Ranks the solutions solely based on the objective values.

    This ranker originates in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_ in which it was part of the
    the optimizing emitter.
    """

    def rank(self, emitter, archive, solution_batch, objective_batch,
             measures_batch, metadata, add_statuses, add_values):
        # Sort only by objective value.
        return np.flip(np.argsort(objective_batch))

    rank.__doc__ = f"""
Ranks the soutions based on their objective values.

Args:
{_args.emitter}
{_args.archive}
{_args.solution_batch}
{_args.objective_batch}
{_args.measures_batch}
{_args.metadata}
{_args.add_statuses}
{_args.add_values}

Returns:
{_returns.indices}
    """


class TwoStageObjectiveRanker(RankerBase):
    """Similar to :class:`ribs.emitters.rankers.ObjectiveRanker`,
    but ranks newly added solutions before improved solutions.

    This ranker originates in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_ as OptimizingEmitter.
    """

    def rank(self, emitter, archive, solution_batch, objective_batch,
             measures_batch, metadata, add_statuses, add_values):
        # Sort by whether the solution was added into the archive, followed
        # by the objective values.
        return np.flip(np.lexsort((objective_batch, add_statuses)))

    rank.__doc__ = f"""
Ranks the soutions based on their objective values, while prioritizing newly added solutions.

Args:
{_args.emitter}
{_args.archive}
{_args.solution_batch}
{_args.objective_batch}
{_args.measures_batch}
{_args.metadata}
{_args.add_statuses}
{_args.add_values}

Returns:
{_returns.indices}
    """


_name_to_ranker_map = {
    "ImprovementRanker": ImprovementRanker,
    "TwoStageImprovementRanker": TwoStageImprovementRanker,
    "imp": ImprovementRanker,
    "2imp": TwoStageImprovementRanker,
    "RandomDirectionRanker": RandomDirectionRanker,
    "TwoStageRandomDirectionRanker": TwoStageRandomDirectionRanker,
    "rd": RandomDirectionRanker,
    "2rd": TwoStageRandomDirectionRanker,
    "ObjectiveRanker": ObjectiveRanker,
    "TwoStageObjectiveRanker": TwoStageObjectiveRanker,
    "obj": ObjectiveRanker,
    "2obj": TwoStageObjectiveRanker
}


def get_ranker(key):
    """Constructs and returns a ranker object

    Args:
        key (str): Full or abbreviated name of ranker.

    Returns:
        a ranker object
    """
    try:
        return _name_to_ranker_map[key]()
    except KeyError as key_error:
        raise RuntimeError("Cannot find ranker with name " + key) from key_error
