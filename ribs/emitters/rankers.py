"""Rankers for use across emitters.

The rankers implemented in this file are intended to be used with emitters.
Specifically, a ranker object should be initialized or passed in the emitters.
The ``Ranker`` object will define the :meth:`rank` method which returns the
result of a descending argsort of the solutions. It will also define a
:meth:`reset` method which resets the internal state of the object.

When specifying which ranker to use for each emitter, one could either pass in
the full name of a ranker, e.g. "ImprovementRanker", or the abbreviated name of
a ranker, e.g. "imp".
The supported abbreviations are:

* ``imp``: :class:`ImprovementRanker`
* ``2imp``: :class:`TwoStageImprovementRanker`
* ``rd``: :class:`RandomDirectionRanker`
* ``2rd``: :class:`TwoStageRandomDirectionRanker`
* ``obj``: :class:`ObjectiveRanker`
* ``2obj``: :class:`TwoStageObjectiveRanker`

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
_args = DocstringComponents(core_args)

_rank_args = f"""
Args:
    emitter (ribs.emitters.EmitterBase): Emitter that this ``ranker``
        object belongs to.
    archive (ribs.archives.ArchiveBase): Archive used by ``emitter``
        when creating and inserting solutions.
    rng (numpy.random.Generator): A random number generator.
{_args.solution_batch}
{_args.objective_batch}
{_args.measures_batch}
{_args.status_batch}
{_args.value_batch}
{_args.metadata_batch}

Returns:
    tuple(numpy.ndarray, numpy.ndarray): the first array (shape
    ``(batch_size,)``) is an array of indices representing a ranking of the
    solutions and the second array (shape ``(batch_size,)`` or (batch_size,
    n_values)``) is an array of values that this ranker used to rank the
    solutions. ``batch_size`` is the number of solutions and ``n_values`` is
    the number of values that the rank function used.
"""

_reset_args = """
Args:
    emitter (ribs.emitters.EmitterBase): Emitter that this ``ranker``
        object belongs to.
    archive (ribs.archives.ArchiveBase): Archive used by ``emitter``
        when creating and inserting solutions.
    rng (numpy.random.Generator): A random number generator.
"""


class RankerBase(ABC):
    """Base class for rankers.

    Every ranker has a :meth:`rank` method that returns a list of indices
    that indicate how the solutions should be ranked and a :meth:`reset` method
    that resets the internal state of the ranker
    (e.g. in :class:`ribs.emitters.rankers._random_direction_ranker`).

    Child classes are only required to override :meth:`rank`.
    """

    @abstractmethod
    def rank(self, emitter, archive, rng, solution_batch, objective_batch,
             measures_batch, status_batch, value_batch, metadata_batch):
        # pylint: disable=missing-function-docstring
        pass

    # Generates the docstring for rank
    rank.__doc__ = f"""
Generates a batch of indices that represents an ordering of ``solution_batch``.

{_rank_args}
    """

    def reset(self, emitter, archive, rng):
        # pylint: disable=missing-function-docstring
        pass

    reset.__doc__ = f"""
Resets the internal state of the ranker.

{_reset_args}
   """


class ImprovementRanker(RankerBase):
    """Ranks the solutions based on the improvement in the objective.

    This ranker ranks solutions in a single stage. The solutions are ranked by
    the improvement "value" described in :meth:`ribs.archives.ArchiveBase.add`.

    This ranker should not be used with CMA-ME. The improvement values for new
    solutions in CMA-ME are on a different scale from the improvement values for
    the other solutions, in that new solutions have an improvement value which
    is simply their objective, while other solutions have an improvement value
    which is the difference between their objective and the objective of their
    corresponding cell in the archive.
    """

    def rank(self, emitter, archive, rng, solution_batch, objective_batch,
             measures_batch, status_batch, value_batch, metadata_batch):
        # Note that lexsort sorts the values in ascending order,
        # so we use np.flip to reverse the sorted array.
        return np.flip(np.argsort(value_batch)), value_batch

    rank.__doc__ = f"""
Generates a list of indices that represents an ordering of solutions.

{_rank_args}
    """


class TwoStageImprovementRanker(RankerBase):
    """Ranks the solutions based on the improvement in the objective.

    This ranker originates in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_ in which it was referred to as
    "Improvement Emitter".
    This ranker ranks solutions in two stages. First, solutions are ranked by
    "status" -- those that found a new cell in the archive rank above those that
    improved an existing cell, which rank above those that were not added to the
    archive. Second, solutions are ranked by the "value" described in
    :meth:`ribs.archives.ArchiveBase.add`
    """

    def rank(self, emitter, archive, rng, solution_batch, objective_batch,
             measures_batch, status_batch, value_batch, metadata_batch):
        # To avoid using an array of tuples, ranking_values is a 2D array
        # [[status_0, value_0], ..., [status_n, value_n]]
        ranking_values = np.stack((status_batch, value_batch), axis=-1)

        # New solutions sort ahead of improved ones, which sort ahead of ones
        # that were not added.
        #
        # Since lexsort uses the last column/row as the key, we flip the
        # ranking_values along the last axis so that we are sorting by status
        # first.
        #
        # Note that lexsort sorts the values in ascending order,
        # so we use np.flip to reverse the sorted array.
        return (
            np.flip(np.lexsort(np.flip(ranking_values, axis=-1).T)),
            ranking_values,
        )

    rank.__doc__ = f"""
Generates a list of indices that represents an ordering of solutions.

{_rank_args}
    """


class RandomDirectionRanker(RankerBase):
    """Ranks the solutions based on projection onto a direction in measure
    space.

    This ranker originates in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_ as random direction emitter.
    The solutions are ranked solely based on their projection onto a random
    direction in the archive.

    To rank the solutions first by whether they were added, and then by
    the projection, refer to
    :class:`ribs.emitters.rankers.TwoStageRandomDirectionRanker`.
    """

    def __init__(self):
        super().__init__()
        self._target_measure_dir = None

    @property
    def target_measure_dir(self):
        """numpy.ndarray: ``(measure_dim,)`` array with the target measure
        direction vector."""
        return self._target_measure_dir

    @target_measure_dir.setter
    def target_measure_dir(self, value):
        self._target_measure_dir = value

    def rank(self, emitter, archive, rng, solution_batch, objective_batch,
             measures_batch, status_batch, value_batch, metadata_batch):
        if self._target_measure_dir is None:
            raise RuntimeError("target measure direction not set")
        projections = np.dot(measures_batch, self._target_measure_dir)
        # Sort only by projection; use np.flip to reverse the order
        return np.flip(np.argsort(projections)), projections

    rank.__doc__ = f"""
Ranks the solutions based on projection onto a direction in the archive.

{_rank_args}
    """

    def reset(self, emitter, archive, rng):
        ranges = archive.upper_bounds - archive.lower_bounds
        measure_dim = len(ranges)
        unscaled_dir = rng.standard_normal(measure_dim)
        self._target_measure_dir = unscaled_dir * ranges

    # Generates the docstring for reset.
    reset.__doc__ = f"""
Generates a random direction in the archive's measure space.

A random direction is sampled from a standard Gaussian -- since the standard
Gaussian is isotropic, there is equal probability for any direction. The
direction is then scaled to the archive bounds so that it is a random archive
direction.

{_reset_args}
   """


class TwoStageRandomDirectionRanker(RankerBase):
    """Similar to :class:`ribs.emitters.rankers.RandomDirectionRanker`, but the
    solutions are first ranked by whether they are added, then by their
    projection onto a random direction in the archive space.

    This ranker originates from `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_ as RandomDirectionEmitter.
    """

    def __init__(self):
        super().__init__()
        self._target_measure_dir = None

    @property
    def target_measure_dir(self):
        """numpy.ndarray: ``(measure_dim,)`` array with the target measure
        direction vector."""
        return self._target_measure_dir

    @target_measure_dir.setter
    def target_measure_dir(self, value):
        self._target_measure_dir = value

    def rank(self, emitter, archive, rng, solution_batch, objective_batch,
             measures_batch, status_batch, value_batch, metadata_batch):
        if self._target_measure_dir is None:
            raise RuntimeError("target measure direction not set")
        projections = np.dot(measures_batch, self._target_measure_dir)

        # To avoid using an array of tuples, ranking_values is a 2D array
        # [[status_0, projection_0], ..., [status_n, projection_n]]
        ranking_values = np.stack((status_batch, projections), axis=-1)

        # Sort by whether the solution was added into the archive,
        # followed by projection.
        return (
            np.flip(np.lexsort(np.flip(ranking_values, axis=-1).T)),
            ranking_values,
        )

    rank.__doc__ = f"""
Ranks the solutions first by whether they are added, then by their projection
onto a random direction in the archive.

{_rank_args}
    """

    def reset(self, emitter, archive, rng):
        ranges = archive.upper_bounds - archive.lower_bounds
        measure_dim = len(ranges)
        unscaled_dir = rng.standard_normal(measure_dim)
        self._target_measure_dir = unscaled_dir * ranges

    reset.__doc__ = RandomDirectionRanker.reset.__doc__


class ObjectiveRanker(RankerBase):
    """Ranks the solutions solely based on the objective values.

    This ranker originates in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_ in which it was part of the optimizing
    emitter.
    """

    def rank(self, emitter, archive, rng, solution_batch, objective_batch,
             measures_batch, status_batch, value_batch, metadata_batch):
        # Sort only by objective value.
        return np.flip(np.argsort(objective_batch)), objective_batch

    rank.__doc__ = f"""
Ranks the solutions based on their objective values.

{_rank_args}
    """


class TwoStageObjectiveRanker(RankerBase):
    """Similar to :class:`ribs.emitters.rankers.ObjectiveRanker`, but ranks
    newly added solutions before improved solutions.

    This ranker originates in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_ as OptimizingEmitter.
    """

    def rank(self, emitter, archive, rng, solution_batch, objective_batch,
             measures_batch, status_batch, value_batch, metadata_batch):
        # To avoid using an array of tuples, ranking_values is a 2D array
        # [[status_0, objective_0], ..., [status_0, objective_n]]
        ranking_values = np.stack((status_batch, objective_batch), axis=-1)

        # Sort by whether the solution was added into the archive, followed
        # by the objective values.
        return (
            np.flip(np.lexsort(np.flip(ranking_values, axis=-1).T)),
            ranking_values,
        )

    rank.__doc__ = f"""
Ranks the solutions based on their objective values, while prioritizing newly
added solutions.

{_rank_args}
    """


_NAME_TO_RANKER_MAP = {
    "ImprovementRanker": ImprovementRanker,
    "TwoStageImprovementRanker": TwoStageImprovementRanker,
    "RandomDirectionRanker": RandomDirectionRanker,
    "TwoStageRandomDirectionRanker": TwoStageRandomDirectionRanker,
    "ObjectiveRanker": ObjectiveRanker,
    "TwoStageObjectiveRanker": TwoStageObjectiveRanker,
    "imp": ImprovementRanker,
    "2imp": TwoStageImprovementRanker,
    "rd": RandomDirectionRanker,
    "2rd": TwoStageRandomDirectionRanker,
    "obj": ObjectiveRanker,
    "2obj": TwoStageObjectiveRanker
}


def _get_ranker(klass):
    """Returns a ranker class based on its name.

    ``klass`` can be a reference to the class of the ranker, the full name of
    a ranker, e.g. "ImprovementRanker", or the abbreviated name for a ranker
    such as "imp".

    Args:
        klass (callable or str): This parameter may either be a callable (e.g.
            a class or a lambda function) that takes in no parameters and
            returns an instance of :class:`RankerBase`, or it may be a full or
            abbreviated ranker name.

    Returns:
        The corresponding ranker class instance.
    """
    if isinstance(klass, str):
        if klass in _NAME_TO_RANKER_MAP:
            return _NAME_TO_RANKER_MAP[klass]()
        raise ValueError(f"`{klass}` is not the full or abbreviated "
                         "name of a valid ranker")
    if callable(klass):
        ranker = klass()
        if isinstance(ranker, RankerBase):
            return ranker
        raise ValueError(f"Callable `{klass}` did not return an instance "
                         "of RankerBase.")
    raise ValueError(f"`{klass}` is neither a callable nor a string")
