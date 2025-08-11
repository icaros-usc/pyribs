"""Rankers for use across emitters.

The rankers implemented in this file are intended to be used with emitters.
Specifically, a ranker object should be initialized or passed in the emitters. The
``Ranker`` object will define the :meth:`~RankerBase.rank` method that returns the
result of a descending argsort of the solutions. It will also define a
:meth:`~RankerBase.reset` method that resets the internal state of the object.

When specifying which ranker to use for each emitter, one could either pass in the full
name of a ranker, e.g., "ImprovementRanker", or the abbreviated name of a ranker, e.g.,
"imp". The supported abbreviations are:

* ``density``: :class:`DensityRanker`
* ``imp``: :class:`ImprovementRanker`
* ``nov``: :class:`NoveltyRanker`
* ``obj``: :class:`ObjectiveRanker`
* ``rd``: :class:`RandomDirectionRanker`
* ``2imp``: :class:`TwoStageImprovementRanker`
* ``2obj``: :class:`TwoStageObjectiveRanker`
* ``2rd``: :class:`TwoStageRandomDirectionRanker`

.. autosummary::
    :toctree:

    DensityRanker
    ImprovementRanker
    NoveltyRanker
    ObjectiveRanker
    RandomDirectionRanker
    TwoStageImprovementRanker
    TwoStageObjectiveRanker
    TwoStageRandomDirectionRanker
    RankerBase
"""

from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "DensityRanker",
    "ImprovementRanker",
    "NoveltyRanker",
    "ObjectiveRanker",
    "RandomDirectionRanker",
    "TwoStageImprovementRanker",
    "TwoStageObjectiveRanker",
    "TwoStageRandomDirectionRanker",
    "RankerBase",
]

_RANK_ARGS = """
Args:
    emitter (ribs.emitters.EmitterBase): Emitter that this ``ranker`` object belongs to.
    archive (ribs.archives.ArchiveBase): Archive used by ``emitter`` when creating and
        inserting solutions.
    data (dict): Dict mapping from field names like ``solution`` and ``objective`` to
        arrays with data for the solutions. Rankers at least assume the presence of the
        ``solution`` key.
    add_info (dict): Information returned by an archive's add() method.

Returns:
    tuple of (numpy.ndarray, numpy.ndarray): The first array (shape ``(batch_size,)``)
    is an array of indices representing a ranking of the solutions and the second array
    (shape ``(batch_size,)`` or (batch_size, n_values)``) is an array of values that
    this ranker used to rank the solutions. ``batch_size`` is the number of solutions
    and ``n_values`` is the number of values that the rank function used.
"""

_RESET_ARGS = """
Args:
    emitter (ribs.emitters.EmitterBase): Emitter that this ``ranker`` object belongs to.
    archive (ribs.archives.ArchiveBase): Archive used by ``emitter`` when creating and
        inserting solutions.
"""


class RankerBase(ABC):
    """Base class for rankers.

    Every ranker has a :meth:`rank` method that returns a list of indices that indicate
    how the solutions should be ranked and a :meth:`reset` method that resets the
    internal state of the ranker (e.g. in
    :class:`~ribs.emitters.rankers.RandomDirectionRanker`).

    Child classes are only required to override :meth:`rank`.
    """

    def __init__(self, seed=None):
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def rank(self, emitter, archive, data, add_info):
        # pylint: disable=missing-function-docstring
        pass

    # Generates the docstring for rank
    rank.__doc__ = f"""
Generates a batch of indices that represents an ordering of ``data["solution"]``.

{_RANK_ARGS}
    """

    def reset(self, emitter, archive):
        # pylint: disable=missing-function-docstring
        pass

    reset.__doc__ = f"""
Resets the internal state of the ranker.

{_RESET_ARGS}
   """


class ImprovementRanker(RankerBase):
    """Ranks the solutions based on the improvement in the objective.

    This ranker ranks solutions in a single stage. The solutions are ranked by the
    improvement "value" described in :meth:`ribs.archives.ArchiveBase.add`.

    This ranker should not be used with CMA-ME. The improvement values for new solutions
    in CMA-ME are on a different scale from the improvement values for the other
    solutions, in that new solutions have an improvement value which is simply their
    objective, while other solutions have an improvement value which is the difference
    between their objective and the objective of their corresponding cell in the
    archive.
    """

    def rank(self, emitter, archive, data, add_info):
        # Note that argsort sorts the values in ascending order,
        # so we use np.flip to reverse the sorted array.
        return np.flip(np.argsort(add_info["value"])), add_info["value"]

    rank.__doc__ = f"""
Generates a list of indices that represents an ordering of solutions.

{_RANK_ARGS}
    """


class TwoStageImprovementRanker(RankerBase):
    """Ranks the solutions based on the improvement in the objective.

    This ranker originates in `Fontaine 2020 <https://arxiv.org/abs/1912.02400>`_ in
    which it was referred to as "Improvement Emitter". This ranker ranks solutions in
    two stages. First, solutions are ranked by "status" -- those that found a new cell
    in the archive rank above those that improved an existing cell, which rank above
    those that were not added to the archive. Second, solutions are ranked by the
    "value" returned by archive ``add`` methods, such as
    :meth:`ribs.archives.GridArchive.add`.
    """

    def rank(self, emitter, archive, data, add_info):
        # To avoid using an array of tuples, ranking_values is a 2D array
        # [[status_0, value_0], ..., [status_n, value_n]]
        ranking_values = np.stack((add_info["status"], add_info["value"]), axis=-1)

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

{_RANK_ARGS}
    """


class RandomDirectionRanker(RankerBase):
    """Ranks the solutions based on projection onto a direction in measure space.

    This ranker originates from the random direction emitter in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_. The solutions are ranked solely based on their
    projection onto a random direction in the archive.

    To rank the solutions first by whether they were added, and then by the projection,
    refer to :class:`ribs.emitters.rankers.TwoStageRandomDirectionRanker`.
    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self._target_measure_dir = None

    @property
    def target_measure_dir(self):
        """numpy.ndarray: ``(measure_dim,)`` array with the target measure direction
        vector."""
        return self._target_measure_dir

    @target_measure_dir.setter
    def target_measure_dir(self, value):
        self._target_measure_dir = value

    def rank(self, emitter, archive, data, add_info):
        if self._target_measure_dir is None:
            raise RuntimeError("target measure direction not set")
        projections = np.dot(data["measures"], self._target_measure_dir)
        # Sort only by projection; use np.flip to reverse the order
        return np.flip(np.argsort(projections)), projections

    rank.__doc__ = f"""
Ranks the solutions based on projection onto a direction in the archive.

{_RANK_ARGS}
    """

    def reset(self, emitter, archive):
        ranges = archive.upper_bounds - archive.lower_bounds
        measure_dim = len(ranges)
        unscaled_dir = self._rng.standard_normal(measure_dim)
        self._target_measure_dir = unscaled_dir * ranges

    # Generates the docstring for reset.
    reset.__doc__ = f"""
Generates a random direction in the archive's measure space.

A random direction is sampled from a standard Gaussian -- since the standard Gaussian is
isotropic, there is equal probability for any direction. The direction is then scaled to
the archive bounds so that it is a random archive direction.

{_RESET_ARGS}
   """


class TwoStageRandomDirectionRanker(RankerBase):
    """Similar to :class:`ribs.emitters.rankers.RandomDirectionRanker`, but the
    solutions are first ranked by whether they are added, then by their projection onto
    a random direction in the archive space.

    This ranker originates from the random direction emitter in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_.
    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self._target_measure_dir = None

    @property
    def target_measure_dir(self):
        """numpy.ndarray: ``(measure_dim,)`` array with the target measure
        direction vector."""
        return self._target_measure_dir

    @target_measure_dir.setter
    def target_measure_dir(self, value):
        self._target_measure_dir = value

    def rank(self, emitter, archive, data, add_info):
        if self._target_measure_dir is None:
            raise RuntimeError("target measure direction not set")
        projections = np.dot(data["measures"], self._target_measure_dir)

        # To avoid using an array of tuples, ranking_values is a 2D array
        # [[status_0, projection_0], ..., [status_n, projection_n]]
        ranking_values = np.stack((add_info["status"], projections), axis=-1)

        # Sort by whether the solution was added into the archive,
        # followed by projection.
        return (
            np.flip(np.lexsort(np.flip(ranking_values, axis=-1).T)),
            ranking_values,
        )

    rank.__doc__ = f"""
Ranks the solutions first by whether they are added, then by their projection onto a
random direction in the archive.

{_RANK_ARGS}
    """

    def reset(self, emitter, archive):
        ranges = archive.upper_bounds - archive.lower_bounds
        measure_dim = len(ranges)
        unscaled_dir = self._rng.standard_normal(measure_dim)
        self._target_measure_dir = unscaled_dir * ranges

    reset.__doc__ = RandomDirectionRanker.reset.__doc__


class ObjectiveRanker(RankerBase):
    """Ranks the solutions solely based on their objective values.

    This ranker originates in the optimizing emitter in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_.
    """

    def rank(self, emitter, archive, data, add_info):
        # Sort only by objective value.
        return np.flip(np.argsort(data["objective"])), data["objective"]

    rank.__doc__ = f"""
Ranks the solutions based on their objective values.

{_RANK_ARGS}
    """


class TwoStageObjectiveRanker(RankerBase):
    """Similar to :class:`ribs.emitters.rankers.ObjectiveRanker`, but ranks newly added
    solutions before improved solutions.

    This ranker originates in the optimizing emitter in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_.
    """

    def rank(self, emitter, archive, data, add_info):
        # To avoid using an array of tuples, ranking_values is a 2D array
        # [[status_0, objective_0], ..., [status_0, objective_n]]
        ranking_values = np.stack((add_info["status"], data["objective"]), axis=-1)

        # Sort by whether the solution was added into the archive, followed
        # by the objective values.
        return (
            np.flip(np.lexsort(np.flip(ranking_values, axis=-1).T)),
            ranking_values,
        )

    rank.__doc__ = f"""
Ranks the solutions based on their objective values, while prioritizing newly added
solutions.

{_RANK_ARGS}
    """


class NoveltyRanker(RankerBase):
    """Ranks solutions based on novelty scores.

    This ranker can only be used with archives that return the ``novelty`` field from
    their ``add`` method, such as :meth:`ribs.archives.ProximityArchive.add`.
    """

    def rank(self, emitter, archive, data, add_info):
        return np.flip(np.argsort(add_info["novelty"])), add_info["novelty"]

    rank.__doc__ = f"""
Ranks solutions based on novelty scores.

{_RANK_ARGS}
    """


class DensityRanker(RankerBase):
    """Ranks solutions based on density in measure space.

    This ranker can only be used with archives that return the ``density`` field from
    their ``add`` method, such as :meth:`ribs.archives.DensityArchive.add`.
    """

    def rank(self, emitter, archive, data, add_info):
        # Lower density is better, so we sort as normal (i.e., ascending order).
        return np.argsort(add_info["density"]), add_info["density"]

    rank.__doc__ = f"""
Ranks solutions based on density in measure space.

{_RANK_ARGS}
    """


_NAME_TO_RANKER_MAP = {
    "DensityRanker": DensityRanker,
    "ImprovementRanker": ImprovementRanker,
    "NoveltyRanker": NoveltyRanker,
    "ObjectiveRanker": ObjectiveRanker,
    "RandomDirectionRanker": RandomDirectionRanker,
    "TwoStageImprovementRanker": TwoStageImprovementRanker,
    "TwoStageObjectiveRanker": TwoStageObjectiveRanker,
    "TwoStageRandomDirectionRanker": TwoStageRandomDirectionRanker,
    "density": DensityRanker,
    "imp": ImprovementRanker,
    "nov": NoveltyRanker,
    "obj": ObjectiveRanker,
    "rd": RandomDirectionRanker,
    "2imp": TwoStageImprovementRanker,
    "2obj": TwoStageObjectiveRanker,
    "2rd": TwoStageRandomDirectionRanker,
}


def _get_ranker(klass, seed):
    """Returns a ranker class based on its name.

    ``klass`` can be a reference to the class of the ranker, the full name of a ranker,
    e.g. "ImprovementRanker", or the abbreviated name for a ranker such as "imp".

    Args:
        klass (callable or str): This parameter may either be a callable (e.g. a class
            or a lambda function) that takes in no parameters and returns an instance of
            :class:`RankerBase`, or it may be a full or abbreviated ranker name.

    Returns:
        The corresponding ranker class instance.
    """
    if isinstance(klass, str):
        if klass in _NAME_TO_RANKER_MAP:
            return _NAME_TO_RANKER_MAP[klass](seed)
        raise ValueError(
            f"`{klass}` is not the full or abbreviated name of a valid ranker"
        )
    if callable(klass):
        ranker = klass(seed)
        if isinstance(ranker, RankerBase):
            return ranker
        raise ValueError(
            f"Callable `{klass}` did not return an instance of RankerBase."
        )
    raise ValueError(f"`{klass}` is neither a callable nor a string")
