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
* ``nslc``: :class:`NSLCRanker`
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
    NSLCRanker
    ObjectiveRanker
    RandomDirectionRanker
    TwoStageImprovementRanker
    TwoStageObjectiveRanker
    TwoStageRandomDirectionRanker
    RankerBase
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from ribs.archives import ArchiveBase
from ribs.emitters._emitter_base import (
    EmitterBase,  # Avoid cyclic import since `rankers` is under `ribs.emitters`.
)
from ribs.typing import BatchData, Int

# Since the docstrings in this module are auto-generated, Ruff does not see them.
# ruff: noqa: D102

__all__ = [
    "DensityRanker",
    "ImprovementRanker",
    "NoveltyRanker",
    "NSLCRanker",
    "ObjectiveRanker",
    "RandomDirectionRanker",
    "TwoStageImprovementRanker",
    "TwoStageObjectiveRanker",
    "TwoStageRandomDirectionRanker",
    "RankerBase",
]

_RANK_ARGS = """
Args:
    emitter: Emitter that this ``ranker`` object belongs to.
    archive: Archive used by ``emitter`` when creating and inserting solutions.
    data: Dict mapping from field names like ``solution`` and ``objective`` to arrays
        with data for the solutions.
    add_info: Information returned by an archive's
        :meth:`~ribs.archives.ArchiveBase.add` method.

Returns:
    The first array (shape ``(batch_size,)``) is an array of indices representing a
    ranking of the solutions and the second array (shape ``(batch_size,)`` or
    ``(batch_size, n_values)``) is an array of values that this ranker used to rank the
    solutions. ``batch_size`` is the number of solutions and ``n_values`` is the number
    of values that the rank function used.
"""

_RESET_ARGS = """
Args:
    emitter: Emitter that this ``ranker`` object belongs to.
    archive: Archive used by ``emitter`` when creating and inserting solutions.
"""


class RankerBase(ABC):
    """Base class for rankers.

    Every ranker has a :meth:`rank` method that returns a list of indices that indicate
    how the solutions should be ranked and a :meth:`reset` method that resets the
    internal state of the ranker (e.g. in
    :class:`~ribs.emitters.rankers.RandomDirectionRanker`).

    Child classes are only required to override :meth:`rank`.
    """

    def __init__(self, seed: Int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def rank(  # pylint: disable = missing-function-docstring
        self,
        emitter: EmitterBase,
        archive: ArchiveBase,
        data: BatchData,
        add_info: BatchData,
    ) -> tuple[np.ndarray, np.ndarray]:
        pass

    # Generates the docstring for rank
    rank.__doc__ = f"""
Generates a batch of indices that represents an ordering of ``data["solution"]``.

{_RANK_ARGS}
    """

    # pylint: disable-next = missing-function-docstring
    def reset(self, emitter: EmitterBase, archive: ArchiveBase) -> None:
        pass

    reset.__doc__ = f"""
Resets the internal state of the ranker.

{_RESET_ARGS}
   """


class ImprovementRanker(RankerBase):
    """Ranks solutions based on improvement in the objective.

    This ranker ranks solutions in a single stage. The solutions are ranked by the
    improvement "value" described in :meth:`ribs.archives.ArchiveBase.add`.

    This ranker should not be used with CMA-ME. The improvement values for new solutions
    in CMA-ME are on a different scale from the improvement values for the other
    solutions, in that new solutions have an improvement value which is simply their
    objective, while other solutions have an improvement value which is the difference
    between their objective and the objective of their corresponding cell in the
    archive.
    """

    def rank(
        self,
        emitter: EmitterBase,
        archive: ArchiveBase,
        data: BatchData,
        add_info: BatchData,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Note that argsort sorts the values in ascending order,
        # so we use np.flip to reverse the sorted array.
        return np.flip(np.argsort(add_info["value"])), add_info["value"]

    rank.__doc__ = f"""
Generates a list of indices that represents an ordering of solutions.

{_RANK_ARGS}
    """


class TwoStageImprovementRanker(RankerBase):
    """Ranks solutions based on status and on improvement in the objective.

    This ranker originates in `Fontaine 2020 <https://arxiv.org/abs/1912.02400>`_ in
    which it was referred to as "Improvement Emitter". This ranker ranks solutions in
    two stages. First, solutions are ranked by "status" -- those that found a new cell
    in the archive rank above those that improved an existing cell, which rank above
    those that were not added to the archive. Second, solutions are ranked by the
    "value" returned by archive ``add`` methods, such as
    :meth:`ribs.archives.GridArchive.add`.
    """

    def rank(
        self,
        emitter: EmitterBase,
        archive: ArchiveBase,
        data: BatchData,
        add_info: BatchData,
    ) -> tuple[np.ndarray, np.ndarray]:
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
    """Ranks solutions based on projection onto a direction in measure space.

    This ranker originates from the random direction emitter in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_. The solutions are ranked solely based on their
    projection onto a random direction in the archive.

    To rank the solutions first by whether they were added, and then by the projection,
    refer to :class:`ribs.emitters.rankers.TwoStageRandomDirectionRanker`.

    Note that archives used with this ranker must have ``lower_bounds`` and
    ``upper_bounds`` attributes.
    """

    def __init__(self, seed: Int | None = None) -> None:
        super().__init__(seed)
        self._target_measure_dir = None

    @property
    def target_measure_dir(self) -> np.ndarray:
        """``(measure_dim,)`` array with the target measure direction vector."""
        return self._target_measure_dir

    @target_measure_dir.setter
    def target_measure_dir(self, value: np.ndarray) -> None:
        self._target_measure_dir = value

    def rank(
        self,
        emitter: EmitterBase,
        archive: ArchiveBase,
        data: BatchData,
        add_info: BatchData,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._target_measure_dir is None:
            raise RuntimeError("target measure direction not set")
        projections = np.dot(data["measures"], self._target_measure_dir)
        # Sort only by projection; use np.flip to reverse the order
        return np.flip(np.argsort(projections)), projections

    rank.__doc__ = f"""
Ranks the solutions based on projection onto a direction in the archive.

{_RANK_ARGS}
    """

    def reset(self, emitter: EmitterBase, archive: ArchiveBase) -> None:
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
    """Ranks solutions based on status and on projection onto a direction in measure space.

    This ranker differs from :class:`ribs.emitters.rankers.RandomDirectionRanker` in
    that solutions are ranked in two stages: first by whether they are added, then by
    their projection onto a random direction in the archive space.

    This ranker originates from the random direction emitter in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_.

    Note that archives used with this ranker must have ``lower_bounds`` and
    ``upper_bounds`` attributes.
    """

    def __init__(self, seed: Int | None = None) -> None:
        super().__init__(seed)
        self._target_measure_dir = None

    @property
    def target_measure_dir(self) -> np.ndarray:
        """``(measure_dim,)`` array with the target measure direction vector."""
        return self._target_measure_dir

    @target_measure_dir.setter
    def target_measure_dir(self, value: np.ndarray) -> None:
        self._target_measure_dir = value

    def rank(
        self,
        emitter: EmitterBase,
        archive: ArchiveBase,
        data: BatchData,
        add_info: BatchData,
    ) -> tuple[np.ndarray, np.ndarray]:
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
Ranks solutions first by whether they are added, then by their projection onto a random
direction in the archive.

{_RANK_ARGS}
    """

    def reset(self, emitter: EmitterBase, archive: ArchiveBase) -> None:
        ranges = archive.upper_bounds - archive.lower_bounds
        measure_dim = len(ranges)
        unscaled_dir = self._rng.standard_normal(measure_dim)
        self._target_measure_dir = unscaled_dir * ranges

    reset.__doc__ = RandomDirectionRanker.reset.__doc__


class ObjectiveRanker(RankerBase):
    """Ranks solutions based on objective values.

    This ranker originates in the optimizing emitter in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_.
    """

    def rank(
        self,
        emitter: EmitterBase,
        archive: ArchiveBase,
        data: BatchData,
        add_info: BatchData,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Sort only by objective value.
        return np.flip(np.argsort(data["objective"])), data["objective"]

    rank.__doc__ = f"""
Ranks the solutions based on their objective values.

{_RANK_ARGS}
    """


class TwoStageObjectiveRanker(RankerBase):
    """Ranks solutions based on status and on objective values.

    This ranker is similar to :class:`ribs.emitters.rankers.ObjectiveRanker`, but ranks
    newly added solutions before improved solutions.

    This ranker originates in the optimizing emitter in `Fontaine 2020
    <https://arxiv.org/abs/1912.02400>`_.
    """

    def rank(
        self,
        emitter: EmitterBase,
        archive: ArchiveBase,
        data: BatchData,
        add_info: BatchData,
    ) -> tuple[np.ndarray, np.ndarray]:
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
Ranks solutions based on their objective values, while prioritizing newly added
solutions.

{_RANK_ARGS}
    """


class NoveltyRanker(RankerBase):
    """Ranks solutions based on novelty scores.

    This ranker can only be used with archives that return the ``novelty`` field from
    their ``add`` method, such as :meth:`ribs.archives.ProximityArchive.add`.
    """

    def rank(
        self,
        emitter: EmitterBase,
        archive: ArchiveBase,
        data: BatchData,
        add_info: BatchData,
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.flip(np.argsort(add_info["novelty"])), add_info["novelty"]

    rank.__doc__ = f"""
Ranks solutions based on novelty scores.

{_RANK_ARGS}
    """


class DensityRanker(RankerBase):
    """Ranks solutions based on density in measure space.

    Solutions with lower density are ranked first.

    This ranker can only be used with archives that return the ``density`` field from
    their ``add`` method, such as :meth:`ribs.archives.DensityArchive.add`.
    """

    def rank(
        self,
        emitter: EmitterBase,
        archive: ArchiveBase,
        data: BatchData,
        add_info: BatchData,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Lower density is better, so we sort as normal (i.e., ascending order).
        return np.argsort(add_info["density"]), add_info["density"]

    rank.__doc__ = f"""
Ranks solutions based on density in measure space.

{_RANK_ARGS}
    """


class NSLCRanker(RankerBase):
    """Ranks solutions with Novelty Search with Local Competition.

    This ranker implements the selection strategy from Novelty Search with Local
    Competition (NSLC) in `Lehman 2011b
    <https://web.archive.org/web/20111206122453/http://eplex.cs.ucf.edu/papers/lehman_gecco11.pdf>`_.
    Solutions are ranked via non-dominated sorting over two objectives:

    1. Novelty (higher is better) -- the average distance in measure space to the
       k-nearest neighbors in the archive.
    2. Local competition (higher is better) -- the number of nearest neighbors the
       solution beats on the objective.

    Within each non-dominated front, solutions are further ordered by crowding
    distance (higher is better), mirroring NSGA-II (`Deb 2002
    <https://ieeexplore.ieee.org/document/996017>`_). This rewards both exploration
    (novelty) and local exploitation (beating nearby elites) without requiring a
    weighted combination of the two.

    This ranker can only be used with archives that return both ``novelty`` and
    ``local_competition`` fields from their ``add`` method. Currently, this is
    :meth:`ribs.archives.ProximityArchive.add` when the archive is constructed with
    ``local_competition=True``.
    """

    def rank(
        self,
        emitter: EmitterBase,
        archive: ArchiveBase,
        data: BatchData,
        add_info: BatchData,
    ) -> tuple[np.ndarray, np.ndarray]:
        novelty = np.asarray(add_info["novelty"], dtype=np.float64)
        local_competition = np.asarray(add_info["local_competition"], dtype=np.float64)
        n = len(novelty)

        # Both objectives are "higher is better"; we negate to cast to the standard
        # "lower dominates" form expected by non-dominated sorting.
        costs = np.stack((-novelty, -local_competition), axis=-1)

        fronts = _fast_non_dominated_sort(costs)

        order = np.empty(n, dtype=np.int64)
        cursor = 0
        for front in fronts:
            if len(front) == 1:
                order[cursor] = front[0]
                cursor += 1
                continue

            # Within a front, order by crowding distance (descending) to prefer
            # boundary solutions and preserve diversity among ties.
            crowding = _crowding_distance(costs[front])
            sorted_front = front[np.argsort(-crowding, kind="stable")]
            order[cursor : cursor + len(front)] = sorted_front
            cursor += len(front)

        ranking_values = np.stack((novelty, local_competition), axis=-1)
        return order, ranking_values

    rank.__doc__ = f"""
Ranks solutions by non-dominated sorting over (novelty, local_competition), breaking
ties within a front by crowding distance.

{_RANK_ARGS}
    """


def _fast_non_dominated_sort(costs: np.ndarray) -> list[np.ndarray]:
    """Computes non-dominated fronts for a batch of 2D cost vectors.

    Uses the "fast non-dominated sort" algorithm from NSGA-II (Deb 2002). A solution
    ``i`` dominates ``j`` if ``costs[i] <= costs[j]`` element-wise and ``costs[i] <
    costs[j]`` in at least one dimension.

    Args:
        costs: ``(batch_size, n_objectives)`` array of costs to minimize.

    Returns:
        A list of 1D numpy arrays. Each array contains the indices belonging to one
        front; earlier fronts dominate later fronts.
    """
    n = len(costs)
    domination_count = np.zeros(n, dtype=np.int64)
    dominated_sets: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            # Check if i dominates j or vice versa. Two solutions that are equal on
            # all objectives do not dominate each other and end up in the same front.
            leq_ij = np.all(costs[i] <= costs[j])
            leq_ji = np.all(costs[j] <= costs[i])
            lt_ij = np.any(costs[i] < costs[j])
            lt_ji = np.any(costs[j] < costs[i])

            if leq_ij and lt_ij:
                dominated_sets[i].append(j)
                domination_count[j] += 1
            elif leq_ji and lt_ji:
                dominated_sets[j].append(i)
                domination_count[i] += 1

        if domination_count[i] == 0:
            fronts[0].append(i)

    current = 0
    while fronts[current]:
        next_front: list[int] = []
        for i in fronts[current]:
            for j in dominated_sets[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current += 1
        fronts.append(next_front)

    # Drop the trailing empty sentinel and cast each front to an array for indexing.
    return [np.asarray(front, dtype=np.int64) for front in fronts[:-1]]


def _crowding_distance(costs: np.ndarray) -> np.ndarray:
    """Computes NSGA-II crowding distance for solutions within a single front.

    Boundary solutions (min or max on any objective) receive an infinite crowding
    distance so they are always preserved first.

    Args:
        costs: ``(front_size, n_objectives)`` array of costs to minimize.

    Returns:
        ``(front_size,)`` array of crowding distances.
    """
    front_size, n_objectives = costs.shape
    distance = np.zeros(front_size, dtype=np.float64)

    for m in range(n_objectives):
        values = costs[:, m]
        order = np.argsort(values, kind="stable")
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        value_range = values[order[-1]] - values[order[0]]
        if value_range == 0:
            continue
        for k in range(1, front_size - 1):
            distance[order[k]] += (
                values[order[k + 1]] - values[order[k - 1]]
            ) / value_range

    return distance


_NAME_TO_RANKER_MAP = {
    "DensityRanker": DensityRanker,
    "ImprovementRanker": ImprovementRanker,
    "NoveltyRanker": NoveltyRanker,
    "NSLCRanker": NSLCRanker,
    "ObjectiveRanker": ObjectiveRanker,
    "RandomDirectionRanker": RandomDirectionRanker,
    "TwoStageImprovementRanker": TwoStageImprovementRanker,
    "TwoStageObjectiveRanker": TwoStageObjectiveRanker,
    "TwoStageRandomDirectionRanker": TwoStageRandomDirectionRanker,
    "density": DensityRanker,
    "imp": ImprovementRanker,
    "nov": NoveltyRanker,
    "nslc": NSLCRanker,
    "obj": ObjectiveRanker,
    "rd": RandomDirectionRanker,
    "2imp": TwoStageImprovementRanker,
    "2obj": TwoStageObjectiveRanker,
    "2rd": TwoStageRandomDirectionRanker,
}


def _get_ranker(
    klass: Callable[[Int | None], RankerBase] | str, seed: Int | None
) -> RankerBase:
    """Returns a ranker class based on its name.

    ``klass`` can be a reference to the class of the ranker, the full name of a ranker,
    e.g. "ImprovementRanker", or the abbreviated name for a ranker such as "imp".

    Args:
        klass: This parameter may either be a callable (e.g. a class or a lambda
            function) that takes in a seed parameter and returns an instance of
            :class:`RankerBase`, or it may be a full or abbreviated ranker name.
        seed: Seed for the ranker.

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
