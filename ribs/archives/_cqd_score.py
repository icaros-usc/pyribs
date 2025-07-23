"""Utilities for computing CQD score."""

import dataclasses
import typing

import numpy as np

from ribs._utils import check_is_1d


@dataclasses.dataclass
class CQDScoreResult:
    """Stores the result of running :func:`cqd_score`."""

    #: Number of times the score was computed.
    iterations: int

    #: The mean score. This is the result most users will need.
    mean: float

    #: Array of scores obtained on each iteration.
    scores: np.ndarray

    #: (iterations, n, measure_dim) array of target points passed into the function.
    target_points: np.ndarray

    #: 1D array of penalties used in the computation. If an array of penalties was
    #: passed in, this will be a copy of that array.
    penalties: np.ndarray

    #: Minimum objective passed into the function.
    obj_min: float

    #: Maximum objective passed into the function.
    obj_max: float

    #: Max distance passed into the function.
    dist_max: float

    #: Order of the norm for distance that was passed into the function. Refer to the
    #: ``ord`` argument in :func:`numpy.linalg.norm` for type info.
    dist_ord: typing.Any


def cqd_score(
    archive,
    *,
    iterations,
    target_points,
    penalties,
    obj_min,
    obj_max,
    dist_max,
    dist_ord=None,
):
    """Computes the CQD score of an archive.

    The Continuous Quality Diversity (CQD) score was introduced in `Kent 2022
    <https://dl.acm.org/doi/10.1145/3520304.3534018>`_. Please see the
    :doc:`/examples/cqd_score` example for an example of how to call this function on an
    archive.

    Args:
        archive (ArchiveBase): Archive for which to compute the CQD score. The archive
            must implement the :meth:`~ribs.archives.ArchiveBase.data` method.
        iterations (int): Number of times to compute the CQD score.
        target_points (array-like): (iterations, n, measure_dim) array that lists n
            target points to use on each iteration.
        penalties (int or array-like): Number of penalty values over which to compute
            the score (the values are distributed evenly over the range [0,1]).
            Alternatively, this may be a 1D array that explicitly lists the penalty
            values. Known as :math:`\\theta` in Kent 2022.
        obj_min (float): Minimum objective value, used when normalizing the objectives.
        obj_max (float): Maximum objective value, used when normalizing the objectives.
        dist_max (float): Maximum distance between points in measure space.
        dist_ord: Order of the norm to use for calculating measure space distance; this
            is passed to :func:`numpy.linalg.norm` as the ``ord`` argument. See
            :func:`numpy.linalg.norm` for possible values. The default is to use
            Euclidean distance (L2 norm).
    Returns:
        CQDScoreResult: Object containing results of the CQD score calculations.
    Raises:
        ValueError: target_points or penalties is an array with the wrong shape.
    """
    target_points = np.copy(target_points)  # Copy since this is returned.
    if (
        target_points.ndim != 3
        or target_points.shape[0] != iterations
        or target_points.shape[2] != archive.measure_dim
    ):
        raise ValueError(
            "Expected target_points to be a 3D array with "
            f"shape ({iterations}, n, {archive.measure_dim}) "
            "(i.e. shape (iterations, n, measure_dim)) but it had "
            f"shape {target_points.shape}"
        )

    if np.isscalar(penalties):
        penalties = np.linspace(0, 1, penalties)
    else:
        penalties = np.copy(penalties)  # Copy since this is returned.
        check_is_1d(penalties, "penalties")

    objectives = archive.data("objective")
    measures = archive.data("measures")

    norm_objectives = objectives / (obj_max - obj_min)

    scores = np.zeros(iterations)

    for itr in range(iterations):
        # Distance calculation -- start by taking the difference between each measure i
        # and all the target points.
        distances = measures[:, None] - target_points[itr]

        # (len(archive), n_target_points) array of distances.
        distances = np.linalg.norm(distances, ord=dist_ord, axis=2)

        norm_distances = distances / dist_max

        for penalty in penalties:
            # Known as omega in Kent 2022 -- a (len(archive), n_target_points) array.
            values = norm_objectives[:, None] - penalty * norm_distances

            # (n_target_points,) array.
            max_values_per_target = np.max(values, axis=0)

            scores[itr] += np.sum(max_values_per_target)

    return CQDScoreResult(
        iterations=iterations,
        mean=np.mean(scores),
        scores=scores,
        target_points=target_points,
        penalties=penalties,
        obj_min=obj_min,
        obj_max=obj_max,
        dist_max=dist_max,
        dist_ord=dist_ord,
    )
