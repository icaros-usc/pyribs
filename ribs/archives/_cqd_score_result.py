"""Provides CQDScoreResult."""
import dataclasses
import typing

import numpy as np


@dataclasses.dataclass
class CQDScoreResult:
    """Stores the result of running :meth:`~ArchiveBase.cqd_score`."""

    #: Number of times the score was computed.
    iterations: int

    #: The mean score. This is the result most users will need.
    mean: float

    #: Array of scores obtained on each iteration.
    scores: np.ndarray

    #: (iterations, n, measure_dim) array of target points used in the
    #: computation. If the user passed in an array of target_points, this
    #: will be a copy of that array.
    target_points: np.ndarray

    #: 1D array of penalties used in the computation. If the user passed in an
    #: array of penalties, this will be a copy of that array.
    penalties: np.ndarray

    #: Minimum objective passed into the method.
    obj_min: float

    #: Maximum objective passed into the method.
    obj_max: float

    #: Max distance passed into the method, or the one that was computed based
    #: on measure space bounds.
    dist_max: float

    #: Order of the norm for distance which is passed into the method. Refer to
    #: the ``ord`` argument in :func:`numpy.linalg.norm` for type info.
    dist_ord: typing.Any
