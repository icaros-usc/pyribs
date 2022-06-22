"""Provides data structures for storing one or more elites."""
from typing import NamedTuple

import numpy as np


class Elite(NamedTuple):
    """Represents a single elite in an archive.

    Note that since this class is a namedtuple, its fields may be accessed
    either by name or by integer indices.
    """

    #: Parameters of the elite's solution.
    solution: np.ndarray

    #: Objective value evaluation.
    objective: float

    #: 1D array of measure values.
    measures: np.ndarray

    #: Index of the elite in the archive (see :meth:`ArchiveBase.index_of`).
    index: int

    #: Metadata object for the elite.
    metadata: object


class EliteBatch(NamedTuple):
    """Represents a batch of elites.

    Each field is an array with dimensions ``(batch, ...)``. Refer to
    :class:`Elite` for the non-batched version of this class.
    """

    #: Batch of solutions -- shape ``(batch, solution_dim)``
    solution_batch: np.ndarray

    #: Batch of objectives -- shape ``(batch,)``
    objective_batch: np.ndarray

    #: Batch of measures -- shape ``(batch, measure_dim)``
    measures_batch: np.ndarray

    #: Batch of indices -- shape ``(batch,)``
    index_batch: np.ndarray

    #: Batch of metadata -- shape ``(batch,)``
    metadata_batch: np.ndarray
