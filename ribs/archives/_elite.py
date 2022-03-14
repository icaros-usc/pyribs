"""Provides Elite."""
from typing import NamedTuple

import numpy as np


class Elite(NamedTuple):
    """Represents a single elite in an archive.

    Note that since this class is a namedtuple, its fields may be accessed
    either by name or by integer indices.
    """

    #: Parameters of the elite's solution.
    sol: np.ndarray

    #: Objective value evaluation.
    obj: float

    #: Behavior values.
    beh: np.ndarray

    #: Index of the elite in the archive (see :meth:`ArchiveBase.get_index`).
    idx: np.int32
    # TODO: Should it be np.int32 or np.int64 or int?

    #: Metadata object for the elite.
    meta: object
