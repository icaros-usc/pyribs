"""Archives store solutions found by a QD algorithm.

.. note:: After construction, each archive must be initialized by calling its
    ``initialize()`` method before it can be used. If you are using the
    optimizers in :mod:`ribs.optimizers`, this will be done automatically by
    the optimizer.
"""
from ribs.archives._archive_base import ArchiveBase
from ribs.archives._cvt_archive import CVTArchive
from ribs.archives._grid_archive import GridArchive
from ribs.archives._sliding_boundary_archive import SlidingBoundaryArchive

__all__ = [
    "GridArchive",
    "CVTArchive",
    "SlidingBoundaryArchive",
    "ArchiveBase",
]
