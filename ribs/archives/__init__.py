"""Archives store solutions found by a QD algorithm."""
from ribs.archives._cvt_archive import CVTArchive, CVTArchiveConfig
from ribs.archives._grid_archive import GridArchive, GridArchiveConfig
from ribs.archives._sliding_boundary_archive import (
    SlidingBoundaryArchive, SlidingBoundaryArchiveConfig)

__all__ = [
    "CVTArchive",
    "CVTArchiveConfig",
    "GridArchive",
    "GridArchiveConfig",
    "SlidingBoundaryArchive",
    "SlidingBoundaryArchiveConfig",
]
