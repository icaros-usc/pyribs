"""Archives store solutions found by a QD algorithm."""
from ribs.archives._cvt_archive import CVTArchive, CVTArchiveConfig
from ribs.archives._grid_archive import GridArchive, GridArchiveConfig

__all__ = [
    "CVTArchiveConfig",
    "CVTArchive",
    "GridArchiveConfig",
    "GridArchive",
]
