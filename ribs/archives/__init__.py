"""Archives store solutions found by a QD algorithm."""
from ribs.archives._grid_archive import GridArchive, GridArchiveConfig
from ribs.archives._individual import Individual

__all__ = [
    "Individual",
    "GridArchiveConfig",
    "GridArchive",
]
