"""Archives store solutions found by a QD algorithm.

.. autosummary::
    :toctree:

    ribs.archives.GridArchive
    ribs.archives.CVTArchive
    ribs.archives.SlidingBoundariesArchive
    ribs.archives.ArchiveBase
    ribs.archives.AddStatus
    ribs.archives.Elite
    ribs.archives.EliteBatch
    ribs.archives.ArchiveDataFrame
    ribs.archives.ArchiveStats
    ribs.archives.CQDScoreResult
"""
from ribs.archives._add_status import AddStatus
from ribs.archives._archive_base import ArchiveBase
from ribs.archives._archive_data_frame import ArchiveDataFrame
from ribs.archives._archive_stats import ArchiveStats
from ribs.archives._cqd_score_result import CQDScoreResult
from ribs.archives._cvt_archive import CVTArchive
from ribs.archives._elite import Elite, EliteBatch
from ribs.archives._grid_archive import GridArchive
from ribs.archives._sliding_boundaries_archive import SlidingBoundariesArchive

__all__ = [
    "GridArchive",
    "CVTArchive",
    "SlidingBoundariesArchive",
    "ArchiveBase",
    "AddStatus",
    "Elite",
    "ArchiveDataFrame",
    "ArchiveStats",
    "CQDScoreResult",
]
