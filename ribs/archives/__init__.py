"""Archive implementations and associated utilities.

The archive is a data structure that stores solutions generated by the QD
algorithm, along with any information relevant to solutions, such as objective
and measure values.

The archives in this subpackage are arranged in a one-layer hierarchy, with all
archives inheriting from :class:`~ribs.archives.ArchiveBase`. This subpackage
also contains several utilities associated with the archives, such as
:class:`~ribs.archives.ArchiveDataFrame`.

.. autosummary::
    :toctree:

    ribs.archives.GridArchive
    ribs.archives.GridUnstructuredArchive
    ribs.archives.CVTArchive
    ribs.archives.SlidingBoundariesArchive
    ribs.archives.ArchiveBase
    ribs.archives.ArrayStore
    ribs.archives.AddStatus
    ribs.archives.ArchiveDataFrame
    ribs.archives.ArchiveStats
    ribs.archives.CQDScoreResult
    ribs.archives.UnstructuredArchive
"""
from ribs.archives._add_status import AddStatus
from ribs.archives._archive_base import ArchiveBase
from ribs.archives._archive_data_frame import ArchiveDataFrame
from ribs.archives._archive_stats import ArchiveStats
from ribs.archives._array_store import ArrayStore
from ribs.archives._cqd_score_result import CQDScoreResult
from ribs.archives._cvt_archive import CVTArchive
from ribs.archives._grid_archive import GridArchive, GridUnstructuredArchive
from ribs.archives._sliding_boundaries_archive import SlidingBoundariesArchive
from ribs.archives._unstructured_archive import UnstructuredArchive

__all__ = [
    "GridArchive", "GridUnstructuredArchive", "CVTArchive",
    "SlidingBoundariesArchive", "ArchiveBase", "ArrayStore", "AddStatus",
    "ArchiveDataFrame", "ArchiveStats", "CQDScoreResult", "UnstructuredArchive"
]
