"""Provides the AddStatus for all archives to use."""

from enum import IntEnum


class AddStatus(IntEnum):
    """Semantic meaning for statuses returned by :meth:`~ArchiveBase.add`.

    Statuses returned by the :meth:`~ArchiveBase.add` method of many archives are
    defined as integers. This class is an :class:`~enum.IntEnum` that provides a
    slightly more meaningful value with the following items:

    - ``NOT_ADDED`` (0): The solution given to :meth:`~ArchiveBase.add` was not added to
      the archive.
    - ``IMPROVE_EXISTING`` (1): The solution given to :meth:`~ArchiveBase.add` improved
      over an elite already in the archive.
    - ``NEW`` (2): The solution given to :meth:`~ArchiveBase.add` created a new elite in
      the archive.

    Example:

        Check the status of an add operation as follows (note that these examples use
        :meth:`~ArchiveBase.add_single` rather than :meth:`~ArchiveBase.add`)::

            from ribs.archives import AddStatus
            add_info = archive.add_single(solution, objective, measures)
            if add_info["status"] == AddStatus.NEW:
                # Do something if the solution made a new elite in the archive.

        To check whether the solution was added to the archive, the status can act like
        a bool::

            from ribs.archives import AddStatus
            add_info = archive.add_single(solution, objective, measures)
            if add_info["status"]:
                # Do something if the solution was added to the archive.

        Finally, there is an ordering on statuses::

            AddStatus.NEW > AddStatus.IMPROVE_EXISTING > AddStatus.NOT_ADDED
    """

    NOT_ADDED = 0
    IMPROVE_EXISTING = 1
    NEW = 2
