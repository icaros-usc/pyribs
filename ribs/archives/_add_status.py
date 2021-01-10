"""Provides the AddStatus for all archives to use."""
from enum import IntEnum


class AddStatus(IntEnum):
    """A status returned by the :meth:`~ArchiveBase.add` method in an archive.

    This class is an :class:`~enum.IntEnum` with the following values:

    - ``NOT_ADDED``: The solution given to :meth:`~ArchiveBase.add` was not
      added to the archive.
    - ``IMPROVE_EXISTING``: The solution given to :meth:`~ArchiveBase.add`
      improved an entry already in the archive.
    - ``NEW``: The solution given to :meth:`~ArchiveBase.add` created a new
      entry in the archive.

    Example:

        You can check the status of an add operation as follows::

            from ribs.archives import AddStatus
            status, _ = archive.add(solution, objective_value, behavior_values)
            if status == AddStatus.NEW:
                # Do something if the solution made a new entry in the archive.

        The status can also act like a bool if you only need to know whether the
        solution was added to the archive::

            from ribs.archives import AddStatus
            status, _ = archive.add(solution, objective_value, behavior_values)
            if status:
                # Do something if the solution was added to the archive.
    """
    NOT_ADDED = 0
    IMPROVE_EXISTING = 1
    NEW = 2
