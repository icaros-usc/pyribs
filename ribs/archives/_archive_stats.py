"""Provides ArchiveStats."""


class ArchiveStats:
    """Tracks various metrics for an archive."""

    def __init__(self):
        self._occupied = 0

    @property
    def occupied(self):
        """int: Number of elites in the archive (i.e. occupied bins)."""
        return self._occupied

    def reset(self):
        """Resets the statistics."""

    def update(self):
        """Adds a new entry."""
        self._occupied += 1  # May not always happen.
