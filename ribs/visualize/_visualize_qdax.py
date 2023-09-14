"""Provides visualization functions for QDax repertoires."""
import numpy as np

from ribs.archives import CVTArchive
from ribs.visualize._cvt_archive_3d_plot import cvt_archive_3d_plot
from ribs.visualize._cvt_archive_heatmap import cvt_archive_heatmap


def _as_cvt_archive(repertoire, ranges):
    """Converts a QDax repertoire into a CVTArchive."""

    # Construct a CVTArchive. We set solution_dim to 0 since we are only
    # plotting and do not need to have the solutions available.
    cvt_archive = CVTArchive(
        solution_dim=0,
        cells=repertoire.centroids.shape[0],
        ranges=ranges,
        custom_centroids=repertoire.centroids,
    )

    # Add everything to the CVTArchive.
    occupied = repertoire.fitnesses != -np.inf
    cvt_archive.add(
        np.empty((occupied.sum(), 0)),
        repertoire.fitnesses[occupied],
        repertoire.descriptors[occupied],
    )

    return cvt_archive


def qdax_repertoire_heatmap(
    repertoire,
    ranges,
    *args,
    **kwargs,
):
    # pylint: disable = line-too-long
    """Plots a heatmap of a QDax MapElitesRepertoire.

    Internally, this function converts a
    :class:`~qdax.core.containers.mapelites_repertoire.MapElitesRepertoire` into
    a :class:`~ribs.archives.CVTArchive` and plots it with
    :meth:`cvt_archive_heatmap`.

    Args:
        repertoire (qdax.core.containers.mapelites_repertoire.MapElitesRepertoire):
            A MAP-Elites repertoire output by an algorithm in QDax.
        ranges (array-like of (float, float)): Upper and lower bound of each
            dimension of the measure space, e.g. ``[(-1, 1), (-2, 2)]``
            indicates the first dimension should have bounds :math:`[-1,1]`
            (inclusive), and the second dimension should have bounds
            :math:`[-2,2]` (inclusive). This is needed since the
            MapElitesRepertoire does not store measure space bounds.
        *args: Positional arguments to pass to :meth:`cvt_archive_heatmap`.
        **kwargs: Keyword arguments to pass to :meth:`cvt_archive_heatmap`.
    """
    # pylint: enable = line-too-long

    cvt_archive_heatmap(_as_cvt_archive(repertoire, ranges), *args, **kwargs)


def qdax_repertoire_3d_plot(
    repertoire,
    ranges,
    *args,
    **kwargs,
):
    # pylint: disable = line-too-long
    """Plots a QDax MapElitesRepertoire with 3D measure space.

    Internally, this function converts a
    :class:`~qdax.core.containers.mapelites_repertoire.MapElitesRepertoire` into
    a :class:`~ribs.archives.CVTArchive` and plots it with
    :meth:`cvt_archive_3d_plot`.

    Args:
        repertoire (qdax.core.containers.mapelites_repertoire.MapElitesRepertoire):
            A MAP-Elites repertoire output by an algorithm in QDax.
        ranges (array-like of (float, float)): Upper and lower bound of each
            dimension of the measure space, e.g. ``[(-1, 1), (-2, 2), (-3, 3)]``
            indicates the first dimension should have bounds :math:`[-1,1]`
            (inclusive), the second dimension should have bounds :math:`[-2,2]`,
            and the third dimension should have bounds :math:`[-3,3]`
            (inclusive). This is needed since the MapElitesRepertoire does not
            store measure space bounds.
        *args: Positional arguments to pass to :meth:`cvt_archive_3d_plot`.
        **kwargs: Keyword arguments to pass to :meth:`cvt_archive_3d_plot`.
    """
    # pylint: enable = line-too-long

    cvt_archive_3d_plot(_as_cvt_archive(repertoire, ranges), *args, **kwargs)
