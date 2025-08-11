"""Visualization tools for pyribs.

These functions are similar to Matplotlib functions like
:func:`~matplotlib.pyplot.scatter` and :func:`~matplotlib.pyplot.pcolormesh`. When
called, these functions default to creating plots on the current axis. After plotting,
functions like :func:`~matplotlib.pyplot.xlabel` and :func:`~matplotlib.pyplot.title`
may be used to further modify the axis. Alternatively, if using Matplotlib's
object-oriented API, pass the `ax` parameter to these functions.

.. note:: This module only works with ``ribs[visualize]`` installed. As such, it is not
    imported with ``import ribs``, and it must be explicitly imported with ``import
    ribs.visualize``.

.. autosummary::
    :toctree:

    cvt_archive_3d_plot
    cvt_archive_heatmap
    grid_archive_heatmap
    parallel_axes_plot
    proximity_archive_plot
    sliding_boundaries_archive_heatmap
    qdax_repertoire_3d_plot
    qdax_repertoire_heatmap
"""

from ribs.visualize._cvt_archive_3d_plot import cvt_archive_3d_plot
from ribs.visualize._cvt_archive_heatmap import cvt_archive_heatmap
from ribs.visualize._grid_archive_heatmap import grid_archive_heatmap
from ribs.visualize._parallel_axes_plot import parallel_axes_plot
from ribs.visualize._proximity_archive_plot import proximity_archive_plot
from ribs.visualize._sliding_boundaries_archive_heatmap import (
    sliding_boundaries_archive_heatmap,
)
from ribs.visualize._visualize_qdax import (
    qdax_repertoire_3d_plot,
    qdax_repertoire_heatmap,
)

__all__ = [
    "cvt_archive_3d_plot",
    "cvt_archive_heatmap",
    "grid_archive_heatmap",
    "parallel_axes_plot",
    "proximity_archive_plot",
    "sliding_boundaries_archive_heatmap",
    "qdax_repertoire_3d_plot",
    "qdax_repertoire_heatmap",
]
