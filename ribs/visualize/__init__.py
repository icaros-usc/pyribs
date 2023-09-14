"""Visualization tools for pyribs.

These functions are similar to Matplotlib functions like
:func:`~matplotlib.pyplot.scatter` and :func:`~matplotlib.pyplot.pcolormesh`.
When called, these functions default to creating plots on the current axis.
After plotting, functions like :func:`~matplotlib.pyplot.xlabel` and
:func:`~matplotlib.pyplot.title` may be used to further modify the axis.
Alternatively, if using Matplotlib's object-oriented API, pass the `ax`
parameter to these functions.

.. note:: This module only works with ``ribs[visualize]`` installed. As such, it
    is not imported with ``import ribs``, and it must be explicitly imported
    with ``import ribs.visualize``.

.. autosummary::
    :toctree:

    ribs.visualize.cvt_archive_3d_plot
    ribs.visualize.cvt_archive_heatmap
    ribs.visualize.grid_archive_heatmap
    ribs.visualize.parallel_axes_plot
    ribs.visualize.sliding_boundaries_archive_heatmap
    ribs.visualize.qdax_repertoire_3d_plot
    ribs.visualize.qdax_repertoire_heatmap
"""
from ribs.visualize._cvt_archive_3d_plot import cvt_archive_3d_plot
from ribs.visualize._cvt_archive_heatmap import cvt_archive_heatmap
from ribs.visualize._grid_archive_heatmap import grid_archive_heatmap
from ribs.visualize._parallel_axes_plot import parallel_axes_plot
from ribs.visualize._sliding_boundaries_archive_heatmap import \
    sliding_boundaries_archive_heatmap
from ribs.visualize._visualize_qdax import (qdax_repertoire_3d_plot,
                                            qdax_repertoire_heatmap)

__all__ = [
    "cvt_archive_3d_plot",
    "cvt_archive_heatmap",
    "grid_archive_heatmap",
    "parallel_axes_plot",
    "sliding_boundaries_archive_heatmap",
    "qdax_repertoire_3d_plot",
    "qdax_repertoire_heatmap",
]
