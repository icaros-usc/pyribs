"""Utilities shared by the visualize module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.axes
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.typing import ColorType
from pandas import DataFrame

from ribs.archives import ArchiveDataFrame, CVTArchive, GridArchive


def retrieve_cmap(
    cmap: str | Sequence[ColorType] | matplotlib.colors.Colormap,
) -> matplotlib.colors.Colormap:
    """Retrieves colormap from Matplotlib."""
    if isinstance(cmap, str):
        return plt.get_cmap(cmap)
    if isinstance(cmap, Sequence):
        return matplotlib.colors.ListedColormap(cmap)
    return cmap  # ty: ignore[invalid-return-type]


def validate_heatmap_visual_args(
    aspect: Literal["auto", "equal"] | float | None,
    cbar: Literal["auto"] | None | Axes,
    measure_dim: int,
    valid_dims: list[int],
    error_msg_measure_dim: str,
) -> None:
    """Helper function to validate arguments passed to `*_archive_heatmap` plotting
    functions.

    Args:
        valid_dims: All specified valid archive dimensions that may be plotted into
            heatmaps.
        error_msg_measure_dim: Error message in ValueError if archive dimension plotting
            is not supported.

    Raises:
        ValueError: if validity checks for heatmap args fail
    """
    if aspect is not None and not (
        isinstance(aspect, float) or aspect in ["equal", "auto"]
    ):
        raise ValueError(
            f"Invalid arg aspect='{aspect}'; must be 'auto', 'equal', or float"
        )
    if measure_dim not in valid_dims:
        raise ValueError(error_msg_measure_dim)
    if not (cbar == "auto" or isinstance(cbar, matplotlib.axes.Axes) or cbar is None):
        raise ValueError(
            f"Invalid arg cbar={cbar}; must be 'auto', None, or matplotlib.axes.Axes"
        )


def validate_df(df: DataFrame | ArchiveDataFrame | None) -> ArchiveDataFrame:
    """Helper to validate the df passed into visualization functions."""

    # Cast to an ArchiveDataFrame in case someone passed in a regular DataFrame
    # or other object.
    if not isinstance(df, ArchiveDataFrame):
        df = ArchiveDataFrame(df)

    return df


def set_cbar(
    t: ScalarMappable,
    ax: Axes,
    cbar: Literal["auto"] | None | Axes,
    cbar_kwargs: dict | None,
) -> None:
    """Sets cbar on the Axes given cbar arg."""
    cbar_kwargs = {} if cbar_kwargs is None else cbar_kwargs
    if cbar == "auto":
        ax.figure.colorbar(t, ax=ax, **cbar_kwargs)
    elif isinstance(cbar, matplotlib.axes.Axes):
        cbar.figure.colorbar(t, ax=cbar, **cbar_kwargs)


def archive_heatmap_1d(
    archive: GridArchive | CVTArchive,
    *,
    cell_boundaries: np.ndarray,
    cell_objectives: np.ndarray,
    ax: Axes | None,
    cmap: matplotlib.colors.Colormap,
    aspect: Literal["auto", "equal"] | float,
    vmin: float | None,
    vmax: float | None,
    cbar: Literal["auto"] | None | Axes,
    cbar_kwargs: dict | None,
    rasterized: bool,
    pcm_kwargs: dict | None,
) -> Axes:
    """Plots a heatmap of a 1D archive.

    The y-bounds of the plot are set to [0, 1].

    Currently, this function supports GridArchive and CVTArchive.

    Args:
        archive: A 1D archive to plot.
        cell_boundaries: 1D array with the boundaries of the cells. Length should be
            archive.cells + 1.
        cell_objectives: Objectives of all cells in the archive, with the cells going
            from left to right. Length should be archive.cells. Empty cells should have
            objective of NaN.
        ax: See heatmap methods, e.g., grid_archive_heatmap.
        cmap: The colormap to use when plotting intensity. Unlike in user-facing
            functions, we expect that this arg was already passed through retrieve_cmap
            to get a colormap object.
        aspect: The aspect ratio of the heatmap. No default value for this function,
            unlike in user-facing functions.
        vmin: See heatmap methods, e.g., grid_archive_heatmap.
        vmax: See heatmap methods, e.g., grid_archive_heatmap.
        cbar: See heatmap methods, e.g., grid_archive_heatmap.
        cbar_kwargs: See heatmap methods, e.g., grid_archive_heatmap.
        rasterized: See heatmap methods, e.g., grid_archive_heatmap.
        pcm_kwargs: Additional kwargs to pass to :func:`~matplotlib.pyplot.pcolormesh`.
    Returns:
        The Axes where the heatmap was plotted. This may be used to further modify the
        plot.
    """
    # Initialize the axis.
    ax = plt.gca() if ax is None else ax
    ax.set_xlim(archive.lower_bounds[0], archive.upper_bounds[0])
    ax.set_aspect(aspect)

    # Turn off yticks; this is a 1D plot so only the x-axis matters.
    ax.set_yticks([])

    # Create the plot.
    pcm_kwargs = {} if pcm_kwargs is None else pcm_kwargs
    vmin = np.nanmin(cell_objectives) if vmin is None and not archive.empty else vmin
    vmax = np.nanmax(cell_objectives) if vmax is None and not archive.empty else vmax
    t = ax.pcolormesh(
        cell_boundaries,
        # y-bounds; needs a sensible default so that aspect ratio is consistent.
        np.array([0, 1]),
        cell_objectives[None, :],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=rasterized,
        **pcm_kwargs,
    )

    # Create color bar.
    set_cbar(t, ax, cbar, cbar_kwargs)
    return ax
