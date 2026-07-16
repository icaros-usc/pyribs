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
    return cmap


def validate_heatmap_visual_args(
    aspect: Literal["auto", "equal"] | float | None,
    cbar: Literal["auto"] | None | Axes,
    measure_dim: int,
    valid_dims: list[int],
    error_msg_measure_dim: str,
) -> None:
    """Helper function to validate arguments passed to `*_archive_heatmap` plotting functions.

    Args:
        aspect: See a visualization function like grid_archive_heatmap.
        cbar: See a visualization function like grid_archive_heatmap.
        measure_dim: See a visualization function like grid_archive_heatmap.
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


# Use this offset to prevent vmin and vmax from being too close to each other.
OBJECTIVE_OFFSET = 0.1


def compute_vmin_vmax(  # pylint: disable = too-many-return-statements
    vmin: float | None,
    vmax: float | None,
    objectives: np.ndarray,
) -> tuple[float, float]:
    """Computes vmin and vmax based on the user's args and objectives in the archive.

    Args:
        vmin: User-supplied value for vmin.
        vmax: User-supplied value for vmax.
        objectives: Array of objective values.

    Returns:
        Tuple containing the new vmin and vmax.

    Raises:
        ValueError: vmin and vmax were both passed in, but vmin is greater than vmax (it
            must be less than or equal to vmax).
    """
    has_objectives = len(objectives) > 0

    # Cache min and max objectives.
    if has_objectives:
        min_obj = np.min(objectives)
        max_obj = np.max(objectives)
    else:
        min_obj = None
        max_obj = None

    # Determine new_vmin and new_vmax. This depends on three conditions:
    # 1. What is the value of vmin?
    # 2. What is the value of vmax? (This is combined with (1) in the branches.)
    # 3. Are there any objectives present?
    if vmin is None and vmax is None:
        if has_objectives:
            # Neither vmin nor vmax were passed in, and there are objectives in the
            # archive, so we use min_obj and max_obj.
            if min_obj == max_obj:
                # We use strict equality here rather than isclose. isclose checks for a tiny
                # difference, and we are okay with tiny differences.
                #
                # Move the objectives apart since they are equal.
                return (min_obj - OBJECTIVE_OFFSET, max_obj + OBJECTIVE_OFFSET)
            else:
                # Here, the objectives are far away, so set them directly.
                return (min_obj, max_obj)
        else:
            # Neither vmin nor vmax were passed in, and there are no objectives, so we
            # can choose any default value.
            return (-OBJECTIVE_OFFSET, OBJECTIVE_OFFSET)
    elif vmin is not None and vmax is None:
        # vmin is passed in, but we need to decide how to set vmax.
        if has_objectives:
            if vmin < max_obj:
                # Ideally, we can just use max_obj as vmax.
                return (vmin, max_obj)
            else:
                # However, if vmin >= max_obj, we choose our own default.
                return (vmin, vmin + 2.0 * OBJECTIVE_OFFSET)
        else:
            # If there are no objectives, we choose our own default.
            return (vmin, vmin + 2.0 * OBJECTIVE_OFFSET)
    elif vmin is None and vmax is not None:
        # vmax is passed in, but we need to decide how to set vmin.
        if has_objectives:
            if min_obj < vmax:
                # Ideally, we can just use min_obj as vmin.
                return (min_obj, vmax)
            else:
                # However, if min_obj is >= vmax, we choose our own default.
                return (vmax - 2.0 * OBJECTIVE_OFFSET, vmax)
        else:
            # If there are no objectives, we choose our own default.
            return (vmax - 2.0 * OBJECTIVE_OFFSET, vmax)
    else:  # vmin is not None and vmax is not None
        # Both vmin and vmax are passed in. Take them as is, subject to verification.
        if vmax < vmin:
            raise ValueError(
                f"vmax ({vmax}) must be greater than or equal to vmin ({vmin})"
            )
        return (vmin, vmax)


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
    vmin, vmax = compute_vmin_vmax(
        vmin, vmax, cell_objectives[~np.isnan(cell_objectives)]
    )
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
