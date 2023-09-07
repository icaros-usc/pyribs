"""Utilities shared by the visualize module."""
import matplotlib
import matplotlib.pyplot as plt


def retrieve_cmap(cmap):
    """Retrieves colormap from Matplotlib."""
    if isinstance(cmap, str):
        return plt.get_cmap(cmap)
    if isinstance(cmap, list):
        return matplotlib.colors.ListedColormap(cmap)
    return cmap


def validate_heatmap_visual_args(aspect, cbar, measure_dim, valid_dims,
                                 error_msg_measure_dim):
    """Helper function to validate arguments passed to `*_archive_heatmap`
    plotting functions.

    Args:
        valid_dims (list[int]): All specified valid archive dimensions that may
            be plotted into heatmaps.
        error_msg_measure_dim (str): Error message in ValueError if archive
            dimension plotting is not supported.

    Raises:
        ValueError: if validity checks for heatmap args fail
    """
    if aspect is not None and not (isinstance(aspect, float) or
                                   aspect in ["equal", "auto"]):
        raise ValueError(
            f"Invalid arg aspect='{aspect}'; must be 'auto', 'equal', or float")
    if measure_dim not in valid_dims:
        raise ValueError(error_msg_measure_dim)
    if not (cbar == "auto" or isinstance(cbar, matplotlib.axes.Axes) or
            cbar is None):
        raise ValueError(f"Invalid arg cbar={cbar}; must be 'auto', None, "
                         "or matplotlib.axes.Axes")


def set_cbar(t, ax, cbar, cbar_kwargs):
    """Sets cbar on the Axes given cbar arg."""
    cbar_kwargs = {} if cbar_kwargs is None else cbar_kwargs
    if cbar == "auto":
        ax.figure.colorbar(t, ax=ax, **cbar_kwargs)
    elif isinstance(cbar, matplotlib.axes.Axes):
        cbar.figure.colorbar(t, ax=cbar, **cbar_kwargs)
