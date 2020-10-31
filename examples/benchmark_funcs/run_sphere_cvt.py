"""Runs CVT-MAP-Elites with the Sphere function."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi

from ribs.archives import CVTArchive


# TODO: Add heatmap
def plot_voronoi(archive, filename):
    """Plots a Voronoi diagram of the 2D archive and saves it to a file."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    # Add faraway points so that the outer regions of the diagram are filled in.
    points = np.append(archive.centroids,
                       [[999, 999], [-999, 999], [999, -999], [-999, -999]],
                       axis=0)
    vor = Voronoi(points)

    # Shade the regions.
    for region in vor.regions:
        if -1 not in region:
            polygon = [vor.vertices[i] for i in region]
            ax.fill(*zip(*polygon), ec="k", lw=0.5)

    # Plot the sample points and centroids.
    ax.plot(archive.samples[:, 0], archive.samples[:, 1], "o", c="gray", ms=1)
    ax.plot(archive.centroids[:, 0], archive.centroids[:, 1], "ko")

    fig.savefig(filename)


def main():
    """Initializes CVT, runs it with Sphere function, and plots results."""
    archive = CVTArchive([(-1, 1), (-1, 1)], 100, config={
        "samples": 10000,
    })

    plot_voronoi(archive, "voronoi.png")


if __name__ == "__main__":
    main()
