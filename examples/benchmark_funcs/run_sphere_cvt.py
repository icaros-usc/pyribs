"""Runs CVT-MAP-Elites with the Sphere function."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi  # pylint: disable=no-name-in-module

from ribs.archives import CVTArchive
from ribs.emitters import GaussianEmitter
from ribs.optimizers import Optimizer


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


def sphere(sol):
    """The sphere function is the sum of squared components of the solution.

    We return the negative because MAP-Elites seeks to maximize.
    """
    return -np.sum(np.square(sol))


def main():
    """Initializes CVT, runs it with Sphere function, and plots results."""
    archive = CVTArchive([(-1, 1), (-1, 1)], 100, config={
        "samples": 10000,
    })
    plot_voronoi(archive, "voronoi.png")

    emitters = [
        GaussianEmitter([0.0] * 10, 0.1, archive, config={"batch_size": 4})
    ]
    opt = Optimizer([0.0] * 10, 0.1, archive, emitters)

    for i in range(10**5):
        sols = opt.ask()
        objs = [sphere(s) for s in sols]
        bcs = [(s[0], s[1]) for s in sols]

        opt.tell(sols, objs, bcs)

        if i % 1000 == 0:
            print('saving {}'.format(i))
            #  data = opt.archive.as_pandas()
            #  data = data.pivot('index-0', 'index-1', 'objective')

            #  ax = sns.heatmap(data)
            #  plt.savefig('images/arc-{:05d}'.format(i))
            #  plt.close()

    data = archive.as_pandas()
    print(data.head())


if __name__ == "__main__":
    main()
