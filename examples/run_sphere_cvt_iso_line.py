"""Runs CVT-MAP-Elites with the Sphere function and IsoLine emitter."""
import numba as nb
import numpy as np

from ribs.archives import CVTArchive
from ribs.emitters import IsoLineEmitter
from ribs.optimizers import Optimizer
from ribs.visualize import cvt_archive_heatmap


@nb.njit
def sphere(sol):
    """The sphere function is the sum of squared components of the solution.

    We return the negative because MAP-Elites seeks to maximize.
    """
    return -np.sum(np.square(sol))


def main():
    """Inits CVT and IsoLine emitter, runs with Sphere, plots results."""
    archive = CVTArchive([(-1, 1), (-1, 1)], 1000, samples=10_000)
    emitters = [
        IsoLineEmitter([0.0] * 10,
                       archive,
                       iso_sigma=0.1,
                       line_sigma=0.2,
                       batch_size=4)
    ]
    opt = Optimizer(archive, emitters)

    for i in range(10**4):
        sols = opt.ask()
        objs = [sphere(s) for s in sols]
        bcs = [(s[0], s[1]) for s in sols]

        opt.tell(objs, bcs)

        if (i + 1) % 1000 == 0:
            print(f"Finished {i + 1} rounds")

    cvt_archive_heatmap(archive, filename="sphere-cvt-iso-line-map-elites.png")
    data = archive.as_pandas()
    print(data.head())


if __name__ == "__main__":
    main()
