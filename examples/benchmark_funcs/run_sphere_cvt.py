"""Runs CVT-MAP-Elites with the Sphere function."""
import numpy as np

from ribs.archives import CVTArchive
from ribs.emitters import GaussianEmitter
from ribs.optimizers import Optimizer


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

    emitters = [
        GaussianEmitter([0.0] * 10, 0.1, archive, config={"batch_size": 4})
    ]
    opt = Optimizer([0.0] * 10, 0.1, archive, emitters)

    for i in range(10**4):
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

    archive.heatmap("sphere-cvt-map-elites.png")
    data = archive.as_pandas()
    print(data.head())


if __name__ == "__main__":
    main()
