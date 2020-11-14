"""Runs MAP-Elites on the Sphere function."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.optimizers import Optimizer


def sphere(sol):
    """The sphere function is the sum of squared components of the solution.

    We return the negative because MAP-Elites seeks to maximize.
    """
    return -np.sum(np.square(sol))


def main():
    """Demo of MAP-Elites on the Sphere function."""
    archive = GridArchive((100, 100), [(-4, 4), (-4, 4)], seed=42)
    emitters = [GaussianEmitter([0.0] * 10, 0.1, archive, batch_size=4)]
    opt = Optimizer(archive, emitters)

    for i in range(10**5):
        sols = opt.ask()
        objs = [sphere(s) for s in sols]
        bcs = [(s[0], s[1]) for s in sols]

        opt.tell(objs, bcs)

        if i % 1000 == 0:
            print('saving {}'.format(i))

            #  data = opt.archive.as_pandas()
            #  data = data.pivot('index-0', 'index-1', 'objective')

            #  ax = sns.heatmap(data)
            #  plt.savefig('images/arc-{:05d}'.format(i))
            #  plt.close()

    data = archive.as_pandas()
    print(data.head())
    data = data.pivot('index-0', 'index-1', 'objective')
    sns.heatmap(data)
    plt.savefig('sphere-map-elites.png')


if __name__ == '__main__':
    main()
