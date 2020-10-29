"""Runs MAP-Elites on the Sphere function."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ribs.archives import GridArchive
from ribs.optimizers import Optimizer


def sphere(sol):
    """The sphere function is the sum of squared components of the solution.

    We return the negative because MAP-Elites seeks to maximize.
    """
    return -np.sum(np.square(sol))


if __name__ == '__main__':

    config = {
        "seed": 42,
        "batch_size": 4,
    }
    archive = GridArchive((100, 100), [(-4, 4), (-4, 4)], config=config)
    opt = Optimizer([0.0] * 10, 0.1, archive, config=config)

    for i in range(10**5):
        sols = opt.ask()
        objs = [sphere(s) for s in sols]
        bcs = [(np.sum(s[:5]), np.sum(s[5:])) for s in sols]

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
    data = data.pivot('index-0', 'index-1', 'objective')
    sns.heatmap(data)
    plt.savefig('sphere-map-elites.png')
