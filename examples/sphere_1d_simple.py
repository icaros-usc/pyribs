import numpy as np

from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter
from ribs.optimizers import Optimizer

archive = GridArchive([20], [(-1, 1)])
emitters = [ImprovementEmitter(archive, [0.0] * 10, 0.1)]
optimizer = Optimizer(archive, emitters)

for itr in range(1000):
    solutions = optimizer.ask()

    objectives = -np.sum(np.square(solutions), axis=1)
    bcs = solutions[:, 0]

    optimizer.tell(objectives, bcs)

import matplotlib.pyplot as plt
from ribs.visualize import grid_archive_heatmap

grid_archive_heatmap(archive, square=True)
plt.show()
