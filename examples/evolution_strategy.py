"""Generates CMA-ES animation on sphere function.

Usage:
    # Generate frames.
    python evolution_strategy.py <sphere or rastrigin> <evolution_strategy_name>

    # Stitch everything together.
    ffmpeg -r -6 -i "es_output/<evolution_strategy_name>_itr_%*.png" es_output/<evolution_strategy_name>
"""
from pathlib import Path

import fire
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

mpl.rcParams["font.weight"] = "light"
mpl.rcParams["axes.titleweight"] = "light"
mpl.rcParams["axes.labelweight"] = "light"
mpl.rcParams["figure.titleweight"] = "light"


def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)


def sphere(x):
    return np.sum(np.square(x), axis=1)


def main(func_name, es_name):
    """Generates the animation."""
    if func_name == "sphere":
        f = sphere
    elif func_name == "rastrigin":
        f = rastrigin

    xxs, yys = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))

    itrs = 13

    x0 = np.array([0.75, 0.75])
    sigma0 = 0.07
    batch_size = 500
    bounds = [(-1, 1), (-1, 1)]

    # Create GridArchive.
    archive = GridArchive(solution_dim=x0.shape[0],
                          dims=(100, 100),
                          ranges=bounds)

    # Create ES Emitters.
    emitters = [
        EvolutionStrategyEmitter(
            archive,
            x0=x0,
            sigma0=sigma0,
            batch_size=batch_size,
        )
    ]

    # Create Scheduler.
    scheduler = Scheduler(archive, emitters)

    outdir = Path("es_output")
    if not outdir.is_dir():
        outdir.mkdir()

    for i in tqdm.trange(itrs):
        solutions = scheduler.ask()
        objective_batch = -f(solutions)

        plt.figure(figsize=(6, 6))
        plt.contourf(xxs, yys, xxs**2 + yys**2)
        plt.plot(solutions[:, 0],
                 solutions[:, 1],
                 "o",
                 color="black",
                 alpha=0.4,
                 zorder=1)
        plt.plot(solutions.mean(),
                 solutions.mean(),
                 "o",
                 color="red",
                 zorder=20,
                 ms=8)

        plt.xlim(bounds[0][0], bounds[0][1])
        plt.ylim(bounds[1][0], bounds[1][1])
        plt.gca().set_aspect("equal")
        plt.tight_layout()

        plt.savefig(str(outdir / f"{es_name}_itr_{i:05d}.png"))
        plt.close()

        scheduler.tell(objective_batch, solutions)


if __name__ == "__main__":
    fire.Fire(main)
