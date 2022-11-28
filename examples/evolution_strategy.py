"""Generates CMA-ES animation on sphere function.

For gradient-based optimizers (inheriting from :class:`GradientOptBase`):
* ``adam``: :class:`AdamOpt`
* ``gradient_ascent``: :class:`GradientAscentOpt`

For evolution strategies (inheriting from :class:`EvolutionStrategyBase`):
* ``cma_es``: :class:`CMAEvolutionStrategy`
* ``sep_cma_es``: :class:`SeparableCMAEvolutionStrategy`
* ``lm_ma_es``: :class:`LMMAEvolutionStrategy`
* ``openai_es``: :class:`OpenAIEvolutionStrategy`

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

from ribs.emitters.opt import _get_es

mpl.rcParams["font.weight"] = "light"
mpl.rcParams["axes.titleweight"] = "light"
mpl.rcParams["axes.labelweight"] = "light"
mpl.rcParams["figure.titleweight"] = "light"


def rastrigin(x1, x2):
    return (10 * 2 + (x1**2 - 10 * np.cos(2 * np.pi * x1)) +
            (x2**2 - 10 * np.cos(2 * np.pi * x2)))


def sphere(x1, x2):
    return np.square(x1) + np.square(x2)


def main(func_name, es_name):
    """Generates the animation."""
    if func_name == "sphere":
        f = sphere
    elif func_name == "rastrigin":
        f = rastrigin

    itrs = 20

    x0 = np.array([0.5, 0.5])
    sigma0 = 0.07
    batch_size = 500
    lower_bound = np.array([-1, -1])
    upper_bound = np.array([1, 1])

    es_kwargs = {
        "sigma0": sigma0,
        "batch_size": batch_size,
        "solution_dim": x0.shape[0],
        "dtype": np.float32,
    }
    es = _get_es(es_name, **es_kwargs)
    es.reset(x0)

    xxs, yys = np.meshgrid(np.linspace(lower_bound[0], upper_bound[0], 250),
                           np.linspace(lower_bound[1], upper_bound[1], 250))

    outdir = Path("es_output")
    if not outdir.is_dir():
        outdir.mkdir()

    for i in tqdm.trange(itrs):
        solutions = es.ask(lower_bound, upper_bound)
        objective_batch = f(solutions[:, 0], solutions[:, 1])

        plt.figure(figsize=(6, 6))
        plt.contourf(xxs, yys, f(xxs, yys))
        plt.plot(solutions[:, 0],
                 solutions[:, 1],
                 "o",
                 color="white",
                 alpha=0.4,
                 zorder=1)
        plt.plot(solutions.mean(),
                 solutions.mean(),
                 "o",
                 color="red",
                 zorder=20,
                 ms=8)

        plt.title(es_name)
        plt.xlim(lower_bound[0], upper_bound[0])
        plt.ylim(lower_bound[0], upper_bound[1])
        plt.gca().set_aspect("equal")
        plt.tight_layout()

        plt.savefig(str(outdir / f"{es_name}_itr_{i:05d}.png"))
        plt.close()

        indices = np.argsort(objective_batch)
        # print(objective_batch)
        es.tell(indices, batch_size)


if __name__ == "__main__":
    fire.Fire(main)
