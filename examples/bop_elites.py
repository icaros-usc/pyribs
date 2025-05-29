"""Example script of running BOP-Elites on the sphere domain. Before running
this example, please install pymoo with:

    pip install ribs[pymoo]
"""

import json
import time
from pathlib import Path
from typing import List

import fire
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from ribs.archives import GridArchive
from ribs.emitters import BayesianOptimizationEmitter
from ribs.schedulers import BayesianOptimizationScheduler
from ribs.visualize import grid_archive_heatmap


def sphere(solutions):
    """Sphere function evaluation and measures for a batch of solutions.

    Args:
        solutions (np.ndarray): (batch_size, dim) batch of solutions.
    Returns:
        objectives (np.ndarray): (batch_size,) batch of objectives.
        objective_grads (np.ndarray): (batch_size, solution_dim) batch of
            objective gradients.
        measures (np.ndarray): (batch_size, 2) batch of measures.
        measure_grads (np.ndarray): (batch_size, 2, solution_dim) batch of
            measure gradients.
    """
    dim = solutions.shape[1]

    # Shift the Sphere function so that the optimal value is at x_i = 2.048.
    sphere_shift = 5.12 * 0.4

    # Normalize the objective to the range [0, 100] where 100 is optimal.
    best_obj = 0.0
    worst_obj = (-5.12 - sphere_shift)**2 * dim
    raw_obj = np.sum(np.square(solutions - sphere_shift), axis=1)
    objectives = (raw_obj - worst_obj) / (best_obj - worst_obj) * 100

    # Compute gradient of the objective.
    objective_grads = -2 * (solutions - sphere_shift)

    # Calculate measures.
    clipped = solutions.copy()
    clip_mask = (clipped < -5.12) | (clipped > 5.12)
    clipped[clip_mask] = 5.12 / clipped[clip_mask]
    measures = np.concatenate(
        (
            np.sum(clipped[:, :dim // 2], axis=1, keepdims=True),
            np.sum(clipped[:, dim // 2:], axis=1, keepdims=True),
        ),
        axis=1,
    )

    # Compute gradient of the measures.
    derivatives = np.ones(solutions.shape)
    derivatives[clip_mask] = -5.12 / np.square(solutions[clip_mask])

    mask_0 = np.concatenate((np.ones(dim // 2), np.zeros(dim - dim // 2)))
    mask_1 = np.concatenate((np.zeros(dim // 2), np.ones(dim - dim // 2)))

    d_measure0 = derivatives * mask_0
    d_measure1 = derivatives * mask_1

    measure_grads = np.stack((d_measure0, d_measure1), axis=1)

    return (
        objectives,
        objective_grads,
        measures,
        measure_grads,
    )


def save_heatmap(archive, heatmap_path):
    """Saves a heatmap of the archive to the given path.

    Args:
        archive (GridArchive): The archive to save.
        heatmap_path: Image path for the heatmap.
    """
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=0, vmax=100, cmap="viridis")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close(plt.gcf())


# pylint:disable = too-many-positional-arguments, dangerous-default-value
def main(iterations: int = 1000,
         solution_dim: int = 4,
         search_nrestarts: int = 5,
         entropy_ejie: bool = False,
         upscale_schedule: List[List] = [[5, 5], [10, 10], [25, 25]],
         num_initial_samples: int = 20,
         initial_solutions: np.ndarray = None,
         batch_size: int = 8,
         num_emitters: int = 1,
         seed: int = 42,
         outdir: str = "test_logs",
         log_every: int = 20):
    """Main function for running BOP-Elites on the sphere domain."""
    # logdir for saving experiment data
    logdir = Path(outdir)
    logdir.mkdir(exist_ok=True)

    # The main grid archive that interacts with BOP-Elites. We start at the
    # lowest resolution from ``upscale_schedule``.
    max_bound = solution_dim / 2 * 5.12
    bounds = [(-max_bound, max_bound), (-max_bound, max_bound)]
    main_archive = GridArchive(
        solution_dim=solution_dim,
        dims=upscale_schedule[0],
        ranges=bounds,
        seed=seed,
    )
    # The passive archive that does NOT interact with BOP-Elites other than
    # storing all evaluated data seen so far. We do not scale the passive
    # archive, and it always stays at the final resolution.
    passive_archive = GridArchive(
        solution_dim=solution_dim,
        dims=upscale_schedule[-1],
        ranges=bounds,
        seed=seed,
    )

    # The main component of BOP-Elites
    emitters = [
        BayesianOptimizationEmitter(
            archive=main_archive,
            # BayesianOptimizationEmitter requires solution space bounds due to
            # Sobol sampling. i.e., bounds cannot be None.
            bounds=[(-10.24, 10.24)] * solution_dim,
            search_nrestarts=search_nrestarts,
            entropy_ejie=entropy_ejie,
            upscale_schedule=upscale_schedule,
            num_initial_samples=num_initial_samples,
            initial_solutions=initial_solutions,
            batch_size=batch_size,
            seed=seed + i,
        ) for i in range(num_emitters)
    ]

    # Scheduler for managing multiple emitters (in what order we ask them for
    # solutions etc.).
    scheduler = BayesianOptimizationScheduler(main_archive,
                                              emitters,
                                              result_archive=passive_archive)

    metrics = {
        "QD Score": {
            "x": [0],
            "y": [0.0],
        },
        "Archive Coverage": {
            "x": [0],
            "y": [0.0],
        },
        "Itr. Time": {
            "x": [0],
            "y": [0.0],
        }
    }

    for i in tqdm.trange(1, iterations + 1):
        itr_start_time = time.time()

        solutions = scheduler.ask()
        objectives, _, measures, _ = sphere(solutions)
        scheduler.tell(objectives, measures)

        final_itr = i == iterations
        if i % log_every == 0 or final_itr:
            if final_itr:
                scheduler.result_archive.data(return_type="pandas").to_csv(
                    logdir / "final_archive.csv")

            metrics["QD Score"]["x"].append(i)
            metrics["QD Score"]["y"].append(
                scheduler.result_archive.stats.qd_score)
            metrics["Archive Coverage"]["x"].append(i)
            metrics["Archive Coverage"]["y"].append(
                scheduler.result_archive.stats.coverage)
            metrics["Itr. Time"]["x"].append(i)
            metrics["Itr. Time"]["y"].append(time.time() - itr_start_time)

            tqdm.tqdm.write(
                f"Iteration {i} | Archive Coverage: "
                f"{metrics['Archive Coverage']['y'][-1] * 100:.3f}% "
                f"QD Score: {metrics['QD Score']['y'][-1]:.3f}")

            save_heatmap(
                scheduler.result_archive,
                logdir / f"heatmap_{i:08d}.png",
            )

    # Plot metrics.
    for metric, values in metrics.items():
        plt.plot(values["x"], values["y"])
        plt.title(metric)
        plt.xlabel("Iteration")
        plt.savefig(str(logdir / f"{metric.lower().replace(' ', '_')}.png"))
        plt.clf()
    with (logdir / "metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)


if __name__ == "__main__":
    fire.Fire(main)
