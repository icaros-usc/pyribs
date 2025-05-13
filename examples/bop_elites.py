"""Example script of running BOP-Elites on the sphere domain."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from ribs.archives import GridArchive
from ribs.emitters import BayesianOptimizationEmitter
from ribs.schedulers import BayesianOptimizationScheduler
from ribs.visualize import grid_archive_heatmap


def sphere(solution_batch):
    """Sphere function evaluation and measures for a batch of solutions.

    Args:
        solution_batch (np.ndarray): (batch_size, dim) batch of solutions.
    Returns:
        objective_batch (np.ndarray): (batch_size,) batch of objectives.
        objective_grad_batch (np.ndarray): (batch_size, solution_dim) batch of
            objective gradients.
        measures_batch (np.ndarray): (batch_size, 2) batch of measures.
        measures_grad_batch (np.ndarray): (batch_size, 2, solution_dim) batch of
            measure gradients.
    """
    dim = solution_batch.shape[1]

    # Shift the Sphere function so that the optimal value is at x_i = 2.048.
    sphere_shift = 5.12 * 0.4

    # Normalize the objective to the range [0, 100] where 100 is optimal.
    best_obj = 0.0
    worst_obj = (-5.12 - sphere_shift)**2 * dim
    raw_obj = np.sum(np.square(solution_batch - sphere_shift), axis=1)
    objective_batch = (raw_obj - worst_obj) / (best_obj - worst_obj) * 100

    # Compute gradient of the objective.
    objective_grad_batch = -2 * (solution_batch - sphere_shift)

    # Calculate measures.
    clipped = solution_batch.copy()
    clip_mask = (clipped < -5.12) | (clipped > 5.12)
    clipped[clip_mask] = 5.12 / clipped[clip_mask]
    measures_batch = np.concatenate(
        (
            np.sum(clipped[:, :dim // 2], axis=1, keepdims=True),
            np.sum(clipped[:, dim // 2:], axis=1, keepdims=True),
        ),
        axis=1,
    )

    # Compute gradient of the measures.
    derivatives = np.ones(solution_batch.shape)
    derivatives[clip_mask] = -5.12 / np.square(solution_batch[clip_mask])

    mask_0 = np.concatenate((np.ones(dim // 2), np.zeros(dim - dim // 2)))
    mask_1 = np.concatenate((np.zeros(dim // 2), np.ones(dim - dim // 2)))

    d_measure0 = derivatives * mask_0
    d_measure1 = derivatives * mask_1

    measures_grad_batch = np.stack((d_measure0, d_measure1), axis=1)

    return (
        objective_batch,
        objective_grad_batch,
        measures_batch,
        measures_grad_batch,
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


# experiment params
@dataclass
class Params:
    """Experiment parameters for BOP-Elites, the sphere domain, and logging."""

    solution_dim: int
    total_itrs: int
    search_nrestarts: int
    entropy_ejie: bool
    upscale_schedule: List[List]
    num_initial_samples: int
    initial_solutions: np.ndarray
    batch_size: int
    num_emitters: int
    seed: int
    logdir: str
    log_every: int


def main():
    """Main function for running BOP-Elites on the sphere domain."""
    # logdir for saving experiment data
    trial_path = Path("test_logs")
    trial_path.mkdir(exist_ok=True)

    params = Params(
        solution_dim=4,
        total_itrs=1000,
        search_nrestarts=5,
        entropy_ejie=False,
        upscale_schedule=[[5, 5], [10, 10], [25, 25]],
        num_initial_samples=20,
        initial_solutions=None,
        batch_size=8,
        num_emitters=1,
        seed=42,
        logdir=trial_path,
        log_every=20,
    )

    # The main grid archive that interacts with BOP-Elites. We start at the
    # lowest resolution from ``upscale_schedule``.
    max_bound = params.solution_dim / 2 * 5.12
    bounds = [(-max_bound, max_bound), (-max_bound, max_bound)]
    main_archive = GridArchive(
        solution_dim=params.solution_dim,
        dims=params.upscale_schedule[0],
        ranges=bounds,
        seed=params.seed,
    )
    # The passive archive that does NOT interact with BOP-Elites other than
    # storing all evaluated data seen so far. We do not scale the passive
    # archive, and it always stays at the final resolution.
    passive_archive = GridArchive(
        solution_dim=params.solution_dim,
        dims=params.upscale_schedule[-1],
        ranges=bounds,
        seed=params.seed,
    )

    # The main component of BOP-Elites
    emitters = [
        BayesianOptimizationEmitter(
            archive=main_archive,
            # BayesianOptimizationEmitter requires solution space bounds due to
            # Sobol sampling. i.e., bounds cannot be None.
            bounds=[(-10.24, 10.24)] * params.solution_dim,
            search_nrestarts=params.search_nrestarts,
            entropy_ejie=params.entropy_ejie,
            upscale_schedule=params.upscale_schedule,
            num_initial_samples=params.num_initial_samples,
            initial_solutions=params.initial_solutions,
            batch_size=params.batch_size,
            seed=params.seed + i,
        ) for i in range(params.num_emitters)
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
        },
    }

    for i in tqdm.trange(1, params.total_itrs + 1):
        itr_start_time = time.time()

        sol = scheduler.ask()
        obj, _, meas, _ = sphere(sol)
        scheduler.tell(obj, meas)

        final_itr = i == params.total_itrs
        if i % params.log_every == 0 or final_itr:
            if final_itr:
                scheduler.result_archive.data(return_type="pandas").to_csv(
                    params.logdir / "final_archive.csv")

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
                passive_archive,
                params.logdir / f"heatmap_{i:08d}_passive.png",
            )

    # Plot metrics.
    for metric, values in metrics.items():
        plt.plot(values["x"], values["y"])
        plt.title(metric)
        plt.xlabel("Iteration")
        plt.savefig(
            str(params.logdir / f"{metric.lower().replace(' ', '_')}.png"))
        plt.clf()
    with (params.logdir / "metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)


if __name__ == "__main__":
    main()
