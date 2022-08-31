"""Runs various QD algorithms on the Sphere function.

The sphere function in this example is adapted from Section 4 of Fontaine 2020
(https://arxiv.org/abs/1912.02400). Namely, each solution value is clipped to
the range [-5.12, 5.12], and the optimum is moved from [0,..] to [0.4 * 5.12 =
2.048,..]. Furthermore, the objectives are normalized to the range [0,
100] where 100 is the maximum and corresponds to 0 on the original sphere
function.

There are two measures in this example. The first is the sum of the first n/2
clipped values of the solution, and the second is the sum of the last n/2
clipped values of the solution. Having each measure depend equally on several
values in the solution space makes the problem more difficult (refer to
Fontaine 2020 for more info).

The supported algorithms are:
- `map_elites`: GridArchive with GaussianEmitter.
- `line_map_elites`: GridArchive with IsoLineEmitter.
- `cvt_map_elites`: CVTArchive with GaussianEmitter.
- `line_cvt_map_elites`: CVTArchive with IsoLineEmitter.
- `cma_me_imp`: GridArchive with EvolutionStrategyEmitter using
  TwoStageImprovmentRanker.
- `cma_me_imp_mu`: GridArchive with EvolutionStrategyEmitter using
  TwoStageImprovmentRanker and mu selection rule.
- `cma_me_rd`: GridArchive with EvolutionStrategyRanker using
  RandomDirectionRanker.
- `cma_me_rd_mu`: GridArchive with EvolutionStrategyEmitter using
  TwoStageRandomDirectionRanker and mu selection rule.
- `cma_me_opt`: GridArchive with EvolutionStrategyEmitter using ObjectiveRanker
  with mu selection rule.
- `cma_me_mixed`: GridArchive with EvolutionStrategyEmitter, where half (7) of
  the emitter are using TwoStageRandomDirectionRanker and half (8) are
  TwoStageImprovementRanker.
- `cma_mega`: GridArchive with GradientAborescenceEmitter.
- `cma_mega_adam`: GridArchive with GradientAborescenceEmitter using Adam
  Optimizer.
- `cma_mae`: GridArchive (learning_rate = 0.01) with EvolutionStrategyEmitter.
- `cma_maega`: GridArchive (learning_rate = 0.01) with GradientAborescenceEmitter.

All algorithms use 15 emitters, each with a batch size of 37. Each one runs for
4500 iterations for a total of 15 * 37 * 4500 ~= 2.5M evaluations.

Exceptions:
    - `cma_mega` and `cma_mega_adam` uses only one emitter and runs for 10,000
      iterations. This is to be consistent with the paper (`Fontaine 2021
      <https://arxiv.org/abs/2106.03894>`_) in which these algorithms were
      proposed.
    - `cma_mae` and `cma_maega`

Note that the CVTArchive in this example uses 10,000 cells, as opposed to the
250,000 (500x500) in the GridArchive, so it is not fair to directly compare
`cvt_map_elites` and `line_cvt_map_elites` to the other algorithms. However, the
other algorithms may be fairly compared because they use the same archive.

Outputs are saved in the `sphere_output/` directory by default. The archive is
saved as a CSV named `{algorithm}_{dim}_archive.csv`, while snapshots of the
heatmap are saved as `{algorithm}_{dim}_heatmap_{iteration}.png`. Metrics about
the run are also saved in `{algorithm}_{dim}_metrics.json`, and plots of the
metrics are saved in PNG's with the name `{algorithm}_{dim}_metric_name.png`.

To generate a video of the heatmap from the heatmap images, use a tool like
ffmpeg. For example, the following will generate a 6FPS video showing the
heatmap for cma_me_imp with 20 dims.

    ffmpeg -r 6 -i "sphere_output/cma_me_imp_20_heatmap_%*.png" \
        sphere_output/cma_me_imp_20_heatmap_video.mp4

Usage (see sphere_main function for all args):
    python sphere.py ALGORITHM DIM
Example:
    python sphere.py map_elites 20

    # To make numpy and sklearn run single-threaded, set env variables for BLAS
    # and OpenMP:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sphere.py map_elites 20
Help:
    python sphere.py --help
"""
import json
import time
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar

from ribs.archives import CVTArchive, GridArchive
from ribs.emitters import (EvolutionStrategyEmitter, GaussianEmitter,
                           GradientAborescenceEmitter, IsoLineEmitter)
from ribs.schedulers import Scheduler
from ribs.visualize import cvt_archive_heatmap, grid_archive_heatmap


def sphere(solution_batch):
    """Sphere function evaluation and measures for a batch of solutions.

    Args:
        solution_batch (np.ndarray): (batch_size, dim) batch of solutions.
    Returns:
        objective_batch (np.ndarray): (batch_size,) batch of objectives.
        measures_batch (np.ndarray): (batch_size, 2) batch of measures.
    """
    dim = solution_batch.shape[1]

    # Shift the Sphere function so that the optimal value is at x_i = 2.048.
    sphere_shift = 5.12 * 0.4

    # Normalize the objective to the range [0, 100] where 100 is optimal.
    best_obj = 0.0
    worst_obj = (-5.12 - sphere_shift)**2 * dim
    raw_obj = np.sum(np.square(solution_batch - sphere_shift), axis=1)
    objective_batch = (raw_obj - worst_obj) / (best_obj - worst_obj) * 100

    # Compute gradient of the objective
    objective_grad_batch = -2 * (solution_batch - sphere_shift)

    # Calculate measures.
    clipped = solution_batch.copy()
    clip_indices = np.where(np.logical_or(clipped > 5.12, clipped < -5.12))
    clipped[clip_indices] = 5.12 / clipped[clip_indices]
    measures_batch = np.concatenate(
        (
            np.sum(clipped[:, :dim // 2], axis=1, keepdims=True),
            np.sum(clipped[:, dim // 2:], axis=1, keepdims=True),
        ),
        axis=1,
    )

    # Compute gradient of the measures
    derivatives = np.ones(solution_batch.shape)
    derivatives[clip_indices] = -5.12 / np.square(solution_batch[clip_indices])

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


def create_scheduler(algorithm,
                     solution_dims,
                     archive_dims,
                     learning_rate,
                     use_result_archive=True,
                     seed=None):
    """Creates a scheduler based on the algorithm name.

    Args:
        algorithm (str): Name of the algorithm passed into sphere_main.
        solution_dims (int): Dimensionality of the sphere function.
        archive_dims (int): Dimensionality of the archive.
        learning_rate (float): Learning rate of archive.
        use_result_archive (bool): Whether to use a separate archive to store
            the results.
        seed (int): Main seed or the various components.
    Returns:
        ribs.schedulers.Scheduler: A ribs scheduler for running the algorithm.
    """
    max_bound = solution_dims / 2 * 5.12
    bounds = [(-max_bound, max_bound), (-max_bound, max_bound)]
    initial_sol = np.zeros(solution_dims)
    batch_size = 37
    num_emitters = 15
    mode = "batch"

    # Create archive.
    if algorithm in ["cvt_map_elites", "line_cvt_map_elites"]:
        archive = CVTArchive(solution_dim=solution_dims,
                             cells=10_000,
                             ranges=bounds,
                             samples=100_000,
                             use_kd_tree=True)
    else:
        archive = GridArchive(solution_dim=solution_dims,
                              dims=archive_dims,
                              ranges=bounds,
                              learning_rate=learning_rate,
                              threshold_min=0,
                              seed=seed)

    # Create result archive.
    result_archive = None
    if use_result_archive:
        result_archive = GridArchive(solution_dim=solution_dims,
                                     dims=archive_dims,
                                     ranges=bounds,
                                     seed=seed)

    # Create emitters. Each emitter needs a different seed, so that they do not
    # all do the same thing.
    emitter_seeds = [None] * num_emitters if seed is None else np.arange(
        seed, seed + num_emitters)
    if algorithm in ["map_elites", "cvt_map_elites"]:
        emitters = [
            GaussianEmitter(
                archive,
                initial_sol,
                0.5,
                batch_size=batch_size,
                seed=s,
            ) for s in emitter_seeds
        ]
    elif algorithm in ["line_map_elites", "line_cvt_map_elites"]:
        emitters = [
            IsoLineEmitter(
                archive,
                initial_sol,
                iso_sigma=0.1,
                line_sigma=0.2,
                batch_size=batch_size,
                seed=s,
            ) for s in emitter_seeds
        ]
    elif algorithm == "cma_me_mixed":
        emitters = [
            EvolutionStrategyEmitter(
                archive,
                initial_sol,
                0.5,
                "2rd",
                batch_size=batch_size,
                seed=s,
            ) for s in emitter_seeds[:7]
        ] + [
            EvolutionStrategyEmitter(
                archive,
                initial_sol,
                0.5,
                "2imp",
                batch_size=batch_size,
                seed=s,
            ) for s in emitter_seeds[7:]
        ]
    elif algorithm.startswith("cma_me_"):
        ranker, selection_rule, restart_rule = {
            "cma_me_imp": ("2imp", "filter", "no_improvement"),
            "cma_me_imp_mu": ("2imp", "mu", "no_improvement"),
            "cma_me_rd": ("2rd", "filter", "no_improvement"),
            "cma_me_rd_mu": ("2rd", "mu", "no_improvement"),
            "cma_me_opt": ("obj", "mu", "basic"),
        }[algorithm]
        emitters = [
            EvolutionStrategyEmitter(
                archive=archive,
                x0=initial_sol,
                sigma0=0.5,
                ranker=ranker,
                selection_rule=selection_rule,
                restart_rule=restart_rule,
                batch_size=batch_size,
                seed=s,
            ) for s in emitter_seeds
        ]
    elif algorithm == "cma_mega":
        # Note that only one emitter is used for cma_mega. This is to be
        # consistent with Fontaine 2021 <https://arxiv.org/abs/2106.03894>.
        emitters = [
            GradientAborescenceEmitter(
                archive,
                initial_sol,
                sigma0=10.0,
                step_size=1.0,
                grad_opt="gradient_ascent",
                selection_rule="mu",
                bounds=None,
                batch_size=batch_size - 1,  # 1 solution is returned by ask_dqd
                seed=emitter_seeds[0])
        ]
    elif algorithm == "cma_mega_adam":
        # Note that only one emitter is used for cma_mega_adam. This is to be
        # consistent with Fontaine 2021 <https://arxiv.org/abs/2106.03894>.
        emitters = [
            GradientAborescenceEmitter(
                archive,
                initial_sol,
                sigma0=10.0,
                step_size=0.002,
                grad_opt="adam",
                selection_rule="mu",
                bounds=None,
                batch_size=batch_size - 1,  # 1 solution is returned by ask_dqd
                seed=emitter_seeds[0])
        ]
    elif algorithm == "cma_mae":
        emitters = [
            EvolutionStrategyEmitter(
                archive=archive,
                x0=initial_sol,
                sigma0=0.5,
                ranker="imp",
                selection_rule="mu",
                restart_rule="basic",
                batch_size=batch_size,
                seed=s,
            ) for s in emitter_seeds
        ]
    elif algorithm in ["cma_maega"]:
        emitters = [
            GradientAborescenceEmitter(archive,
                                       initial_sol,
                                       sigma0=10.0,
                                       step_size=1.0,
                                       grad_opt="gradient_ascent",
                                       restart_rule="basic",
                                       bounds=None,
                                       batch_size=batch_size,
                                       seed=s) for s in emitter_seeds
        ]

    print(
        f"Created Scheduler for {algorithm} with learning rate {learning_rate} and add mode {mode}, "
        f"using solution dims {solution_dims} and archive dims {archive_dims}.")
    return Scheduler(archive, emitters, result_archive, add_mode=mode)


def save_heatmap(archive, heatmap_path):
    """Saves a heatmap of the archive to the given path.

    Args:
        archive (GridArchive or CVTArchive): The archive to save.
        heatmap_path: Image path for the heatmap.
    """
    if isinstance(archive, GridArchive):
        plt.figure(figsize=(8, 6))
        grid_archive_heatmap(archive, vmin=0, vmax=100)
        plt.tight_layout()
        plt.savefig(heatmap_path)
    elif isinstance(archive, CVTArchive):
        plt.figure(figsize=(16, 12))
        cvt_archive_heatmap(archive, vmin=0, vmax=100)
        plt.tight_layout()
        plt.savefig(heatmap_path)
    plt.close(plt.gcf())


def sphere_main(algorithm,
                dim=None,
                itrs=None,
                archive_dims=None,
                learning_rate=None,
                outdir="sphere_output",
                log_freq=250,
                seed=None):
    """Demo on the Sphere function.

    Args:
        algorithm (str): Name of the algorithm.
        dim (int): Dimensionality of the sphere function.
        itrs (int): Iterations to run.
        outdir (str): Directory to save output.
        log_freq (int): Number of iterations to wait before recording metrics
            and saving heatmap.
        seed (int): Seed for the algorithm. By default, there is no seed.
    """
    # Use default dim for each algorithm.
    if dim is None:
        if algorithm in ["cma_mega", "cma_mega_adam"]:
            dim = 1_000
        elif algorithm in ["cma_mae", "cma_maega"]:
            dim = 100
        elif algorithm in [
                "map_elites", "line_map_elites", "cma_me_imp", "cma_me_imp_mu",
                "cma_me_rd", "cma_me_rd_mu", "cma_me_opt", "cma_me_mixed"
        ]:
            dim = 20

    # Use default itrs for each algorithm.
    if itrs is None:
        if algorithm in ["cma_mega", "cma_mega_adam", "cma_mae", "cma_maega"]:
            itrs = 10_000
        elif algorithm in [
                "map_elites", "line_map_elites", "cma_me_imp", "cma_me_imp_mu",
                "cma_me_rd", "cma_me_rd_mu", "cma_me_opt", "cma_me_mixed"
        ]:
            itrs = 4500

    # Use default archive_dim for each algorithm.
    if archive_dims is None:
        if algorithm in ["cma_mega", "cma_mega_adam", "cma_mae", "cma_maega"]:
            archive_dims = (100, 100)
        elif algorithm in [
                "map_elites", "line_map_elites", "cma_me_imp", "cma_me_imp_mu",
                "cma_me_rd", "cma_me_rd_mu", "cma_me_opt", "cma_me_mixed"
        ]:
            archive_dims = (500, 500)

    # Use default learning_rate for each algorithm.
    if learning_rate is None:
        if algorithm in ["cma_mae", "cma_maega"]:
            learning_rate = 0.01
        elif algorithm in [
                "map_elites", "line_map_elites", "cma_me_imp", "cma_me_imp_mu",
                "cma_me_rd", "cma_me_rd_mu", "cma_me_opt", "cma_me_mixed",
                "cma_mega", "cma_mega_adam"
        ]:
            learning_rate = 1.0

    name = f"{algorithm}_{dim}"
    outdir = Path(outdir)
    if not outdir.is_dir():
        outdir.mkdir()

    is_dqd = algorithm in ["cma_mega", "cma_mega_adam"]
    use_result_archive = algorithm in ["cma_mae", "cma_maega"]

    scheduler = create_scheduler(algorithm,
                                 dim,
                                 archive_dims,
                                 learning_rate,
                                 use_result_archive=use_result_archive,
                                 seed=seed)
    archive = scheduler.archive
    result_archive = scheduler.result_archive
    metrics = {
        "QD Score": {
            "x": [0],
            "y": [0.0],
        },
        "Archive Coverage": {
            "x": [0],
            "y": [0.0],
        },
    }

    non_logging_time = 0.0
    with alive_bar(itrs) as progress:
        save_heatmap(archive, str(outdir / f"{name}_heatmap_{0:05d}_main.png"))
        save_heatmap(result_archive,
                     str(outdir / f"{name}_heatmap_{0:05d}_result.png"))

        for itr in range(1, itrs + 1):
            itr_start = time.time()

            if is_dqd:
                solution_batch = scheduler.ask_dqd()
                (objective_batch, objective_grad_batch, measures_batch,
                 measures_grad_batch) = sphere(solution_batch)
                objective_grad_batch = np.expand_dims(objective_grad_batch,
                                                      axis=1)
                jacobian_batch = np.concatenate(
                    (objective_grad_batch, measures_grad_batch), axis=1)
                scheduler.tell_dqd(objective_batch, measures_batch,
                                   jacobian_batch)

            solution_batch = scheduler.ask()
            objective_batch, _, measures_batch, _ = sphere(solution_batch)

            scheduler.tell(objective_batch, measures_batch)
            non_logging_time += time.time() - itr_start
            progress()

            # Logging and output.
            final_itr = itr == itrs
            if itr % log_freq == 0 or final_itr:
                if final_itr:
                    result_archive.as_pandas(
                        include_solutions=final_itr).to_csv(
                            outdir / f"{name}_archive.csv")

                # Record and display metrics.
                metrics["QD Score"]["x"].append(itr)
                metrics["QD Score"]["y"].append(result_archive.stats.qd_score)
                metrics["Archive Coverage"]["x"].append(itr)
                metrics["Archive Coverage"]["y"].append(
                    result_archive.stats.coverage)
                print(f"Iteration {itr} | Archive Coverage: "
                      f"{metrics['Archive Coverage']['y'][-1] * 100:.3f}% "
                      f"QD Score: {metrics['QD Score']['y'][-1]:.3f}")

                save_heatmap(archive,
                             str(outdir / f"{name}_heatmap_{itr:05d}_main.png"))

                # Save result_archive
                save_heatmap(
                    result_archive,
                    str(outdir / f"{name}_heatmap_{itr:05d}_result.png"))

    # Plot metrics.
    print(f"Algorithm Time (Excludes Logging and Setup): {non_logging_time}s")
    for metric in metrics:
        plt.plot(metrics[metric]["x"], metrics[metric]["y"])
        plt.title(metric)
        plt.xlabel("Iteration")
        plt.savefig(
            str(outdir / f"{name}_{metric.lower().replace(' ', '_')}.png"))
        plt.clf()
    with (outdir / f"{name}_metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)


if __name__ == '__main__':
    fire.Fire(sphere_main)
