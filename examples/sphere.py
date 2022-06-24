"""Runs various QD algorithms on the Sphere function.

The sphere function in this example is adapted from Section 4 of Fontaine 2020
(https://arxiv.org/abs/1912.02400). Namely, each solution value is clipped to
the range [-5.12, 5.12], and the optimum is moved from [0,..] to [0.4 * 5.12 =
2.048,..]. Furthermore, the objective values are normalized to the range [0,
100] where 100 is the maximum and corresponds to 0 on the original sphere
function.

There are two BCs in this example. The first is the sum of the first n/2 clipped
values of the solution, and the second is the sum of the last n/2 clipped values
of the solution. Having each BC depend equally on several values in the solution
space makes the problem more difficult (refer to Fontaine 2020 for more info).

The supported algorithms are:
- `map_elites`: GridArchive with GaussianEmitter
- `line_map_elites`: GridArchive with IsoLineEmitter
- `cvt_map_elites`: CVTArchive with GaussianEmitter
- `line_cvt_map_elites`: CVTArchive with IsoLineEmitter
- `cma_me_imp`: GridArchive with ImprovementEmitter
- `cma_me_imp_mu`: GridArchive with ImprovementEmitter with mu selection rule
- `cma_me_rd`: GridArchive with RandomDirectionEmitter
- `cma_me_rd_mu`: GridArchive with RandomDirectionEmitter with mu selection rule
- `cma_me_opt`: GridArchive with OptimizingEmitter
- `cma_me_mixed`: GridArchive, and half (7) of the emitter are
  RandomDirectionEmitter and half (8) are ImprovementEmitter

All algorithms use 15 emitters, each with a batch size of 37. Each one runs for
4500 iterations for a total of 15 * 37 * 4500 ~= 2.5M evaluations.

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
                           IsoLineEmitter, OptimizingEmitter)
from ribs.emitters.rankers import get_ranker
from ribs.optimizers import Optimizer
from ribs.visualize import cvt_archive_heatmap, grid_archive_heatmap


def sphere(sol):
    """Sphere function evaluation and BCs for a batch of solutions.

    Args:
        sol (np.ndarray): (batch_size, dim) array of solutions
    Returns:
        objs (np.ndarray): (batch_size,) array of objective values
        bcs (np.ndarray): (batch_size, 2) array of behavior values
    """
    dim = sol.shape[1]

    # Shift the Sphere function so that the optimal value is at x_i = 2.048.
    sphere_shift = 5.12 * 0.4

    # Normalize the objective to the range [0, 100] where 100 is optimal.
    best_obj = 0.0
    worst_obj = (-5.12 - sphere_shift)**2 * dim
    raw_obj = np.sum(np.square(sol - sphere_shift), axis=1)
    objs = (raw_obj - worst_obj) / (best_obj - worst_obj) * 100

    # Calculate BCs.
    clipped = sol.copy()
    clip_indices = np.where(np.logical_or(clipped > 5.12, clipped < -5.12))
    clipped[clip_indices] = 5.12 / clipped[clip_indices]
    bcs = np.concatenate(
        (
            np.sum(clipped[:, :dim // 2], axis=1, keepdims=True),
            np.sum(clipped[:, dim // 2:], axis=1, keepdims=True),
        ),
        axis=1,
    )

    return objs, bcs


def create_optimizer(algorithm, dim, seed):
    """Creates an optimizer based on the algorithm name.

    Args:
        algorithm (str): Name of the algorithm passed into sphere_main.
        dim (int): Dimensionality of the sphere function.
        seed (int): Main seed or the various components.
    Returns:
        Optimizer: A ribs Optimizer for running the algorithm.
    """
    max_bound = dim / 2 * 5.12
    bounds = [(-max_bound, max_bound), (-max_bound, max_bound)]
    initial_sol = np.zeros(dim)
    batch_size = 37
    num_emitters = 15

    # Create archive.
    if algorithm in [
            "map_elites", "line_map_elites", "cma_me_imp", "cma_me_imp_mu",
            "cma_me_rd", "cma_me_rd_mu", "cma_me_opt", "cma_me_opt_mu",
            "cma_me_mixed"
    ]:
        archive = GridArchive(solution_dim=dim,
                              dims=(500, 500),
                              ranges=bounds,
                              seed=seed)
    elif algorithm in ["cvt_map_elites", "line_cvt_map_elites"]:
        archive = CVTArchive(solution_dim=dim,
                             cells=10_000,
                             ranges=bounds,
                             samples=100_000,
                             use_kd_tree=True)
    else:
        raise ValueError(f"Algorithm `{algorithm}` is not recognized")

    # Create emitters. Each emitter needs a different seed, so that they do not
    # all do the same thing.
    emitter_seeds = [None] * num_emitters if seed is None else list(
        range(seed, seed + num_emitters))
    if algorithm in ["map_elites", "cvt_map_elites"]:
        emitters = [
            GaussianEmitter(archive,
                            initial_sol,
                            0.5,
                            batch_size=batch_size,
                            seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["line_map_elites", "line_cvt_map_elites"]:
        emitters = [
            IsoLineEmitter(archive,
                           initial_sol,
                           iso_sigma=0.1,
                           line_sigma=0.2,
                           batch_size=batch_size,
                           seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_me_imp", "cma_me_imp_mu"]:
        if algorithm == "cma_me_imp":
            selection_rule = "filter"
            ranker = get_ranker("2imp")
        else:
            selection_rule = "mu"
            ranker = get_ranker("imp")
        emitters = [
            EvolutionStrategyEmitter(archive,
                                     initial_sol,
                                     0.5,
                                     ranker,
                                     batch_size=batch_size,
                                     selection_rule=selection_rule,
                                     seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_me_rd", "cma_me_rd_mu"]:
        if algorithm == "cma_me_rd":
            selection_rule = "filter"
            ranker = get_ranker("2rd")
        else:
            selection_rule = "mu"
            ranker = get_ranker("rd")
        emitters = [
            EvolutionStrategyEmitter(archive,
                                     initial_sol,
                                     0.5,
                                     ranker,
                                     batch_size=batch_size,
                                     selection_rule=selection_rule,
                                     seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_me_opt", "cma_me_opt_mu"]:
        if algorithm == "cma_me_opt":
            selection_rule = "filter"
            ranker = get_ranker("2obj")
        else:
            selection_rule = "mu"
            ranker = get_ranker("obj")
            print(ranker)
        emitters = [
            EvolutionStrategyEmitter(archive,
                                     initial_sol,
                                     0.5,
                                     ranker,
                                     batch_size=batch_size,
                                     seed=s) for s in emitter_seeds
            # OptimizingEmitter(archive,
            #                   initial_sol,
            #                   0.5,
            #                   batch_size=batch_size,
            #                   seed=s) for s in emitter_seeds
        ]
    elif algorithm == "cma_me_mixed":
        emitters = [
            EvolutionStrategyEmitter(archive,
                                     initial_sol,
                                     0.5,
                                     get_ranker("2rd"),
                                     batch_size=batch_size,
                                     seed=s) for s in emitter_seeds[:7]
        ] + [
            EvolutionStrategyEmitter(archive,
                                     initial_sol,
                                     0.5,
                                     get_ranker("2imp"),
                                     batch_size=batch_size,
                                     seed=s) for s in emitter_seeds[7:]
        ]

    return Optimizer(archive, emitters)


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
                dim=20,
                itrs=4500,
                outdir="sphere_output",
                log_freq=250,
                seed=None):
    """Demo on the Sphere function.

    Args:
        algorithm (str): Name of the algorithm.
        dim (int): Dimensionality of solutions.
        itrs (int): Iterations to run.
        outdir (str): Directory to save output.
        log_freq (int): Number of iterations to wait before recording metrics
            and saving heatmap.
        seed (int): Seed for the algorithm. By default, there is no seed.
    """
    name = f"{algorithm}_{dim}"
    outdir = Path(outdir)
    if not outdir.is_dir():
        outdir.mkdir()

    optimizer = create_optimizer(algorithm, dim, seed)
    archive = optimizer.archive
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
        save_heatmap(archive, str(outdir / f"{name}_heatmap_{0:05d}.png"))

        for itr in range(1, itrs + 1):
            itr_start = time.time()
            sols = optimizer.ask()
            objs, bcs = sphere(sols)
            optimizer.tell(objs, bcs)
            non_logging_time += time.time() - itr_start
            progress()

            # Logging and output.
            final_itr = itr == itrs
            if itr % log_freq == 0 or final_itr:
                if final_itr:
                    archive.as_pandas(include_solutions=final_itr).to_csv(
                        outdir / f"{name}_archive.csv")

                # Record and display metrics.
                metrics["QD Score"]["x"].append(itr)
                metrics["QD Score"]["y"].append(archive.stats.qd_score)
                metrics["Archive Coverage"]["x"].append(itr)
                metrics["Archive Coverage"]["y"].append(archive.stats.coverage)
                print(f"Iteration {itr} | Archive Coverage: "
                      f"{metrics['Archive Coverage']['y'][-1] * 100:.3f}% "
                      f"QD Score: {metrics['QD Score']['y'][-1]:.3f}")

                save_heatmap(archive,
                             str(outdir / f"{name}_heatmap_{itr:05d}.png"))

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
