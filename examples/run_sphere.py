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

All algorithms use 15 emitters, each with a batch size of 37. Each one runs for
4500 iterations for a total of 15 * 37 * 4500 ~= 2.5M evaluations. Outputs are
saved in the directory `run_sphere_output` by default. The archive is saved as a
CSV named `{algorithm}_{dim}_archive.csv`, while snapshots of the heatmap are
saved as `{algorithm}_{dim}_heatmap_{iteration}.png`.

Usage (see run_sphere function for all args):
    python run_sphere.py ALGORITHM DIM
Example:
    python run_sphere.py map_elites 20

    # To make numpy and sklearn run single-threaded, set env variables for BLAS
    # and OpenMP:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_sphere.py map_elites 20
Help:
    python run_sphere.py --help
"""
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from alive_progress import alive_bar

from ribs.archives import CVTArchive, GridArchive
from ribs.emitters import (GaussianEmitter, ImprovementEmitter, IsoLineEmitter,
                           OptimizingEmitter, RandomDirectionEmitter)
from ribs.optimizers import Optimizer
from ribs.visualize import cvt_archive_heatmap


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
        algorithm (str): Name of the algorithm passed into run_sphere.
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
            "cma_me_rd", "cma_me_rd_mu", "cma_me_opt"
    ]:
        archive = GridArchive((500, 500), bounds, seed=seed)
    elif algorithm in ["cvt_map_elites", "line_cvt_map_elites"]:
        archive = CVTArchive(bounds, 10_000, samples=100_000, use_kd_tree=True)
    else:
        raise ValueError(f"Algorithm `{algorithm}` is not recognized")

    # Create emitters. Each emitter needs a different seed, so that they do not
    # all do the same thing.
    emitter_seeds = [None] * num_emitters if seed is None else list(
        range(seed, seed + num_emitters))
    if algorithm in ["map_elites", "cvt_map_elites"]:
        emitters = [
            GaussianEmitter(initial_sol,
                            0.5,
                            archive,
                            batch_size=batch_size,
                            seed=s) for s in emitter_seeds
        ]
    elif algorithm == ["line_map_elites", "line_cvt_map_elites"]:
        emitters = [
            IsoLineEmitter(initial_sol,
                           archive,
                           iso_sigma=0.1,
                           line_sigma=0.2,
                           batch_size=batch_size,
                           seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_me_imp", "cma_me_imp_mu"]:
        selection_rule = "filter" if algorithm == "cma_me_imp" else "mu"
        emitters = [
            ImprovementEmitter(initial_sol,
                               0.5,
                               archive,
                               batch_size=batch_size,
                               selection_rule=selection_rule,
                               seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_me_rd", "cma_me_rd_mu"]:
        selection_rule = "filter" if algorithm == "cma_me_rd" else "mu"
        emitters = [
            RandomDirectionEmitter(initial_sol,
                                   0.5,
                                   archive,
                                   batch_size=batch_size,
                                   selection_rule=selection_rule,
                                   seed=s) for s in emitter_seeds
        ]
    elif algorithm == "cma_me_opt":
        emitters = [
            OptimizingEmitter(initial_sol,
                              0.5,
                              archive,
                              batch_size=batch_size,
                              seed=s) for s in emitter_seeds
        ]

    return Optimizer(archive, emitters)


def run_sphere(algorithm,
               dim=20,
               itrs=4500,
               outdir="run_sphere_output",
               log_freq=250,
               seed=None):
    """Demo on the Sphere function.

    Args:
        algorithm (str): Name of the algorithm.
        dim (int): Dimensionality of solutions.
        itrs (int): Iterations to run.
        outdir (str): Directory to save output.
        log_freq (int): Number of iterations to wait before printing metrics and
            saving heatmap.
        seed (int): Seed for the algorithm. By default, there is no seed.
    """
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

    with alive_bar(itrs) as progress:
        for itr in range(1, itrs + 1):
            sols = optimizer.ask()
            objs, bcs = sphere(sols)
            optimizer.tell(objs, bcs)
            progress()

            # Logging and output.
            final_itr = itr == itrs
            if itr % log_freq == 0 or final_itr:
                name = f"{algorithm}_{dim}"
                data = archive.as_pandas(include_solutions=final_itr)
                if final_itr:
                    data.to_csv(str(outdir / f"{name}_archive.csv"))

                # Record and display metrics.
                total_cells = 10_000 if isinstance(archive,
                                                   CVTArchive) else 500 * 500
                metrics["QD Score"]["x"].append(itr)
                metrics["QD Score"]["y"].append(data['objective'].sum())
                metrics["Archive Coverage"]["x"].append(itr)
                metrics["Archive Coverage"]["y"].append(
                    len(data) / total_cells * 100)
                print(f"Iteration {itr} | Archive Coverage: "
                      f"{metrics['Archive Coverage']['y'][-1]:.3f}% "
                      f"QD Score: {metrics['QD Score']['y'][-1]:.3f}")

                # Generate heatmap.
                heatmap_path = str(outdir / f"{name}_heatmap_{itr:05d}.png")
                if isinstance(archive, GridArchive):
                    # TODO: Replace _1 and _2 when as_pandas() is fixed (see
                    # https://github.com/icaros-usc/pyribs/issues/44)
                    heatmap_data = np.full(archive.dims, np.nan)
                    for row in data.itertuples():
                        # pylint: disable = protected-access
                        heatmap_data[row._1, row._2] = row.objective
                    sns.heatmap(heatmap_data, cmap="magma", vmin=0, vmax=100)
                    plt.savefig(heatmap_path)
                elif isinstance(archive, CVTArchive):
                    plt.figure(figsize=(16, 12))
                    cvt_archive_heatmap(archive, vmin=0, vmax=100)
                    plt.savefig(heatmap_path)
                plt.clf()

    # TODO: save metrics JSON
    # Plot metrics.
    for metric in metrics:
        plt.plot(metrics[metric]["x"], metrics[metric]["y"])
        plt.title(metric)
        plt.xlabel("Iteration")
        plt.savefig(
            str(outdir / f"{name}_{metric.lower().replace(' ', '_')}.png"))
        plt.clf()


if __name__ == '__main__':
    fire.Fire(run_sphere)
