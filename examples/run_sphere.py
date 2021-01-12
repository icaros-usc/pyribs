"""Runs various QD algorithms on the Sphere function.

The sphere function in this example is adopted from Section 4 of Fontaine 2020
(https://arxiv.org/abs/1912.02400). Namely, each solution value is clipped to
the range [-5.12, 5.12], and the optimum is moved from [0,..] to [0.4 * 5.12 =
2.048,..]. Furthermore, the objective values are normalized to the range [0,100]
where 100 is the maximum and corresponds to 0 on the original sphere function.

There are two BCs in this example. The first is the sum of the first n/2 clipped
values of the solution, and the second is the sum of the last n/2 clipped values
of the solution. Having each BC depend equally on several values in the solution
space makes the problem more difficult (refer to Fontaine 2020 for more info).

The supported algorithms are:
- `map_elites`: GridArchive with GaussianEmitter
- `line_map_elites`: GridArchive with IsoLineEmitter
- `cvt_map_elites`: CVTArchive with GaussianEmitter
- `line_cvt_map_elites`: CVTArchive with IsoLineEmitter

All algorithms are run for 100,000 iterations with a batch size of 25 in the
emitters. Outputs are saved in a directory, `run_sphere_output` by default. The
archive is saved as a CSV named `{algorithm}_{dim}_archive.csv`, while the
heatmap is saved as a PNG named `{algorithm}_{dim}_heatmap.png`.

Usage:
    # Where ALGORITHM is chosen from above, DIM is the dimensionality of the
    # Sphere function, ITRS is the number of iterations to run, and OUTDIR is
    # the directory to save outputs. By default, DIMS is 20, ITRS is 100,000,
    # and OUTDIR is `run_sphere_output`.
    python run_sphere.py ALGORITHM DIM ITRS OUTDIR
Example:
    python run_sphere.py map_elites 20
Help:
    python run_sphere.py --help
"""
import time
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ribs.archives import CVTArchive, GridArchive
from ribs.emitters import GaussianEmitter, IsoLineEmitter
from ribs.optimizers import Optimizer
from ribs.visualize import cvt_archive_heatmap


def sphere(sol):
    """Sphere function evaluation and BCs for a single solution.

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


def save_and_display_outputs(archive, algorithm, dim, outdir):
    """Creates an output directory and saves the archive and its heatmap.

    Args:
        archive: The archive to save.
        algorithm (str): Name of the algorithm.
        dim (int): Dimensionality of solutions.
        outdir (str): Directory to save output.
    """
    name = f"{algorithm}_{dim}"

    # Create directory.
    outdir = Path(outdir)
    if not outdir.is_dir():
        outdir.mkdir()

    # Save archive as CSV.
    data = archive.as_pandas()
    data.to_csv(str(outdir / f"{name}_archive.csv"))

    # Display some outputs.
    print("===== Sample Outputs =====")
    print(data.head())

    # Generate heatmaps.
    heatmap_path = str(outdir / f"{name}_heatmap.png")
    if algorithm in ["map_elites", "line_map_elites"]:
        heatmap_data = data.pivot('index-0', 'index-1', 'objective')
        sns.heatmap(heatmap_data, cmap="magma")
        plt.savefig(heatmap_path)
    elif algorithm in ["cvt_map_elites", "line_cvt_map_elites"]:
        plt.figure(figsize=(16, 12))
        cvt_archive_heatmap(archive)
        plt.savefig(heatmap_path)


def run_sphere(algorithm, dim=20, itrs=100_000, outdir="run_sphere_output"):
    """Demo on the Sphere function.

    Args:
        algorithm (str): Name of the algorithm.
        dim (int): Dimensionality of solutions.
        itrs (int): Iterations to run.
        outdir (str): Directory to save output.
    """
    max_bound = dim * 5.12
    bounds = [(-max_bound, max_bound), (-max_bound, max_bound)]
    initial_sol = np.zeros(dim)

    # Different components depending on the algorithm chosen.
    if algorithm == "map_elites":
        archive = GridArchive((500, 500), bounds)
        emitters = [GaussianEmitter(initial_sol, 0.5, archive, batch_size=25)]
        opt = Optimizer(archive, emitters)
    elif algorithm == "line_map_elites":
        archive = GridArchive((500, 500), bounds)
        emitters = [
            IsoLineEmitter(initial_sol,
                           archive,
                           iso_sigma=0.1,
                           line_sigma=0.2,
                           batch_size=25)
        ]
        opt = Optimizer(archive, emitters)
    elif algorithm == "cvt_map_elites":
        archive = CVTArchive(bounds, 10_000, samples=100_000, use_kd_tree=True)
        emitters = [GaussianEmitter(initial_sol, 0.5, archive, batch_size=25)]
        opt = Optimizer(archive, emitters)
    elif algorithm == "line_cvt_map_elites":
        archive = CVTArchive(bounds, 10_000, samples=100_000, use_kd_tree=True)
        emitters = [
            IsoLineEmitter(initial_sol,
                           archive,
                           iso_sigma=0.1,
                           line_sigma=0.2,
                           batch_size=25)
        ]
        opt = Optimizer(archive, emitters)
    else:
        raise ValueError(f"Algorithm `{algorithm}` is not recognized")

    # Run the algorithm.
    start_time = time.time()
    for i in range(itrs):
        sols = opt.ask()
        objs, bcs = sphere(sols)

        opt.tell(objs, bcs)

        if (i + 1) % 1000 == 0:
            print(f"Finished {i + 1} itrs after {time.time() - start_time} s")

    save_and_display_outputs(archive, algorithm, dim, outdir)


if __name__ == '__main__':
    fire.Fire(run_sphere)
