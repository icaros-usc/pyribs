"""Compare performance of adding to the CVTArchive with and without k-D tree.

In CVTArchive, we use a k-D tree to identify the cell by finding the nearest
centroid to a solution in measure space. Though a k-D tree is theoretically
more efficient than brute force, constant factors mean that brute force can be
faster than k-D tree for smaller numbers of centroids / cells. In this script,
we increase the number of cells in the archive and see when the k-D tree becomes
faster than brute force.

In this experiment, we construct archives with 10, 50, 100, 500, 1k cells in
the measure space of [(-1, 1), (-1, 1)] and 100k samples.  In each archive, we
then time how long it takes to add 1k batches of 100 random solutions sampled
uniformly at random. from the measure space. We run each experiment with brute
force and with the k-D tree, 5 times each, and take the minimum runtime (see
https://docs.python.org/3/library/timeit.html#timeit.Timer.repeat).

Usage:
    python cvt_add.py

This script will run for a few minutes and produce two outputs. The first is
cvt_add_times.json, which holds the raw times. The second is cvt_add_plot.png,
which is a plot of the times with respect to number of cells.

To re-plot the results without re-running the benchmarks, modify plot_times and
run:

    import cvt_add  # The name of this file.
    cvt_add.plot_times(*cvt_add.load_times())
"""
import json
import timeit
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from ribs.archives import CVTArchive


def save_times(n_cells,
               brute_force_t,
               kd_tree_t,
               filename="cvt_add_times.json"):
    """Saves a dict of the results to the given file."""
    with open(filename, "w") as file:
        json.dump(
            {
                "n_cells": n_cells,
                "brute_force_t": brute_force_t,
                "kd_tree_t": kd_tree_t,
            }, file)


def load_times(filename="cvt_add_times.json"):
    """Loads the results from the given file."""
    with open(filename, "r") as file:
        data = json.load(file)
        return data["n_cells"], data["brute_force_t"], data["kd_tree_t"]


def plot_times(n_cells, brute_force_t, kd_tree_t, filename="cvt_add_plot.png"):
    """Plots the results to the given file."""
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.tight_layout()
    ax.set_title(
        "Runtime to insert 1k batches of 100 2D entries\ninto CVTArchive")
    ax.set_xlabel("log(number of cells in archive)")
    ax.set_ylabel("log(time) (s)")
    ax.set_yscale("log")
    ax.semilogx(n_cells, brute_force_t, "-o", label="Brute Force", c="#304FFE")
    ax.semilogx(n_cells, kd_tree_t, "-x", label="k-D Tree", c="#e62020")
    ax.grid(True, which="major", linestyle="--", linewidth=1)
    ax.legend(loc="upper left")
    fig.savefig(filename, bbox_inches="tight", dpi=120)


def main():
    """Creates archives, times insertion into them, and plots results."""
    archive = None
    n_cells = [10, 50, 100, 500, 1_000]

    # Pre-made solutions to insert.
    n_batches = 1_000
    batch_size = 100
    all_solution_batch = np.random.uniform(-1, 1, (n_batches, batch_size, 10))
    all_objective_batch = np.random.standard_normal((n_batches, batch_size))
    all_measures_batch = np.random.uniform(-1, 1, (n_batches, batch_size, 2))

    # Set up these archives so we can use the same centroids across all
    # experiments for a certain number of cells (and also save time).
    ref_archives = {
        cells: CVTArchive(
            solution_dim=all_solution_batch.shape[2],
            cells=cells,
            ranges=[(-1, 1), (-1, 1)],
            # Use 200k cells to avoid dropping clusters. However, note that we
            # no longer test with 10k cells.
            samples=100_000 if cells != 10_000 else 200_000,
            use_kd_tree=False) for cells in n_cells
    }

    def setup(cells, use_kd_tree):
        nonlocal archive
        archive = CVTArchive(solution_dim=all_solution_batch.shape[2],
                             cells=cells,
                             ranges=[(-1, 1), (-1, 1)],
                             custom_centroids=ref_archives[cells].centroids,
                             use_kd_tree=use_kd_tree)

    def add_entries():
        nonlocal archive
        for i in range(n_batches):
            archive.add(
                all_solution_batch[i],
                all_objective_batch[i],
                all_measures_batch[i],
            )

    # Run the timing.
    brute_force_t = []
    kd_tree_t = []
    for cells in n_cells:
        for use_kd_tree in (False, True):
            print(f"--------------\n"
                  f"Cells: {cells}\n"
                  f"Method: {'k-D Tree' if use_kd_tree else 'Brute Force'}")
            setup_func = partial(setup, cells, use_kd_tree)
            res_t = min(
                timeit.repeat(add_entries, setup_func, repeat=5, number=1))
            print(f"Time: {res_t} s")

            if use_kd_tree:
                kd_tree_t.append(res_t)
            else:
                brute_force_t.append(res_t)
        save_times(n_cells, brute_force_t, kd_tree_t, "cvt_add_times.json")

    # Save the results.
    plot_times(n_cells, brute_force_t, kd_tree_t)
    save_times(n_cells, brute_force_t, kd_tree_t)


if __name__ == "__main__":
    main()
