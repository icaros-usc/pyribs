"""Compare performance of adding to the CVTArchive with and without k-D tree.

In CVTArchive, we use a k-D tree to identify the bin by finding the nearest
centroid to a solution in behavior space. Though a k-D tree is theoretically
more efficient than brute force, constant factors mean that brute force can be
faster than k-D tree for smaller numbers of centroids / bins. In this script, we
want to increase the number of bins in the archive and see when the k-D tree
becomes faster than brute force.

In this experiment, we construct archives with 10, 50, 100, 500, 1k, 5k, 10k,
100k bins in the behavior space of [(-1, 1), (-1, 1)] and 100k samples (except
for 10k bins, where we use 200k samples so that the CVT generation does not drop
a cluster). In each archive, we then time how long it takes to add 100k random
solutions sampled u.a.r. from the behavior space. We run each experiment with
brute force and with the k-D tree, 5 times each, and take the minimum runtime
(see https://docs.python.org/3/library/timeit.html#timeit.Timer.repeat).

Usage:
    python cvt_add.py

This script will run for a while (~30 min) and produce two outputs. The first is
cvt_add_times.json, which holds the raw times. The second is cvt_add_plot.png,
which is a plot of the times with respect to number of bins.

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


def save_times(n_bins, brute_force_t, kd_tree_t, filename="cvt_add_times.json"):
    """Saves a dict of the results to the given file."""
    with open(filename, "w") as file:
        json.dump(
            {
                "n_bins": n_bins,
                "brute_force_t": brute_force_t,
                "kd_tree_t": kd_tree_t,
            }, file)


def load_times(filename="cvt_add_times.json"):
    """Loads the results from the given file."""
    with open(filename, "r") as file:
        data = json.load(file)
        return data["n_bins"], data["brute_force_t"], data["kd_tree_t"]


def plot_times(n_bins, brute_force_t, kd_tree_t, filename="cvt_add_plot.png"):
    """Plots the results to the given file."""
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.tight_layout()
    ax.set_title("Runtime to insert 100k 2D entries into CVTArchive")
    ax.set_xlabel("Archive bins")
    ax.set_ylabel("Time (s)")
    ax.set_yscale("log")
    ax.semilogx(n_bins, brute_force_t, "-o", label="Brute Force", c="#304FFE")
    ax.semilogx(n_bins, kd_tree_t, "-o", label="k-D Tree", c="#e62020")
    ax.grid(True, which="major", linestyle="--", linewidth=1)
    ax.legend(loc="upper left")
    fig.savefig(filename, bbox_inches="tight", dpi=120)


def main():
    """Creates archives, times insertion into them, and plots results."""
    archive = None
    n_bins = [10, 50, 100, 500, 1_000, 5_000, 10_000, 100_000]

    # Pre-made solutions to insert.
    n_vals = 100_000
    solutions = np.random.uniform(-1, 1, (n_vals, 10))
    objective_values = np.random.randn(n_vals)
    behavior_values = np.random.uniform(-1, 1, (n_vals, 2))

    # Set up these archives so we can use the same centroids across all
    # experiments for a certain number of bins (and also save time).
    ref_archives = {
        bins: CVTArchive(
            bins,
            [(-1, 1), (-1, 1)],
            # Use 200k bins to avoid dropping clusters.
            samples=n_vals if bins != 10_000 else 200_000,
            use_kd_tree=False) for bins in n_bins
    }
    for bins, archive in ref_archives.items():
        print(f"Setting up archive with {bins} bins")
        archive.initialize(solutions.shape[1])

    def setup(bins, use_kd_tree):
        nonlocal archive
        archive = CVTArchive(bins, [(-1, 1), (-1, 1)],
                             custom_centroids=ref_archives[bins].centroids,
                             use_kd_tree=use_kd_tree)
        archive.initialize(solutions.shape[1])

    def add_100k_entries():
        nonlocal archive
        for i in range(n_vals):
            archive.add(solutions[i], objective_values[i], behavior_values[i])

    # Run the timing.
    brute_force_t = []
    kd_tree_t = []
    for bins in n_bins:
        for use_kd_tree in (False, True):
            print(f"--------------\n"
                  f"Bins: {bins}\n"
                  f"Method: {'k-D Tree' if use_kd_tree else 'Brute Force'}")
            setup_func = partial(setup, bins, use_kd_tree)
            res_t = min(
                timeit.repeat(add_100k_entries, setup_func, repeat=5, number=1))
            print(f"Time: {res_t} s")

            if use_kd_tree:
                kd_tree_t.append(res_t)
            else:
                brute_force_t.append(res_t)
        save_times(n_bins, brute_force_t, kd_tree_t, "cvt_add_times.json")

    # Save the results.
    plot_times(n_bins, brute_force_t, kd_tree_t)
    save_times(n_bins, brute_force_t, kd_tree_t)


if __name__ == "__main__":
    main()
