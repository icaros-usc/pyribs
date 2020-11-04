"""Compare performance of adding to the CVTArchive with and without KDTree.

In CVTArchive, we use KDTree to identify the bin by finding the nearest centroid
to a solution in behavior space. Though KDTree is theoretically more efficient
than brute force, constant factors mean that brute force can be faster than
KDTree for smaller numbers of centroids / bins. In this script, we want to
increase the number of bins in the archive and see when KDTree becomes faster
than brute force.

In this experiment, we construct archives with 10, 50, 100, 500, 1k, 5k, 10k,
50k, 100k bins in the behavior space of [(-1, 1), (-1, 1)] and 100k samples. In
each archive, we then time how long it takes to add 100k random solutions
sampled u.a.r. from the behavior space. We run each experiment with brute force
and with the KD tree, 5 times each, and take the minimum runtime (see
https://docs.python.org/3/library/timeit.html#timeit.Timer.repeat).

Usage:
    python cvt_add.py

This script will run for a while (~2 hours) and produce two outputs. The first
is cvt_add_times.json, which holds the raw times. The second is
cvt_add_plot.png, which is a plot of the times with respect to number of bins.

If you wish to re-plot the results without re-running the benchmarks, you can
modify plot_times and then run:

    import cvt_add  # The name of this file.
    data = cvt_add.load_times()
    cvt_add.plot_times(*data)
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
    ax.set_title("Runtime to insert 100k entries into CVTArchive")
    ax.set_xlabel("Archive bins")
    ax.set_ylabel("Time (s)")
    ax.semilogx(n_bins, brute_force_t, "-o", label="Brute Force", c="#304FFE")
    ax.semilogx(n_bins, kd_tree_t, "-o", label="KD-Tree", c="#e62020")
    ax.legend(loc="upper left")
    fig.savefig(filename, bbox_inches="tight", dpi=120)


def main():
    """Creates archives, times insertion into them, and plots results."""
    archive = None
    n_bins = [10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]

    # Pre-made solutions to insert.
    n_vals = 100_000
    solutions = np.random.uniform(-1, 1, (n_vals, 10))
    objective_values = np.random.randn(n_vals)
    behavior_values = np.random.uniform(-1, 1, (n_vals, 2))

    def setup(bins, use_kd_tree):
        nonlocal archive
        archive = CVTArchive([(-1, 1), (-1, 1)],
                             bins,
                             config={
                                 "samples": n_vals,
                                 "use_kd_tree": use_kd_tree,
                             })

    def add_100k_entries():
        nonlocal archive
        for i in range(n_vals):
            archive.add(solutions[i], objective_values[i], behavior_values[i])

    # Run the timing.
    brute_force_t = []
    kd_tree_t = []
    for bins in n_bins:
        for use_kd_tree in (False, True):
            print(f"--------------\nBins: {bins}\nUse KD tree: {use_kd_tree}")
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
