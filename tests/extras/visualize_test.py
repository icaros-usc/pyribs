"""Tests for ribs.visualize."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.testing.decorators import image_comparison

from ribs.archives import CVTArchive
from ribs.visualize import cvt_archive_heatmap


def _sphere(sol):
    """The sphere function is the sum of squared components of the solution.

    We return the negative because MAP-Elites seeks to maximize.
    """
    return -np.sum(np.square(sol))


@image_comparison(baseline_images=["cvt_archive_heatmap"],
                  remove_text=False,
                  extensions=["png"])
def test_cvt_archive_heatmap():
    seed = 42  # For repeatability.
    np.random.seed(seed)  # Make scipy's k-means also deterministic.
    rng = np.random.default_rng(seed)

    archive = CVTArchive(
        [(-1, 1), (-1, 1)],
        100,
        config={
            "seed": seed,
            "samples": 10_000,
            "use_kd_tree": False,
            "k_means_threshold": 1e-6,
        })

    # Add solutions.
    n_vals = 100_000
    for _ in range(n_vals):
        solution = rng.uniform(-1, 1, 2)
        archive.add(
            solution=solution,
            objective_value=_sphere(solution),
            behavior_values=solution,
        )

    # Plot.
    _, ax = plt.subplots(figsize=(8, 6))
    cvt_archive_heatmap(archive, ax=ax)
