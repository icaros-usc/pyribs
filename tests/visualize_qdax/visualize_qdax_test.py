"""Tests for ribs.visualize that use qdax.

Instructions are identical as in visualize_test.py, but images are stored in
tests/visualize_qdax_test/baseline_images/visualize_qdax_test instead.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest
from matplotlib.testing.decorators import image_comparison
from qdax.core.containers.mapelites_repertoire import (MapElitesRepertoire,
                                                       compute_cvt_centroids)

from ribs.visualize import qdax_repertoire_heatmap


@pytest.fixture(autouse=True)
def clean_matplotlib():
    """Cleans up matplotlib figures before and after each test."""
    # Before the test.
    plt.close("all")

    yield

    # After the test.
    plt.close("all")


@image_comparison(baseline_images=["qdax_repertoire_heatmap"],
                  remove_text=False,
                  extensions=["png"])
def test_qdax_repertoire_heatmap():
    plt.figure(figsize=(8, 6))

    # Compute the CVT centroids.
    centroids, _ = compute_cvt_centroids(
        num_descriptors=2,
        num_init_cvt_samples=1000,
        num_centroids=100,
        minval=-1,
        maxval=1,
        random_key=jax.random.PRNGKey(42),
    )

    # Create initial population.
    init_pop_x, init_pop_y = jnp.meshgrid(jnp.linspace(-1, 1, 50),
                                          jnp.linspace(-1, 1, 50))
    init_pop = jnp.stack((init_pop_x.flatten(), init_pop_y.flatten()), axis=1)

    # Create repertoire with the initial population inserted.
    repertoire = MapElitesRepertoire.init(
        genotypes=init_pop,
        # Negative sphere function.
        fitnesses=-jnp.sum(jnp.square(init_pop), axis=1),
        descriptors=init_pop,
        centroids=centroids,
    )

    # Plot heatmap.
    qdax_repertoire_heatmap(repertoire, ranges=[(-1, 1), (-1, 1)])
