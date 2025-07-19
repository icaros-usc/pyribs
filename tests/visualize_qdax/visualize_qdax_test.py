"""Tests for ribs.visualize that use qdax.

Instructions are identical as in visualize_test.py, but images are stored in
tests/visualize_qdax_test/baseline_images/visualize_qdax_test instead.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest
from matplotlib.testing.decorators import image_comparison

from ribs.visualize import qdax_repertoire_3d_plot, qdax_repertoire_heatmap

try:
    from qdax.core.containers.mapelites_repertoire import (
        MapElitesRepertoire,
        compute_cvt_centroids,
    )
except ImportError:
    pytest.skip(allow_module_level=True)


@pytest.fixture(autouse=True)
def clean_matplotlib():
    """Cleans up matplotlib figures before and after each test."""
    # Before the test.
    plt.close("all")

    yield

    # After the test.
    plt.close("all")


@image_comparison(
    baseline_images=["qdax_repertoire_heatmap"],
    remove_text=False,
    extensions=["png"],
    tol=0.1,  # See CVT_IMAGE_TOLERANCE in cvt_archive_heatmap_test.py
)
def test_qdax_repertoire_heatmap():
    plt.figure(figsize=(8, 6))

    # Compute the CVT centroids.
    centroids = compute_cvt_centroids(
        num_descriptors=2,
        num_init_cvt_samples=1000,
        num_centroids=100,
        minval=-1,
        maxval=1,
        key=jax.random.PRNGKey(42),
    )

    # Create initial population.
    init_pop_x, init_pop_y = jnp.meshgrid(
        jnp.linspace(-1, 1, 50), jnp.linspace(-1, 1, 50)
    )
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


@image_comparison(
    baseline_images=["qdax_repertoire_3d_plot"],
    remove_text=False,
    extensions=["png"],
    tol=0.1,  # See CVT_IMAGE_TOLERANCE in cvt_archive_3d_plot_test.py
)
def test_qdax_repertoire_3d_plot():
    plt.figure(figsize=(8, 6))

    key = jax.random.PRNGKey(42)

    # Compute the CVT centroids.
    key, subkey = jax.random.split(key)
    centroids = compute_cvt_centroids(
        num_descriptors=3,
        num_init_cvt_samples=1000,
        num_centroids=500,
        minval=-1,
        maxval=1,
        key=subkey,
    )

    # Create initial population.
    key, *subkeys = jax.random.split(key, 4)
    x = jax.random.uniform(subkeys[0], (10000,), minval=-1.0, maxval=1.0)
    y = jax.random.uniform(subkeys[1], (10000,), minval=-1.0, maxval=1.0)
    z = jax.random.uniform(subkeys[2], (10000,), minval=-1.0, maxval=1.0)
    init_pop = jnp.stack((x, y, z), axis=1)

    # Create repertoire with the initial population inserted.
    repertoire = MapElitesRepertoire.init(
        genotypes=init_pop,
        # Negative sphere function.
        fitnesses=-jnp.sum(jnp.square(init_pop), axis=1),
        descriptors=init_pop,
        centroids=centroids,
    )

    # Plot heatmap.
    qdax_repertoire_3d_plot(repertoire, ranges=[(-1, 1), (-1, 1), (-1, 1)])
