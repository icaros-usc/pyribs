"""Different methods for generating centroids for CVT-MAP Elites

The following 5 methods for centroid generation are implementations
described in the paper: https://dl.acm.org/doi/pdf/10.1145/3583133.3590726.
These methods are to be used and benchmarked in the benchmarks.py file

Usage:
    import centroids.py
"""
import numpy as np
from scipy.stats.qmc import Halton, Sobol, scale


def random_generation(seed, shape):
    """Returns a numpy array of filled with random numbers.

    Args:
        seed: RNG seed
        shape: shape of generated array of centroids

    Returns:
        numpy.array: array of shape, shape, filled with randomly
        generated numbers, ie. centroids.

    """
    rng = np.random.default_rng(seed=seed)
    return rng.random(shape)


def sobol_sequence(seed, shape):
    """Returns an array of a randomly generated sobol sequence.

    Args:
        seed: RNG seed
        shape: shape of generated array of centroids

    Returns:
        numpy.array: array of shape, shape, filled with sobol numbers.

    """
    sampler = Sobol(d=shape[1], scramble=False, seed=seed)
    sample = np.array(sampler.random_base2(shape[0]))
    return sample[0:shape[0]]


def scrambled_sobol_sequence(seed, shape):
    """Returns an array of a randomly generated scrambled sobol sequence.

    Args:
        seed: RNG seed
        shape: shape of generated array of centroids

    Returns:
        numpy.array: array of shape, shape, filled with scrambled
        sobol numbers.

    """

    sampler = Sobol(d=shape[1], scramble=True, seed=seed)
    sample = np.array(sampler.random_base2(shape[0]))
    return sample[0:shape[0]]


def halton_numbers(seed, shape, lower, upper):
    """Returns an array of Halton numbers.

    Args:
        seed: RNG seed
        shape: shape of generated array of centroids

    Returns:
        numpy.array: array of shape, shape, filled with Halton numbers.

    """

    sampler = Halton(d=shape[1], scramble=False, seed=seed)
    sample = sampler.random(n=shape[0])
    return scale(sample, lower, upper)
