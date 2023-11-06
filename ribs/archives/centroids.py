'''importing stuff'''
import numpy as np
from scipy.stats.qmc import Halton, Sobol, scale


def random_generation(seed, shape):
    '''generating random floats between 0 and 1 with a specified shape'''
    rng = np.random.default_rng(seed=seed)
    return rng.random(shape)


def sobol_sequence(seed, shape):
    '''generating sobol numbers with a specified shape
    using base2, we first sample 2^shape[0] results
    take only the first shape[0] results
    shape[0] < 2^shape[0]'''
    sampler = Sobol(d=shape[1], scramble=False, seed=seed)
    sample = np.array(sampler.random_base2(shape[0]))
    return sample[0:shape[0]]


def scrambled_sobol_sequence(seed, shape):
    '''same logic as normal sobol sequence but scrambled'''
    sampler = Sobol(d=shape[1], scramble=True, seed=seed)
    sample = np.array(sampler.random_base2(shape[0]))
    return sample[0:shape[0]]


def halton_numbers(seed, shape, lower, upper):
    '''drawing halton numbers using scipy'''
    sampler = Halton(d=shape[1], scramble=False, seed=seed)
    sample = sampler.random(n=shape[0])
    return scale(sample, lower, upper)
