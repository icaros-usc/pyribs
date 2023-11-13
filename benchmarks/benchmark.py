"""Quantifies the performance of different centroid generation techniques

To measure how well a generation technique, i.e., random centroids, CVT, etc,
performs, we measure the probability of generating a random point within a
certain region defined by the centroid of that region.

The equations for this benchmark can be found in Mouret 2023:
https://dl.acm.org/doi/pdf/10.1145/3583133.3590726.

Usage:
    python benchmarks.py

This script will generate centroids using 2 techniques, CVT and random
generation. These centroids will then be evaluated by the get_score()
function which will output a probability score between [0, 1].
"""

import numpy as np
from scipy.spatial import distance

from ribs.archives import CVTArchive


def get_score(centroids, num_samples, seed):
    """Returns the performance of generated centroids

    Args:
        centroids (numpy.ndarray): centroids being evaluated
        num_samples (int): number of random points generated
        seed (int): RNG seed

    Returns:
        float: probability a sampled point hits a region

    """

    num_centroids = centroids.shape[0]
    centroid_dim = centroids.shape[1]

    rng = np.random.default_rng(seed=seed)
    random_samples = rng.random(size=(num_samples, centroid_dim))

    num_closest_pts = np.zeros(num_centroids)

    closest_idx = distance.cdist(random_samples, centroids).argmin(axis=1)

    for idx in closest_idx:
        num_closest_pts[idx] += 1
    # Note: The method in the paper detailed the additional division of
    # centroid_vol by num_samples. We did not include that here, however
    # results remain similar to the paper's.

    centroid_vol = num_closest_pts / num_samples

    score = np.sum(np.abs(centroid_vol - 1 / num_centroids))

    return score


def main():
    """main() function that benchmarks 6 different centroid generation
    techniques used in the aforementioned paper.
    """

    score_seed = 1
    num_samples = 10000
    archive = CVTArchive(
        solution_dim=20,
        cells=512,
        ranges=[(0., 1.), (0., 1.)],
    )
    cvt_centroids = archive.centroids
    print(
        "Score for CVT generation: ",
        get_score(centroids=cvt_centroids,
                  num_samples=num_samples,
                  seed=score_seed))

    centroid_gen_seed = 100
    num_centroids = 1024
    dim = 2
    rng = np.random.default_rng(seed=centroid_gen_seed)
    random_centroids = rng.random((num_centroids, dim))
    print(
        "Score for random generation: ",
        get_score(centroids=random_centroids,
                  num_samples=num_samples,
                  seed=score_seed))


if __name__ == "__main__":
    main()
