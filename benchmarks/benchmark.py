'''
numpy module used for vectorization
scipy for distance calculation
ribs for generating centroids to be benchmarked
'''
import numpy as np
from scipy.spatial import distance

from ribs.archives import CVTArchive


def get_score(centroids, num_samples, seed):
    '''
    gets the score for all centroids documented as described in
    the paper, fast generation of centroids for map elites

    final score equation is different but same results are achieved
    '''
    num_centroids = centroids.shape[0]
    centroid_dim = centroids.shape[1]

    rng = np.random.default_rng(seed=seed)
    random_samples = rng.random(size=(num_samples, centroid_dim))

    num_closest_pts = np.zeros(num_centroids)

    closest_idx = distance.cdist(random_samples, centroids).argmin(axis=1)
    assert len(closest_idx) == num_samples
    for idx in closest_idx:
        num_closest_pts[idx] += 1

    centroid_vol = num_closest_pts / num_samples

    score = np.sum(np.abs(centroid_vol - 1 / num_centroids))

    return score


def main():
    '''
    standard main function
    '''
    score_seed = 1
    num_samples = 10000
    archive = CVTArchive(
        solution_dim=20,
        cells=512,
        ranges=[(0., 1.), (0., 1.)],
    )
    cvt_centroids = archive.centroids
    print(
        get_score(centroids=cvt_centroids,
                  num_samples=num_samples,
                  seed=score_seed))

    centroid_gen_seed = 100
    num_centroids = 1024
    dim = 2
    rng = np.random.default_rng(seed=centroid_gen_seed)
    random_centroids = rng.random((num_centroids, dim))
    print(
        get_score(centroids=random_centroids,
                  num_samples=num_samples,
                  seed=score_seed))


if __name__ == "__main__":
    main()
