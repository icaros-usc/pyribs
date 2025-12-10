"""Benchmarks for the CVTArhive."""

from ribs.archives import CVTArchive


def benchmark_init(nearest_neighbors, benchmark):
    """Construction includes k-means clustering and building a kd-tree."""

    def init():
        CVTArchive(
            solution_dim=2,
            centroids=1000,
            ranges=[(-1, 1), (-1, 1)],
            samples=20_000,
            nearest_neighbors=nearest_neighbors,
        )

    benchmark(init)


def benchmark_add_10k(nearest_neighbors, benchmark, benchmark_data_10k):
    _, solution_batch, objective_batch, measures_batch = benchmark_data_10k

    def setup():
        archive = CVTArchive(
            solution_dim=solution_batch.shape[1],
            centroids=1000,
            ranges=[(-1, 1), (-1, 1)],
            samples=20_000,
            nearest_neighbors=nearest_neighbors,
        )

        # Let numba compile.
        archive.add_single(solution_batch[0], objective_batch[0], measures_batch[0])

        return (archive,), {}

    def add_10k(archive):
        archive.add(solution_batch, objective_batch, measures_batch)

    benchmark.pedantic(add_10k, setup=setup, rounds=5, iterations=1)
