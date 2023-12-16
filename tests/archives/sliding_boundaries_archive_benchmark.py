"""Benchmarks for the SlidingBoundariesArchive."""
from ribs.archives import SlidingBoundariesArchive


def benchmark_add_10k(benchmark, benchmark_data_10k):
    _, solution_batch, objective_batch, measures_batch = benchmark_data_10k

    def setup():
        archive = SlidingBoundariesArchive(solution_dim=solution_batch.shape[1],
                                           dims=[10, 20],
                                           ranges=[(-1, 1), (-2, 2)],
                                           remap_frequency=100,
                                           buffer_capacity=1000)

        # Let numba compile.
        archive.add_single(solution_batch[0], objective_batch[0],
                           measures_batch[0])

        return (archive,), {}

    def add_10k(archive):
        archive.add(solution_batch, objective_batch, measures_batch)

    benchmark.pedantic(add_10k, setup=setup, rounds=5, iterations=1)
