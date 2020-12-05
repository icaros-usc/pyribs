"""Benchmarks for the SlidingBoundaryArchive."""

from ribs.archives import SlidingBoundaryArchive

# pylint: disable = invalid-name, unused-variable


def benchmark_add_10k(benchmark, benchmark_data_100k):
    _, solutions, objective_values, behavior_values = benchmark_data_100k
    n = int(1e4)

    def setup():
        archive = SlidingBoundaryArchive([10, 20], [(-1, 1,), (-2, 2)],
                                         remap_frequency=100,
                                         buffer_capacity=1000)
        archive.initialize(solutions.shape[1])

        # Let numba compile.
        archive.add(solutions[0], objective_values[0], behavior_values[0])

        return (archive,), {}

    def add_10k(archive):
        for i in range(n):
            archive.add(solutions[i], objective_values[i], behavior_values[i])

    benchmark.pedantic(add_10k, setup=setup, rounds=5, iterations=1)



def benchmark_get_10k_random_elites(benchmark, benchmark_data_100k):
    _, solutions, objective_values, behavior_values = benchmark_data_100k
    n = int(1e4)
    archive = SlidingBoundaryArchive([10, 20], [(-1, 1,), (-2, 2)],
                                     remap_frequency=100,
                                     buffer_capacity=1000)
    archive.initialize(solutions.shape[1])

    for i in range(n):
        archive.add(solutions[i], objective_values[i], behavior_values[i])

    @benchmark
    def get_elites():
        for i in range(n):
            sol, obj, beh = archive.get_random_elite()
