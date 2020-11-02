"""Benchmarks for the CVTArhive."""

from ribs.archives import CVTArchive

# pylint: disable = invalid-name, unused-variable


def benchmark_construction(benchmark):
    """Construction includes k-means clustering and building a kd-tree."""

    @benchmark
    def construct():
        CVTArchive([(-1, 1), (-1, 1)], 1000, config={"samples": 10_000})


def benchmark_100k_additions(benchmark, benchmark_data_100k):
    n, solutions, objective_values, behavior_values = benchmark_data_100k

    def setup():
        archive = CVTArchive([(-1, 1), (-1, 1)],
                             1000,
                             config={"samples": 10_000})
        return (archive,), {}

    def add_100k(archive):
        for i in range(n):
            archive.add(solutions[i], objective_values[i], behavior_values[i])

    benchmark.pedantic(add_100k, setup=setup, rounds=5, iterations=1)


def benchmark_get_100k_random_elites(benchmark, benchmark_data_100k):
    n, solutions, objective_values, behavior_values = benchmark_data_100k
    archive = CVTArchive([(-1, 1), (-1, 1)], 1000, config={"samples": 10_000})
    for i in range(n):
        archive.add(solutions[i], objective_values[i], behavior_values[i])

    @benchmark
    def get_elites():
        for i in range(n):
            sol, obj, beh = archive.get_random_elite()
