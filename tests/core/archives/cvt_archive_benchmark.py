"""Benchmarks for the CVTArhive."""

from ribs.archives import CVTArchive

# pylint: disable = invalid-name, unused-variable


def benchmark_construction_and_k_means(benchmark):

    @benchmark
    def construct():
        CVTArchive([(-1, 1), (-1, 1)], 100, config={"samples": 10_000})
