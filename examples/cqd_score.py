"""Example of computing the CQD score on an archive.

Install the following dependencies before running this example:
    pip install fire
"""
import fire
import numpy as np

from ribs.archives import GridArchive, cqd_score
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler


def main(itrs=1000):
    """Runs CMA-ME on a basic function, computing CQD score along the way."""
    # Set up pyribs components.
    archive = GridArchive(
        solution_dim=10,
        dims=[100, 100],
        ranges=[(-1, 1), (-1, 1)],
    )
    emitters = [
        EvolutionStrategyEmitter(
            archive,
            x0=[0.0] * 10,
            sigma0=0.1,
            batch_size=36,
        ) for _ in range(3)
    ]
    scheduler = Scheduler(archive, emitters)

    # Needed for sampling target points.
    rng = np.random.default_rng(42)

    for itr in range(1, itrs + 1):
        solutions = scheduler.ask()
        # Negative sphere function with slight offset.
        objectives = 2.0 - np.sum(np.square(solutions), axis=1)
        measures = solutions[:, :2]
        scheduler.tell(objectives, measures)

        if itr % 100 == 0 or itr == itrs:
            cqd_iterations = 5

            # Here, 200 target points are sampled in measure space within the
            # bounds of the archive. Note that not all archives have lower and
            # upper bounds. For example, ProximityArchive is unstructured, so
            # its lower and upper bounds adjust over time to match the solutions
            # it contains. This differs from GridArchive, where the bounds are
            # set in advance. Thus, for ProximityArchive, it does not make sense
            # to sample target points within its lower and upper bounds.
            # Instead, if using ProximityArchive, there should be a predefined
            # lower and upper bound within which to sample target points. Note
            # that target points can also be generated in other ways, i.e., they
            # do not have to be sampled uniformly within a hyperrectangle as is
            # done here.
            target_points = rng.uniform(
                low=archive.lower_bounds,
                high=archive.upper_bounds,
                size=(cqd_iterations, 200, archive.measure_dim),
            )

            result = cqd_score(
                archive,
                iterations=cqd_iterations,
                target_points=target_points,
                penalties=5,
                obj_min=0.0,
                obj_max=2.0,
                dist_max=np.linalg.norm(archive.upper_bounds -
                                        archive.lower_bounds),
            )

            # The `result` is an instance of CQDScoreResult that contains a
            # number of attributes. The most relevant will be the mean CQD score
            # and the scores computed across the individual iterations.
            print(f"----- Iteration {itr} -----")
            print("CQD score:", result.mean)
            print("Scores on each iteration:", result.scores)


if __name__ == "__main__":
    fire.Fire(main)
