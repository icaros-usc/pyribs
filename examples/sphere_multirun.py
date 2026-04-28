r"""Runs multiple trials of algorithms in sphere.py and computes statistics.

Usage:
    # To run all algorithms at once. Use OPENBLAS_NUM_THREADS and OMP_NUM_THREADS to
    # make sure each run only takes up a single thread. Otherwise, they would use many
    # threads and create a lot of contention.
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sphere_multirun.py \
        --algos=ALL --trials=20 --max-workers=40

    # To run just a single algorithm.
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sphere_multirun.py \
        --algos=cma_mae --trials=20 --max-workers=40

    # To run two algorithms for 5000 itrs.
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sphere_multirun.py \
        --algos=cma_mae,dms --trials=20 --max-workers=40 --itrs=5000
"""

import collections
import concurrent.futures
from datetime import datetime
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import sphere
import tqdm
from loguru import logger as log


def aggregate_results(results: dict[list], outdir: Path) -> None:
    """Aggregates and saves the results from the runs."""
    log.info("Recording results")

    df = pd.DataFrame(results).sort_values("Algorithm")
    df.to_csv(outdir / "results.csv")

    metrics = df[["Algorithm", "QD Score", "Coverage"]].groupby("Algorithm")
    metrics_mean = metrics.mean()
    metrics_std = metrics.std()

    count = df[["Algorithm", "Trial"]].groupby("Algorithm").count()

    summary_df = pd.DataFrame(index=metrics_mean.index)
    summary_df["QD Score"] = [
        f"{mean:,.2f} ± {std:,.2f}"
        for mean, std in zip(
            metrics_mean["QD Score"], metrics_std["QD Score"], strict=True
        )
    ]
    summary_df["Coverage"] = [
        f"{mean * 100:,.2f} ± {std:,.2%}"
        for mean, std in zip(
            metrics_mean["Coverage"], metrics_std["Coverage"], strict=True
        )
    ]
    summary_df["Trials"] = count["Trial"]

    summary_df.to_csv(outdir / "results_summary.csv")
    summary_df.to_markdown(outdir / "results_summary.md", stralign="right")


def main(
    algos: str | list[str],
    trials: int,
    itrs: int = 10000,
    outdir: str | None = None,
    seed: int | None = None,
    max_workers: int | None = None,
) -> None:
    """Runs multiple trials of algorithms in sphere.py and computes statistics.

    Args:
        algos: Algorithms to evaluate. On the command line, this can be passed as a
            single algorithm name, e.g., "cma_mae". It can also be a comma-separated
            list, e.g., "cma_mae,dms". Finally, it can be "ALL", to indicate running all
            algorithms in sphere.py.
        trials: Number of trials to run each algorithm.
        itrs: Iterations to run the algorithm.
        outdir: Directory to save output. If not provided, it will be automatically set
            to `logs/sphere_multirun/YYYY-MM-DD_HH-MM-SS_seed-{seed}`.
        seed: Base seed for the trials. By default, there is no seed.
        max_workers: Maximum number of workers when using ProcessPoolExecutor to run the
            trials.
    """
    # Initialize output directory for the overall run.
    outdir = (
        (
            Path("logs")
            / Path(__file__).stem
            / datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S_seed-{seed}")
        )
        if outdir is None
        else Path(outdir)
    )
    outdir.mkdir(parents=True, exist_ok=False)

    log.remove()
    log.add(lambda msg: tqdm.tqdm.write(msg, end=""), colorize=True)
    log.add(outdir / "out.log")  # Save logs in outdir.
    log.info("Saving overall outputs to: {}", outdir)

    rng = np.random.default_rng(seed)
    results = collections.defaultdict(list)

    if isinstance(algos, str):
        if algos == "ALL":  # Run all available algos.
            algo_list = list(sphere.CONFIG)
        else:
            algo_list = [algos]
    else:
        algo_list = algos

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs simultaneously.
        future_to_info = {}
        for algo in algo_list:
            seeds = rng.choice(10_000, size=trials)
            for trial, trial_seed in enumerate(seeds):
                f = executor.submit(
                    sphere.sphere_main,
                    algorithm=algo,
                    outdir=outdir / algo / f"{trial:02d}_seed-{trial_seed}",
                    seed=trial_seed,
                    itrs=itrs,
                    verbose=False,
                    save_archive=False,
                )
                future_to_info[f] = (algo, trial, trial_seed)

        # Collect all results.
        for i, f in enumerate(
            concurrent.futures.as_completed(future_to_info.keys()), start=1
        ):
            algo, trial, trial_seed = future_to_info[f]
            try:
                res = f.result()
                results["Algorithm"].append(algo)
                results["Trial"].append(trial)
                results["Seed"].append(trial_seed)
                for metric, val in res.items():
                    results[metric].append(val)
                log.info(
                    "{}/{} (SUCCESS): {} trial {}, seed {}",
                    i,
                    len(future_to_info),
                    algo,
                    trial,
                    trial_seed,
                )
            except Exception as e:  # pylint: disable = broad-exception-caught
                # Any uncaught exception is costly because it kills the whole run.
                log.info(
                    "{}/{} (FAILURE): {} trial {}, seed {}\n{}",
                    i,
                    len(future_to_info),
                    algo,
                    trial,
                    trial_seed,
                    e,
                )

    aggregate_results(results, outdir)
    log.info("Done")


if __name__ == "__main__":
    fire.Fire(main)
