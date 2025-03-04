import numpy as np
from ribs.archives import GridArchive
from ribs.schedulers import BayesianOptimizationScheduler
from ribs.emitters import BayesianOptimizationEmitter

import os
import time
import wandb
import pickle
import matplotlib.pyplot as plt
from typing import List
from dataclasses import dataclass, asdict
from pathlib import Path
from ribs.visualize import grid_archive_heatmap


class Sphere:
    def __init__(self, solution_dim, shift):
        self.solution_dim = solution_dim
        self.shift = shift

        # We min-max normalize all objective scores to [0, 100]
        self._best_obj = 0
        self._worst_obj = max(
            (10.24 - shift) ** 2 * self.solution_dim,
            (-10.24 - shift) ** 2 * self.solution_dim,
        )

    def evaluate(self, sols):
        if not sols.shape[1] == self.solution_dim:
            raise ValueError(
                f"Expects sols to have shape (,{self.solution_dim}), actually gets shape {sols.shape}"
            )

        displacement = sols - self.shift
        raw_obj = np.sum(np.square(displacement), axis=1)
        objs = (raw_obj - self._worst_obj) / (self._best_obj - self._worst_obj) * 100

        clipped = sols.copy()
        clip_indices = np.where(np.logical_or(clipped > 5.12, clipped < -5.12))
        clipped[clip_indices] = 5.12 / clipped[clip_indices]
        measures = np.concatenate(
            (
                np.sum(clipped[:, : self.solution_dim // 2], axis=1, keepdims=True),
                np.sum(clipped[:, self.solution_dim // 2 :], axis=1, keepdims=True),
            ),
            axis=1,
        )

        return objs, measures


def save_heatmap(archive, heatmap_path):
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=0, vmax=100, cmap="viridis")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close(plt.gcf())


# logdir for saving experiment data
trial_outdir = os.path.join(
    os.path.curdir,
    "test_logs",
)
trial_path = Path(trial_outdir)
if not trial_path.is_dir():
    trial_path.mkdir()


# experiment params
@dataclass
class Params:
    init_nsamples: int
    total_itrs: int
    upscale_schedule: List[List]
    seed: int
    logdir: str
    log_evry: int


params = Params(
    init_nsamples=20,
    total_itrs=1000,
    upscale_schedule=[[5, 5], [10, 10], [25, 25]],
    seed=42,
    logdir=trial_outdir,
    log_evry=20,
)

wandb_logger = wandb.init(
    project="BOP-Elites",
    config=asdict(params),
    name="BOP-Elites, Sphere",
    id=str(params.seed),
    tags=["BOP-Elites", "Sphere"],
    resume="allow",  # Allow resuming from same run ID
)

# Sphere domain evaluator
sphere_domain = Sphere(solution_dim=4, shift=2)

reload_checkpoint = None
if reload_checkpoint is None:
    start_itr = 0
    # The main grid archive that interacts with BOP-Elites. We start at the lowest
    # resolution from ``upscale_schedule``.
    archive = GridArchive(
        solution_dim=sphere_domain.solution_dim,
        dims=params.upscale_schedule[0],
        ranges=[[-5.12, 5.12]] * 2,
        seed=params.seed,
    )
    # The passive archive that does NOT interact with BOP-Elites other than storing
    # all evaluated data seen so far. We do not scale the passive archive, and it
    # always stays at the final resolution.
    passive_archive = GridArchive(
        solution_dim=sphere_domain.solution_dim,
        dims=params.upscale_schedule[-1],
        ranges=[[-5.12, 5.12]] * 2,
        seed=params.seed,
    )

    # An initial batch of data used to warm up GPs
    init_sol = np.random.default_rng(params.seed).normal(
        loc=0,
        scale=0.5,
        size=(params.init_nsamples, sphere_domain.solution_dim),
    )
    init_obj, init_meas = sphere_domain.evaluate(init_sol)

    # The main component of BOP-Elites
    emitters = [
        BayesianOptimizationEmitter(
            archive=archive,
            init_solution=init_sol,
            init_objective=init_obj,
            init_measures=init_meas,
            bounds=[(-10.24, 10.24)] * sphere_domain.solution_dim,
            upscale_schedule=params.upscale_schedule,
            seed=params.seed,
        )
    ]

    # Scheduler for managing multiple emitters (in what order we ask them for
    # solutions etc.).
    # TODO(DrKent): What should the behavior be when there are more than one bayesian
    # emitters? Can they all fill solutions to the same archive (and if so,
    # what happens when one of the emitters converge earlier than the others
    # and requests an archive upscale when the others are not ready?)
    scheduler = BayesianOptimizationScheduler(archive, emitters)

    # Adds the initial batch of data we evaluated to the main and passive archives
    scheduler.archive.add(init_sol, init_obj.flatten(), init_meas)
    passive_archive.add(init_sol, init_obj.flatten(), init_meas)
else:
    with open(reload_checkpoint, "rb") as f:
        data = pickle.load(f)
        start_itr = data["iteration"]
        scheduler = data["scheduler"]
        passive_archive = data["passive_archive"]

exp_start_time = time.time()
for i in range(start_itr, params.total_itrs):
    print(f"----------------------- itr{i} -----------------------")
    itr_start_time = time.time()

    sol = scheduler.ask()
    obj, meas = sphere_domain.evaluate(sol)
    scheduler.tell(obj, meas)
    passive_archive.add(sol, obj, meas)

    metrics_to_fields = {
        "QD score": "qd_score",
        "Coverage": "coverage",
        "Max Obj.": "obj_max",
        "Mean Obj": "obj_mean",
    }

    # Log main archive stats
    wandb.log(
        {
            f"Main Archive/{metric}": getattr(scheduler.archive.stats, field)
            for metric, field in metrics_to_fields.items()
        },
        commit=False,
        step=i,
    )

    # Log passive archive stats
    wandb.log(
        {
            f"Passive Archive/{metric}": getattr(passive_archive.stats, field)
            for metric, field in metrics_to_fields.items()
        },
        commit=False,
        step=i,
    )

    # Log time
    wandb.log(
        {
            "Itr. Time": time.time() - itr_start_time,
        },
        commit=True,
        step=i,
    )

    if i % params.log_evry == 0:
        save_heatmap(
            passive_archive,
            os.path.join(params.logdir, f"heatmap_{i:08d}.png"),
        )

        checkpoint = {
            "iteration": i,
            "scheduler": scheduler,
            "passive_archive": passive_archive,
        }

        with open(os.path.join(params.logdir, f"checkpoint_{i:08d}.pkl"), "wb") as file:
            pickle.dump(checkpoint, file)

wandb_logger.finish()
