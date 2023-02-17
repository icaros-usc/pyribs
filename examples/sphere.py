"""Runs various QD algorithms on the Sphere function.

Install the following dependencies before running this example:
    pip install ribs[visualize] tqdm fire

The sphere function in this example is adapted from Section 4 of Fontaine 2020
(https://arxiv.org/abs/1912.02400). Namely, each solution value is clipped to
the range [-5.12, 5.12], and the optimum is moved from [0,..] to [0.4 * 5.12 =
2.048,..]. Furthermore, the objectives are normalized to the range [0,
100] where 100 is the maximum and corresponds to 0 on the original sphere
function.

There are two measures in this example. The first is the sum of the first n/2
clipped values of the solution, and the second is the sum of the last n/2
clipped values of the solution. Having each measure depend equally on several
values in the solution space makes the problem more difficult (refer to
Fontaine 2020 for more info).

The supported algorithms are:
- `map_elites`: GridArchive with GaussianEmitter.
- `line_map_elites`: GridArchive with IsoLineEmitter.
- `cvt_map_elites`: CVTArchive with GaussianEmitter.
- `line_cvt_map_elites`: CVTArchive with IsoLineEmitter.
- `me_map_elites`: MAP-Elites with Bandit Scheduler.
- `cma_me_imp`: GridArchive with EvolutionStrategyEmitter using
  TwoStageImprovmentRanker.
- `cma_me_imp_mu`: GridArchive with EvolutionStrategyEmitter using
  TwoStageImprovmentRanker and mu selection rule.
- `cma_me_rd`: GridArchive with EvolutionStrategyEmitter using
  RandomDirectionRanker.
- `cma_me_rd_mu`: GridArchive with EvolutionStrategyEmitter using
  TwoStageRandomDirectionRanker and mu selection rule.
- `cma_me_opt`: GridArchive with EvolutionStrategyEmitter using ObjectiveRanker
  with mu selection rule.
- `cma_me_mixed`: GridArchive with EvolutionStrategyEmitter, where half (7) of
  the emitter are using TwoStageRandomDirectionRanker and half (8) are
  TwoStageImprovementRanker.
- `cma_mega`: GridArchive with GradientArborescenceEmitter.
- `cma_mega_adam`: GridArchive with GradientArborescenceEmitter using Adam
  Optimizer.
- `cma_mae`: GridArchive (learning_rate = 0.01) with EvolutionStrategyEmitter
  using ImprovementRanker.
- `cma_maega`: GridArchive (learning_rate = 0.01) with
  GradientArborescenceEmitter using ImprovementRanker.

All algorithms use 15 emitters, each with a batch size of 37. Each one runs for
4500 iterations for a total of 15 * 37 * 4500 ~= 2.5M evaluations.

Notes:
- `cma_mega` and `cma_mega_adam` use only one emitter and run for 10,000
  iterations. This is to be consistent with the paper (`Fontaine 2021
  <https://arxiv.org/abs/2106.03894>`_) in which these algorithms were proposed.
- `cma_mae` and `cma_maega` run for 10,000 iterations as well.
- CVTArchive in this example uses 10,000 cells, as opposed to the 250,000
  (500x500) in the GridArchive, so it is not fair to directly compare
  `cvt_map_elites` and `line_cvt_map_elites` to the other algorithms.

Outputs are saved in the `sphere_output/` directory by default. The archive is
saved as a CSV named `{algorithm}_{dim}_archive.csv`, while snapshots of the
heatmap are saved as `{algorithm}_{dim}_heatmap_{iteration}.png`. Metrics about
the run are also saved in `{algorithm}_{dim}_metrics.json`, and plots of the
metrics are saved in PNG's with the name `{algorithm}_{dim}_metric_name.png`.

To generate a video of the heatmap from the heatmap images, use a tool like
ffmpeg. For example, the following will generate a 6FPS video showing the
heatmap for cma_me_imp with 20 dims.

    ffmpeg -r 6 -i "sphere_output/cma_me_imp_20_heatmap_%*.png" \
        sphere_output/cma_me_imp_20_heatmap_video.mp4

Usage (see sphere_main function for all args or run `python sphere.py --help`):
    python sphere.py ALGORITHM
Example:
    python sphere.py map_elites

    # To make numpy and sklearn run single-threaded, set env variables for BLAS
    # and OpenMP:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sphere.py map_elites 20
Help:
    python sphere.py --help
"""
import copy
import json
import time
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from ribs.archives import CVTArchive, GridArchive
from ribs.emitters import (EvolutionStrategyEmitter, GaussianEmitter,
                           GradientArborescenceEmitter, IsoLineEmitter)
from ribs.schedulers import BanditScheduler, Scheduler
from ribs.visualize import cvt_archive_heatmap, grid_archive_heatmap

CONFIG = {
    "map_elites": {
        "dim": 20,
        "iters": 4500,
        "archive_dims": (500, 500),
        "use_result_archive": False,
        "is_dqd": False,
        "batch_size": 37,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": GaussianEmitter,
            "kwargs": {
                "sigma": 0.5
            },
            "num_emitters": 15
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "line_map_elites": {
        "dim": 20,
        "iters": 4500,
        "archive_dims": (500, 500),
        "use_result_archive": False,
        "is_dqd": False,
        "batch_size": 37,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": IsoLineEmitter,
            "kwargs": {
                "iso_sigma": 0.1,
                "line_sigma": 0.2
            },
            "num_emitters": 15
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "cvt_map_elites": {
        "dim": 20,
        "iters": 4500,
        "archive_dims": (500, 500),
        "use_result_archive": False,
        "is_dqd": False,
        "batch_size": 37,
        "archive": {
            "class": CVTArchive,
            "kwargs": {
                "cells": 10_000,
                "samples": 100_000,
                "use_kd_tree": True
            }
        },
        "emitters": [{
            "class": GaussianEmitter,
            "kwargs": {
                "sigma": 0.5
            },
            "num_emitters": 15
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "line_cvt_map_elites": {
        "dim": 20,
        "iters": 4500,
        "archive_dims": (500, 500),
        "use_result_archive": False,
        "is_dqd": False,
        "batch_size": 37,
        "archive": {
            "class": CVTArchive,
            "kwargs": {
                "cells": 10_000,
                "samples": 100_000,
                "use_kd_tree": True
            }
        },
        "emitters": [{
            "class": IsoLineEmitter,
            "kwargs": {
                "iso_sigma": 0.1,
                "line_sigma": 0.2
            },
            "num_emitters": 15
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "me_map_elites": {
        "dim": 100,
        "iters": 20_000,
        "archive_dims": (100, 100),
        "use_result_archive": False,
        "is_dqd": False,
        "batch_size": 50,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.5,
                "ranker": "obj"
            },
            "num_emitters": 12
        }, {
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.5,
                "ranker": "2rd"
            },
            "num_emitters": 12
        }, {
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.5,
                "ranker": "2imp"
            },
            "num_emitters": 12
        }, {
            "class": IsoLineEmitter,
            "kwargs": {
                "iso_sigma": 0.01,
                "line_sigma": 0.1
            },
            "num_emitters": 12
        }],
        "scheduler": {
            "class": BanditScheduler,
            "kwargs": {
                "num_active": 12,
                "reselect": "terminated"
            }
        }
    },
    "cma_me_mixed": {
        "dim": 20,
        "iters": 4500,
        "archive_dims": (500, 500),
        "use_result_archive": False,
        "is_dqd": False,
        "batch_size": 37,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.5,
                "ranker": "2rd"
            },
            "num_emitters": 7
        }, {
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.5,
                "ranker": "2imp"
            },
            "num_emitters": 8
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "cma_me_imp": {
        "dim": 20,
        "iters": 4500,
        "archive_dims": (500, 500),
        "use_result_archive": False,
        "is_dqd": False,
        "batch_size": 37,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.5,
                "ranker": "2imp",
                "selection_rule": "filter",
                "restart_rule": "no_improvement"
            },
            "num_emitters": 15
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "cma_me_imp_mu": {
        "dim": 20,
        "iters": 4500,
        "archive_dims": (500, 500),
        "use_result_archive": False,
        "is_dqd": False,
        "batch_size": 37,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.5,
                "ranker": "2imp",
                "selection_rule": "mu",
                "restart_rule": "no_improvement"
            },
            "num_emitters": 15
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "cma_me_rd": {
        "dim": 20,
        "iters": 4500,
        "archive_dims": (500, 500),
        "use_result_archive": False,
        "is_dqd": False,
        "batch_size": 37,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.5,
                "ranker": "2rd",
                "selection_rule": "filter",
                "restart_rule": "no_improvement"
            },
            "num_emitters": 15
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "cma_me_rd_mu": {
        "dim": 20,
        "iters": 4500,
        "archive_dims": (500, 500),
        "use_result_archive": False,
        "is_dqd": False,
        "batch_size": 37,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.5,
                "ranker": "2rd",
                "selection_rule": "mu",
                "restart_rule": "no_improvement"
            },
            "num_emitters": 15
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "cma_me_opt": {
        "dim": 20,
        "iters": 4500,
        "archive_dims": (500, 500),
        "use_result_archive": False,
        "is_dqd": False,
        "batch_size": 37,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.5,
                "ranker": "obj",
                "selection_rule": "mu",
                "restart_rule": "basic"
            },
            "num_emitters": 15
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "cma_mega": {
        "dim": 1_000,
        "iters": 10_000,
        "archive_dims": (100, 100),
        "use_result_archive": False,
        "is_dqd": True,
        "batch_size": 36,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": GradientArborescenceEmitter,
            "kwargs": {
                "sigma0": 10.0,
                "lr": 1.0,
                "grad_opt": "gradient_ascent",
                "selection_rule": "mu"
            },
            "num_emitters": 1
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "cma_mega_adam": {
        "dim": 1_000,
        "iters": 10_000,
        "archive_dims": (100, 100),
        "use_result_archive": False,
        "is_dqd": True,
        "batch_size": 36,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": -np.inf
            }
        },
        "emitters": [{
            "class": GradientArborescenceEmitter,
            "kwargs": {
                "sigma0": 10.0,
                "lr": 0.002,
                "grad_opt": "adam",
                "selection_rule": "mu"
            },
            "num_emitters": 1
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "cma_mae": {
        "dim": 100,
        "iters": 10_000,
        "archive_dims": (100, 100),
        "use_result_archive": True,
        "is_dqd": False,
        "batch_size": 37,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": 0,
                "learning_rate": 0.01
            }
        },
        "emitters": [{
            "class": EvolutionStrategyEmitter,
            "kwargs": {
                "sigma0": 0.5,
                "ranker": "imp",
                "selection_rule": "mu",
                "restart_rule": "basic"
            },
            "num_emitters": 15
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    },
    "cma_maega": {
        "dim": 1_000,
        "iters": 10_000,
        "archive_dims": (100, 100),
        "use_result_archive": True,
        "is_dqd": True,
        "batch_size": 37,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "threshold_min": 0,
                "learning_rate": 0.01
            }
        },
        "emitters": [{
            "class": GradientArborescenceEmitter,
            "kwargs": {
                "sigma0": 10.0,
                "lr": 1.0,
                "ranker": "imp",
                "grad_opt": "gradient_ascent",
                "restart_rule": "basic"
            },
            "num_emitters": 15
        }],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {}
        }
    }
}


def sphere(solution_batch):
    """Sphere function evaluation and measures for a batch of solutions.

    Args:
        solution_batch (np.ndarray): (batch_size, dim) batch of solutions.
    Returns:
        objective_batch (np.ndarray): (batch_size,) batch of objectives.
        objective_grad_batch (np.ndarray): (batch_size, solution_dim) batch of
            objective gradients.
        measures_batch (np.ndarray): (batch_size, 2) batch of measures.
        measures_grad_batch (np.ndarray): (batch_size, 2, solution_dim) batch of
            measure gradients.
    """
    dim = solution_batch.shape[1]

    # Shift the Sphere function so that the optimal value is at x_i = 2.048.
    sphere_shift = 5.12 * 0.4

    # Normalize the objective to the range [0, 100] where 100 is optimal.
    best_obj = 0.0
    worst_obj = (-5.12 - sphere_shift)**2 * dim
    raw_obj = np.sum(np.square(solution_batch - sphere_shift), axis=1)
    objective_batch = (raw_obj - worst_obj) / (best_obj - worst_obj) * 100

    # Compute gradient of the objective.
    objective_grad_batch = -2 * (solution_batch - sphere_shift)

    # Calculate measures.
    clipped = solution_batch.copy()
    clip_mask = (clipped < -5.12) | (clipped > 5.12)
    clipped[clip_mask] = 5.12 / clipped[clip_mask]
    measures_batch = np.concatenate(
        (
            np.sum(clipped[:, :dim // 2], axis=1, keepdims=True),
            np.sum(clipped[:, dim // 2:], axis=1, keepdims=True),
        ),
        axis=1,
    )

    # Compute gradient of the measures.
    derivatives = np.ones(solution_batch.shape)
    derivatives[clip_mask] = -5.12 / np.square(solution_batch[clip_mask])

    mask_0 = np.concatenate((np.ones(dim // 2), np.zeros(dim - dim // 2)))
    mask_1 = np.concatenate((np.zeros(dim // 2), np.ones(dim - dim // 2)))

    d_measure0 = derivatives * mask_0
    d_measure1 = derivatives * mask_1

    measures_grad_batch = np.stack((d_measure0, d_measure1), axis=1)

    return (
        objective_batch,
        objective_grad_batch,
        measures_batch,
        measures_grad_batch,
    )


def create_scheduler(config, algorithm, seed=None):
    """Creates a scheduler based on the algorithm.

    Args:
        config (dict): Configuration dictionary with parameters for the various
            components.
        algorithm (string): Name of the algorithm
        seed (int): Main seed or the various components.
    Returns:
        ribs.schedulers.Scheduler: A ribs scheduler for running the algorithm.
    """
    solution_dim = config["dim"]
    archive_dims = config["archive_dims"]
    learning_rate = 1.0 if "learning_rate" not in config["archive"][
        "kwargs"] else config["archive"]["kwargs"]["learning_rate"]
    use_result_archive = config["use_result_archive"]
    max_bound = solution_dim / 2 * 5.12
    bounds = [(-max_bound, max_bound), (-max_bound, max_bound)]
    initial_sol = np.zeros(solution_dim)
    mode = "batch"

    # Create archive.
    archive_class = config["archive"]["class"]
    if archive_class == GridArchive:
        archive = archive_class(solution_dim=solution_dim,
                                ranges=bounds,
                                dims=archive_dims,
                                seed=seed,
                                **config["archive"]["kwargs"])
    else:
        archive = archive_class(solution_dim=solution_dim,
                                ranges=bounds,
                                **config["archive"]["kwargs"])

    # Create result archive.
    result_archive = None
    if use_result_archive:
        result_archive = GridArchive(solution_dim=solution_dim,
                                     dims=archive_dims,
                                     ranges=bounds,
                                     seed=seed)

    # Create emitters. Each emitter needs a different seed so that they do not
    # all do the same thing, hence we create an rng here to generate seeds. The
    # rng may be seeded with None or with a user-provided seed.
    rng = np.random.default_rng(seed)
    emitters = []
    for e in config["emitters"]:
        emitter_class = e["class"]
        emitters += [
            emitter_class(archive,
                          x0=initial_sol,
                          **e["kwargs"],
                          batch_size=config["batch_size"],
                          seed=s)
            for s in rng.integers(0, 1_000_000, e["num_emitters"])
        ]

    # Create Scheduler
    scheduler_class = config["scheduler"]["class"]
    scheduler = scheduler_class(archive,
                                emitters,
                                result_archive=result_archive,
                                add_mode=mode,
                                **config["scheduler"]["kwargs"])
    scheduler_name = scheduler.__class__.__name__

    print(f"Create {scheduler_name} for {algorithm} with learning rate "
          f"{learning_rate} and add mode {mode}, using solution dim "
          f"{solution_dim}, archive dims {archive_dims}, and "
          f"{len(emitters)} emitters.")
    return scheduler


def save_heatmap(archive, heatmap_path):
    """Saves a heatmap of the archive to the given path.

    Args:
        archive (GridArchive or CVTArchive): The archive to save.
        heatmap_path: Image path for the heatmap.
    """
    if isinstance(archive, GridArchive):
        plt.figure(figsize=(8, 6))
        grid_archive_heatmap(archive, vmin=0, vmax=100)
        plt.tight_layout()
        plt.savefig(heatmap_path)
    elif isinstance(archive, CVTArchive):
        plt.figure(figsize=(16, 12))
        cvt_archive_heatmap(archive, vmin=0, vmax=100)
        plt.tight_layout()
        plt.savefig(heatmap_path)
    plt.close(plt.gcf())


def sphere_main(algorithm,
                dim=None,
                itrs=None,
                archive_dims=None,
                learning_rate=None,
                outdir="sphere_output",
                log_freq=250,
                seed=None):
    """Demo on the Sphere function.

    Args:
        algorithm (str): Name of the algorithm.
        dim (int): Dimensionality of the sphere function.
        itrs (int): Iterations to run.
        archive_dims (tuple): Dimensionality of the archive.
        learning_rate (float): The archive learning rate.
        outdir (str): Directory to save output.
        log_freq (int): Number of iterations to wait before recording metrics
            and saving heatmap.
        seed (int): Seed for the algorithm. By default, there is no seed.
    """
    config = copy.deepcopy(CONFIG[algorithm])

    # Use default dim for each algorithm.
    if dim is not None:
        config["dim"] = dim

    # Use default itrs for each algorithm.
    if itrs is not None:
        config["iters"] = itrs

    # Use default archive_dim for each algorithm.
    if archive_dims is not None:
        config["archive_dims"] = archive_dims

    # Use default learning_rate for each algorithm.
    if learning_rate is not None:
        config["archive"]["kwargs"]["learning_rate"] = learning_rate

    name = f"{algorithm}_{config['dim']}"
    outdir = Path(outdir)
    if not outdir.is_dir():
        outdir.mkdir()

    scheduler = create_scheduler(config, algorithm, seed=seed)
    result_archive = scheduler.result_archive
    is_dqd = config["is_dqd"]
    itrs = config["iters"]
    metrics = {
        "QD Score": {
            "x": [0],
            "y": [0.0],
        },
        "Archive Coverage": {
            "x": [0],
            "y": [0.0],
        },
    }

    non_logging_time = 0.0
    save_heatmap(result_archive, str(outdir / f"{name}_heatmap_{0:05d}.png"))

    for itr in tqdm.trange(1, itrs + 1):
        itr_start = time.time()

        if is_dqd:
            solution_batch = scheduler.ask_dqd()
            (objective_batch, objective_grad_batch, measures_batch,
             measures_grad_batch) = sphere(solution_batch)
            objective_grad_batch = np.expand_dims(objective_grad_batch, axis=1)
            jacobian_batch = np.concatenate(
                (objective_grad_batch, measures_grad_batch), axis=1)
            scheduler.tell_dqd(objective_batch, measures_batch, jacobian_batch)

        solution_batch = scheduler.ask()
        objective_batch, _, measure_batch, _ = sphere(solution_batch)
        scheduler.tell(objective_batch, measure_batch)
        non_logging_time += time.time() - itr_start

        # Logging and output.
        final_itr = itr == itrs
        if itr % log_freq == 0 or final_itr:
            if final_itr:
                result_archive.as_pandas(include_solutions=final_itr).to_csv(
                    outdir / f"{name}_archive.csv")

            # Record and display metrics.
            metrics["QD Score"]["x"].append(itr)
            metrics["QD Score"]["y"].append(result_archive.stats.qd_score)
            metrics["Archive Coverage"]["x"].append(itr)
            metrics["Archive Coverage"]["y"].append(
                result_archive.stats.coverage)
            tqdm.tqdm.write(
                f"Iteration {itr} | Archive Coverage: "
                f"{metrics['Archive Coverage']['y'][-1] * 100:.3f}% "
                f"QD Score: {metrics['QD Score']['y'][-1]:.3f}")

            save_heatmap(result_archive,
                         str(outdir / f"{name}_heatmap_{itr:05d}.png"))

    # Plot metrics.
    print(f"Algorithm Time (Excludes Logging and Setup): {non_logging_time}s")
    for metric in metrics:
        plt.plot(metrics[metric]["x"], metrics[metric]["y"])
        plt.title(metric)
        plt.xlabel("Iteration")
        plt.savefig(
            str(outdir / f"{name}_{metric.lower().replace(' ', '_')}.png"))
        plt.clf()
    with (outdir / f"{name}_metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)


if __name__ == '__main__':
    fire.Fire(sphere_main)
