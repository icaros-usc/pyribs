"""Runs various QD algorithms on the sphere linear projection benchmark.

Install the following dependencies before running this example:
    pip install ribs[visualize] tqdm fire

The sphere function in this example is adapted from Section 4 of Fontaine 2020
(https://arxiv.org/abs/1912.02400). Namely, each solution value is clipped to the range
[-5.12, 5.12], and the optimum is moved from [0,..] to [0.4 * 5.12 = 2.048,..].
Furthermore, the objectives are normalized to the range [0, 100] where 100 is the
maximum and corresponds to 0 on the original sphere function.

There are two measures in this example. The first is the sum of the first n/2 clipped
values of the solution, and the second is the sum of the last n/2 clipped values of the
solution. Having each measure depend equally on several values in the solution space
makes the problem more difficult (refer to Fontaine 2020 for more info).

We support a number of algorithms in this script. The parameters for each algorithm are
stored in CONFIG. The parameters roughly reproduce the results from the CMA-MAE paper
(Fontaine 2023, https://arxiv.org/abs/2205.10752), i.e., they use the following
settings:
- Archives have 10,000 cells, either as a 100x100 grid archive or a 10,000-cell CVT
  archive.
- Each algorithm generates 540 solutions every iteration, typically as one emitter
  generating 540 solutions or 15 emitters generating 36 solutions each.
- We default to run each algorithm for 10,000 iterations.
- We default to run on the 100-dimensional version of the sphere problem.
Below we list the algorithms available.

MAP-Elites and MAP-Elites (line):
- `map_elites`: GridArchive with GaussianEmitter.
- `line_map_elites`: GridArchive with IsoLineEmitter.
- `cvt_map_elites`: CVTArchive with GaussianEmitter.
- `line_cvt_map_elites`: CVTArchive with IsoLineEmitter.

Multi-Emitter MAP-Elites:
- `me_map_elites`: MAP-Elites with Bandit Scheduler.

CMA-ME:
- `cma_me_imp`: GridArchive with EvolutionStrategyEmitter using
  TwoStageImprovmentRanker; this is the suggested version of CMA-ME.
- `cma_me_imp_mu`: GridArchive with EvolutionStrategyEmitter using
  TwoStageImprovmentRanker and mu selection rule.
- `cma_me_basic`: GridArchive with EvolutionStrategyEmitter using
  TwoStageImprovmentRanker, mu selection rule, and basic restart rule. This is the
  version of CMA-ME that was used as a baseline in Fontaine 2023.
- `cma_me_rd`: GridArchive with EvolutionStrategyEmitter using RandomDirectionRanker.
- `cma_me_rd_mu`: GridArchive with EvolutionStrategyEmitter using
  TwoStageRandomDirectionRanker and mu selection rule.
- `cma_me_opt`: GridArchive with EvolutionStrategyEmitter using ObjectiveRanker with mu
  selection rule.
- `cma_me_mixed`: GridArchive with EvolutionStrategyEmitter, where half (7) of the
  emitters use TwoStageRandomDirectionRanker and half (8) use TwoStageImprovementRanker.

DQD algorithms:
- `og_map_elites`: GridArchive with GradientOperatorEmitter; does not use measure
  gradients.
- `omg_mega`: GridArchive with GradientOperatorEmitter; uses measure gradients.
- `cma_mega`: GridArchive with GradientArborescenceEmitter.
- `cma_mega_adam`: GridArchive with GradientArborescenceEmitter using Adam Optimizer.

CMA-MAE and CMA-MAEGA:
- `cma_mae`: GridArchive (learning_rate = 0.01) with EvolutionStrategyEmitter using
  ImprovementRanker.
- `cma_maega`: GridArchive (learning_rate = 0.01) with GradientArborescenceEmitter using
  ImprovementRanker.

Novelty Search:
- `ns_cma`: Novelty Search with CMA-ES; implemented using a ProximityArchive with
  EvolutionStrategyEmitter. Results are stored in a passive GridArchive. Note that the
  objective will not be optimized in this case.
- `nslc_cma_imp`: EvolutionStrategyEmitter with a ProximityArchive with local
  competition turned on. Thus, the archive returns two-stage improvement information
  that is fed to the EvolutionStrategyEmitter just like in CMA-ME.

DDS:
- `dds`: Density Descent Search (Lee 2024; https://arxiv.org/abs/2312.11331) with a KDE
  as the density estimator. Uses DensityArchive and EvolutionStrategyEmitter with
  DensityRanker.
- `dds_kde_sklearn`: Density Descent Search using scikit-learn's KernelDensity as the
  density estimator.

Outputs are saved in the `sphere_output/` directory by default. The archive is saved as
a CSV named `{algorithm}_{dim}_archive.csv`, while snapshots of the heatmap are saved as
`{algorithm}_{dim}_heatmap_{iteration}.png`. Metrics about the run are also saved in
`{algorithm}_{dim}_metrics.json`, and plots of the metrics are saved in PNG's with the
name `{algorithm}_{dim}_metric_name.png`.

To generate a video of the heatmap from the heatmap images, use a tool like ffmpeg. For
example, the following will generate a 6 FPS (Frames Per Second) video showing the
heatmap for cma_me_imp with 100 dims.

    ffmpeg -r 6 -i "sphere_output/cma_me_imp_100_heatmap_%*.png" \
        sphere_output/cma_me_imp_100_heatmap_video.mp4

Usage (see sphere_main function for all args or run `python sphere.py --help`):
    python sphere.py ALGORITHM
Example:
    python sphere.py map_elites

    # To make numpy and sklearn run single-threaded, set env variables for BLAS
    # and OpenMP:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sphere.py map_elites 100
Help:
    python sphere.py --help
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from ribs.archives import (
    ArchiveBase,
    CVTArchive,
    DensityArchive,
    GridArchive,
    ProximityArchive,
)
from ribs.emitters import (
    EvolutionStrategyEmitter,
    GaussianEmitter,
    GradientArborescenceEmitter,
    GradientOperatorEmitter,
    IsoLineEmitter,
)
from ribs.schedulers import BanditScheduler, Scheduler
from ribs.visualize import cvt_archive_heatmap, grid_archive_heatmap

CONFIG = {
    ## MAP-Elites and MAP-Elites (line) ##
    "map_elites": {
        "is_dqd": False,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "dims": (100, 100),
            },
        },
        "result_archive": None,
        "emitters": [
            {
                "class": GaussianEmitter,
                "kwargs": {
                    "sigma": 0.5,
                    "batch_size": 540,
                },
                "num_emitters": 1,
            }
        ],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {},
        },
    },
    "line_map_elites": {
        "is_dqd": False,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "dims": (100, 100),
            },
        },
        "result_archive": None,
        "emitters": [
            {
                "class": IsoLineEmitter,
                "kwargs": {
                    "iso_sigma": 0.5,
                    "line_sigma": 0.2,
                    "batch_size": 540,
                },
                "num_emitters": 1,
            }
        ],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {},
        },
    },
    "cvt_map_elites": {
        "is_dqd": False,
        "archive": {
            "class": CVTArchive,
            "kwargs": {
                "cells": 10000,
                "samples": 100000,
                "nearest_neighbors": "scipy_kd_tree",
            },
        },
        "result_archive": None,
        "emitters": [
            {
                "class": GaussianEmitter,
                "kwargs": {
                    "sigma": 0.5,
                    "batch_size": 540,
                },
                "num_emitters": 1,
            }
        ],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {},
        },
    },
    "line_cvt_map_elites": {
        "is_dqd": False,
        "archive": {
            "class": CVTArchive,
            "kwargs": {
                "cells": 10000,
                "samples": 100000,
                "nearest_neighbors": "scipy_kd_tree",
            },
        },
        "result_archive": None,
        "emitters": [
            {
                "class": IsoLineEmitter,
                "kwargs": {
                    "iso_sigma": 0.5,
                    "line_sigma": 0.2,
                    "batch_size": 540,
                },
                "num_emitters": 1,
            }
        ],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {},
        },
    },
    ## Multi-Emitter MAP-Elites (ME-MAP-Elites) ##
    "me_map_elites": {
        "is_dqd": False,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "dims": (100, 100),
            },
        },
        "result_archive": None,
        "emitters": [
            {
                "class": EvolutionStrategyEmitter,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "obj",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            },
            {
                "class": EvolutionStrategyEmitter,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "2rd",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            },
            {
                "class": EvolutionStrategyEmitter,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "2imp",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            },
            {
                "class": IsoLineEmitter,
                "kwargs": {
                    "iso_sigma": 0.01,
                    "line_sigma": 0.1,
                    "batch_size": 36,
                },
                "num_emitters": 15,
            },
        ],
        "scheduler": {
            "class": BanditScheduler,
            "kwargs": {
                "num_active": 15,
                "reselect": "terminated",
            },
        },
    },
    ## CMA-ME ##
    "cma_me_imp": {
        "is_dqd": False,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "dims": (100, 100),
            },
        },
        "result_archive": None,
        "emitters": [
            {
                "class": EvolutionStrategyEmitter,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "2imp",
                    "selection_rule": "filter",
                    "restart_rule": "no_improvement",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {},
        },
    },
    "cma_me_imp_mu": {
        "is_dqd": False,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "dims": (100, 100),
            },
        },
        "result_archive": None,
        "emitters": [
            {
                "class": EvolutionStrategyEmitter,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "2imp",
                    "selection_rule": "mu",
                    "restart_rule": "no_improvement",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {},
        },
    },
    "cma_me_basic": {
        "is_dqd": False,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "dims": (100, 100),
            },
        },
        "result_archive": None,
        "emitters": [
            {
                "class": EvolutionStrategyEmitter,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "2imp",
                    "selection_rule": "mu",
                    "restart_rule": "basic",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {},
        },
    },
    "cma_me_rd": {
        "is_dqd": False,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "dims": (100, 100),
            },
        },
        "result_archive": None,
        "emitters": [
            {
                "class": EvolutionStrategyEmitter,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "2rd",
                    "selection_rule": "filter",
                    "restart_rule": "no_improvement",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {},
        },
    },
    "cma_me_rd_mu": {
        "is_dqd": False,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "dims": (100, 100),
            },
        },
        "result_archive": None,
        "emitters": [
            {
                "class": EvolutionStrategyEmitter,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "2rd",
                    "selection_rule": "mu",
                    "restart_rule": "no_improvement",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {},
        },
    },
    "cma_me_opt": {
        "is_dqd": False,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "dims": (100, 100),
            },
        },
        "result_archive": None,
        "emitters": [
            {
                "class": EvolutionStrategyEmitter,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "obj",
                    "selection_rule": "mu",
                    "restart_rule": "basic",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {},
        },
    },
    "cma_me_mixed": {
        "is_dqd": False,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "dims": (100, 100),
            },
        },
        "result_archive": None,
        "emitters": [
            {
                "class": EvolutionStrategyEmitter,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "2rd",
                    "batch_size": 36,
                },
                "num_emitters": 7,
            },
            {
                "class": EvolutionStrategyEmitter,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "2imp",
                    "batch_size": 36,
                },
                "num_emitters": 8,
            },
        ],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {},
        },
    },
    ## DQD algorithms ##
    "og_map_elites": {
        "is_dqd": True,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "dims": (100, 100),
            },
        },
        "result_archive": None,
        "emitters": [
            {
                "class": GradientOperatorEmitter,
                "kwargs": {
                    "sigma": 0.5,
                    "sigma_g": 0.5,
                    "measure_gradients": False,
                    "normalize_grad": False,
                    # Divide by 2 since half of the solutions are used in ask_dqd(),
                    # and the other half are used in ask().
                    "batch_size": 540 // 2,
                },
                "num_emitters": 1,
            }
        ],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {},
        },
    },
    "omg_mega": {
        "is_dqd": True,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "dims": (100, 100),
            },
        },
        "result_archive": None,
        "emitters": [
            {
                "class": GradientOperatorEmitter,
                "kwargs": {
                    "sigma": 0.0,
                    "sigma_g": 10.0,
                    "measure_gradients": True,
                    "normalize_grad": True,
                    # Divide by 2 since half of the solutions are used in ask_dqd(),
                    # and the other half are used in ask().
                    "batch_size": 540 // 2,
                },
                "num_emitters": 1,
            }
        ],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {},
        },
    },
    "cma_mega": {
        "is_dqd": True,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "dims": (100, 100),
            },
        },
        "result_archive": None,
        "emitters": [
            {
                "class": GradientArborescenceEmitter,
                "kwargs": {
                    "sigma0": 10.0,
                    "lr": 1.0,
                    "grad_opt": "gradient_ascent",
                    "selection_rule": "mu",
                    # Subtract 1 since one solution is used in ask_dqd() and the
                    # rest are used in ask().
                    "batch_size": 36 - 1,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {},
        },
    },
    "cma_mega_adam": {
        "is_dqd": True,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "dims": (100, 100),
            },
        },
        "result_archive": None,
        "emitters": [
            {
                "class": GradientArborescenceEmitter,
                "kwargs": {
                    "sigma0": 10.0,
                    "lr": 0.002,
                    "grad_opt": "adam",
                    "selection_rule": "mu",
                    # Subtract 1 since one solution is used in ask_dqd() and the
                    # rest are used in ask().
                    "batch_size": 36 - 1,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {},
        },
    },
    ## CMA-MAE and CMA-MAEGA ##
    "cma_mae": {
        "is_dqd": False,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "dims": (100, 100),
                "threshold_min": 0,
                "learning_rate": 0.01,
            },
        },
        "result_archive": {
            "class": GridArchive,
            "kwargs": {
                "dims": (100, 100),
            },
        },
        "emitters": [
            {
                "class": EvolutionStrategyEmitter,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "imp",
                    "selection_rule": "mu",
                    "restart_rule": "basic",
                    "batch_size": 36,
                    "es": "sep_cma_es",
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {},
        },
    },
    "cma_maega": {
        "is_dqd": True,
        "archive": {
            "class": GridArchive,
            "kwargs": {
                "dims": (100, 100),
                "threshold_min": 0,
                "learning_rate": 0.01,
            },
        },
        "result_archive": {
            "class": GridArchive,
            "kwargs": {
                "dims": (100, 100),
            },
        },
        "emitters": [
            {
                "class": GradientArborescenceEmitter,
                "kwargs": {
                    "sigma0": 10.0,
                    "lr": 1.0,
                    "ranker": "imp",
                    "grad_opt": "gradient_ascent",
                    "restart_rule": "basic",
                    # Subtract 1 since one solution is used in ask_dqd() and the
                    # rest are used in ask().
                    "batch_size": 36 - 1,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {},
        },
    },
    ## Novelty Search ##
    "ns_cma": {
        # Hyperparameters from DDS paper: https://arxiv.org/abs/2312.11331
        "is_dqd": False,
        "archive": {
            "class": ProximityArchive,
            "kwargs": {
                "k_neighbors": 15,
                "novelty_threshold": 0.037 * 512,
            },
        },
        "result_archive": {
            "class": GridArchive,
            "kwargs": {
                "dims": (100, 100),
            },
        },
        "emitters": [
            {
                "class": EvolutionStrategyEmitter,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "nov",
                    "selection_rule": "mu",
                    "restart_rule": "basic",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {},
        },
    },
    "nslc_cma_imp": {
        "is_dqd": False,
        "archive": {
            "class": ProximityArchive,
            "kwargs": {
                "k_neighbors": 15,
                # Note: This is untuned.
                "novelty_threshold": 0.037 * 100,
                "local_competition": True,
            },
        },
        "result_archive": {
            "class": GridArchive,
            "kwargs": {
                "dims": (100, 100),
            },
        },
        "emitters": [
            {
                "class": EvolutionStrategyEmitter,
                "kwargs": {
                    "sigma0": 0.5,
                    "ranker": "2imp",
                    "selection_rule": "filter",
                    "restart_rule": "no_improvement",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {},
        },
    },
    ## DDS ##
    "dds": {
        # Hyperparameters from DDS paper: https://arxiv.org/abs/2312.11331
        "is_dqd": False,
        # In DDS, the DensityArchive does not store any solutions, so emitters
        # must use the result archive instead.
        "pass_result_archive_to_emitters": True,
        "archive": {
            "class": DensityArchive,
            "kwargs": {
                "buffer_size": 10000,
                "density_method": "kde",
                "bandwidth": 25.6,
            },
        },
        "result_archive": {
            "class": GridArchive,
            "kwargs": {
                "dims": (100, 100),
            },
        },
        "emitters": [
            {
                "class": EvolutionStrategyEmitter,
                "kwargs": {
                    "sigma0": 1.5,
                    "ranker": "density",
                    "selection_rule": "mu",
                    "restart_rule": "basic",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {},
        },
    },
    "dds_kde_sklearn": {
        # Hyperparameters from DDS paper: https://arxiv.org/abs/2312.11331
        "is_dqd": False,
        # In DDS, the DensityArchive does not store any solutions, so emitters
        # must use the result archive instead.
        "pass_result_archive_to_emitters": True,
        "archive": {
            "class": DensityArchive,
            "kwargs": {
                # `density_method` and `sklearn_kwargs` are the only differences
                # from the `dds` config above. `kde_sklearn` tends to be slower
                # but it has more options available.
                "buffer_size": 10000,
                "density_method": "kde_sklearn",
                "bandwidth": 25.6,
                "sklearn_kwargs": {
                    "kernel": "gaussian",
                },
            },
        },
        "result_archive": {
            "class": GridArchive,
            "kwargs": {
                "dims": (100, 100),
            },
        },
        "emitters": [
            {
                "class": EvolutionStrategyEmitter,
                "kwargs": {
                    "sigma0": 1.5,
                    "ranker": "density",
                    "selection_rule": "mu",
                    "restart_rule": "basic",
                    "batch_size": 36,
                },
                "num_emitters": 15,
            }
        ],
        "scheduler": {
            "class": Scheduler,
            "kwargs": {},
        },
    },
}


def sphere(
    solutions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sphere function evaluation and measures for a batch of solutions.

    Args:
        solutions: (batch_size, dim) batch of solutions.

    Returns:
        objectives: (batch_size,) batch of objectives.
        objective_grads: (batch_size, solution_dim) batch of objective gradients.
        measures: (batch_size, 2) batch of measures.
        measure_grads: (batch_size, 2, solution_dim) batch of measure gradients.
    """
    dim = solutions.shape[1]

    # Shift the Sphere function so that the optimal value is at x_i = 2.048.
    sphere_shift = 5.12 * 0.4

    # Normalize the objective to the range [0, 100] where 100 is optimal.
    best_obj = 0.0
    worst_obj = (-5.12 - sphere_shift) ** 2 * dim
    raw_obj = np.sum(np.square(solutions - sphere_shift), axis=1)
    objectives = (raw_obj - worst_obj) / (best_obj - worst_obj) * 100

    # Compute gradient of the objective.
    objective_grads = -2 * (solutions - sphere_shift)

    # Calculate measures.
    clipped = solutions.copy()
    clip_mask = (clipped < -5.12) | (clipped > 5.12)
    clipped[clip_mask] = 5.12 / clipped[clip_mask]
    measures = np.concatenate(
        (
            np.sum(clipped[:, : dim // 2], axis=1, keepdims=True),
            np.sum(clipped[:, dim // 2 :], axis=1, keepdims=True),
        ),
        axis=1,
    )

    # Compute gradient of the measures.
    derivatives = np.ones(solutions.shape)
    derivatives[clip_mask] = -5.12 / np.square(solutions[clip_mask])

    mask_0 = np.concatenate((np.ones(dim // 2), np.zeros(dim - dim // 2)))
    mask_1 = np.concatenate((np.zeros(dim // 2), np.ones(dim - dim // 2)))

    d_measure0 = derivatives * mask_0
    d_measure1 = derivatives * mask_1

    measure_grads = np.stack((d_measure0, d_measure1), axis=1)

    return (
        objectives,
        objective_grads,
        measures,
        measure_grads,
    )


def create_scheduler(
    config: dict, algorithm: str, seed: int | None = None
) -> Scheduler:
    """Creates a scheduler based on the algorithm.

    Args:
        config: Configuration dictionary with parameters for the various components.
        algorithm: Name of the algorithm.
        seed: Main seed for the various components.

    Returns:
        ribs.schedulers.Scheduler: A ribs scheduler for running the algorithm.
    """
    # Properties of the Sphere problem.
    solution_dim = config["dim"]
    max_bound = solution_dim / 2 * 5.12
    bounds = [(-max_bound, max_bound), (-max_bound, max_bound)]
    initial_sol = np.zeros(solution_dim)

    # Create archive.
    archive_class = config["archive"]["class"]
    if archive_class == ProximityArchive:
        # Takes `measure_dim` instead of `ranges`.
        archive = archive_class(
            solution_dim=solution_dim,
            measure_dim=len(bounds),
            seed=seed,
            **config["archive"]["kwargs"],
        )
    elif archive_class == DensityArchive:
        archive = archive_class(
            measure_dim=len(bounds),
            seed=seed,
            **config["archive"]["kwargs"],
        )
    else:
        archive = archive_class(
            solution_dim=solution_dim,
            ranges=bounds,
            seed=seed,
            **config["archive"]["kwargs"],
        )

    # Create result archive.
    if config["result_archive"] is None:
        result_archive = None
    else:
        result_archive = config["result_archive"]["class"](
            solution_dim=solution_dim,
            # Note that using ranges here means we assume the result archive is a
            # GridArchive or CVTArchive. This will need to be modified for other result
            # archives.
            ranges=bounds,
            seed=seed,
            **config["result_archive"]["kwargs"],
        )

    # Usually, emitters take in the archive. However, it may sometimes be necessary to
    # take in the result_archive, such as in DDS.
    archive_for_emitter = (
        result_archive if config.get("pass_result_archive_to_emitters") else archive
    )

    # Create emitters. Each emitter needs a different seed so that they do not all do
    # the same thing, hence we create an rng here to generate seeds. The rng may be
    # seeded with None or with a user-provided seed.
    seed_sequence = np.random.SeedSequence(seed)
    emitters = []
    for e in config["emitters"]:
        emitter_class = e["class"]
        emitters += [
            emitter_class(
                archive_for_emitter,
                x0=initial_sol,
                **e["kwargs"],
                seed=s,
            )
            for s in seed_sequence.spawn(e["num_emitters"])
        ]

    # Create Scheduler
    scheduler = config["scheduler"]["class"](
        archive,
        emitters,
        result_archive,
        **config["scheduler"]["kwargs"],
    )

    print(
        f"Create {scheduler.__class__.__name__} for {algorithm} "
        f"using solution dim {solution_dim} and {len(emitters)} emitters."
    )
    return scheduler


def save_heatmap(archive: ArchiveBase, heatmap_path: str | Path) -> None:
    """Saves a heatmap of the archive to the given path.

    Args:
        archive: The archive to save.
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
    else:
        raise NotImplementedError(
            "This script currently does not plot heatmaps for this archive."
        )
    plt.close(plt.gcf())


def sphere_main(
    algorithm: str,
    dim: int = 100,
    itrs: int = 10000,
    grid_dims: tuple[int, int] | None = None,
    learning_rate: float | None = None,
    es: str | None = None,
    outdir: str = "sphere_output",
    log_freq: int = 250,
    seed: int | None = None,
) -> None:
    """Demo on the Sphere function.

    Args:
        algorithm: Name of the algorithm.
        dim: Dimensionality of the sphere function.
        itrs: Iterations to run.
        grid_dims: Grid dimensions for GridArchive.
        learning_rate: The archive learning rate.
        es: If passed, this will set the ES for all EvolutionStrategyEmitter instances.
        outdir: Directory to save output.
        log_freq: Number of iterations to wait before recording metrics and saving
            heatmap.
        seed: Seed for the algorithm. By default, there is no seed.
    """
    config = copy.deepcopy(CONFIG[algorithm])

    # Add params that are not in the config.
    config["dim"] = dim
    config["itrs"] = itrs

    # Add params that are in the config by default but may be passed in.
    if grid_dims is not None:
        if config["archive"]["class"] == GridArchive:
            config["archive"]["kwargs"]["dims"] = grid_dims
        if config["result_archive"]["class"] == GridArchive:
            config["result_archive"]["kwargs"]["dims"] = grid_dims
    if learning_rate is not None:
        config["archive"]["kwargs"]["learning_rate"] = learning_rate
    if es is not None:
        # Set ES for all EvolutionStrategyEmitter.
        for e in config["emitters"]:
            if e["class"] == EvolutionStrategyEmitter:
                e["kwargs"]["es"] = es

    name = f"{algorithm}_{config['dim']}"
    if es is not None:
        name += f"_{es}"
    outdir: Path = Path(outdir)
    if not outdir.is_dir():
        outdir.mkdir()

    scheduler = create_scheduler(config, algorithm, seed=seed)
    result_archive = scheduler.result_archive
    is_dqd = config["is_dqd"]
    itrs = config["itrs"]
    metrics = {
        "QD Score": {
            "x": [0],
            "y": [result_archive.stats.qd_score],
        },
        "Archive Coverage": {
            "x": [0],
            "y": [result_archive.stats.coverage],
        },
    }

    non_logging_time = 0.0
    save_heatmap(result_archive, str(outdir / f"{name}_heatmap_{0:05d}.png"))

    for itr in tqdm.trange(1, itrs + 1):
        itr_start = time.time()

        if is_dqd:
            solutions = scheduler.ask_dqd()
            (objectives, objective_grads, measures, measure_grads) = sphere(solutions)
            objective_grads = np.expand_dims(objective_grads, axis=1)
            jacobians = np.concatenate((objective_grads, measure_grads), axis=1)
            scheduler.tell_dqd(objectives, measures, jacobians)

        solutions = scheduler.ask()
        objectives, _, measures, _ = sphere(solutions)
        scheduler.tell(objectives, measures)
        non_logging_time += time.time() - itr_start

        # Logging and output.
        final_itr = itr == itrs
        if itr % log_freq == 0 or final_itr:
            if final_itr:
                result_archive.data(return_type="pandas").to_csv(
                    outdir / f"{name}_archive.csv"
                )

            # Record and display metrics.
            metrics["QD Score"]["x"].append(itr)
            metrics["QD Score"]["y"].append(result_archive.stats.qd_score)
            metrics["Archive Coverage"]["x"].append(itr)
            metrics["Archive Coverage"]["y"].append(result_archive.stats.coverage)
            tqdm.tqdm.write(
                f"Iteration {itr} | Archive Coverage: "
                f"{metrics['Archive Coverage']['y'][-1] * 100:.3f}% "
                f"QD Score: {metrics['QD Score']['y'][-1]:.3f}"
            )

            save_heatmap(result_archive, str(outdir / f"{name}_heatmap_{itr:05d}.png"))

    # Plot metrics.
    print(f"Algorithm Time (Excludes Logging and Setup): {non_logging_time}s")
    for metric, values in metrics.items():
        plt.plot(values["x"], values["y"])
        plt.title(metric)
        plt.xlabel("Iteration")
        plt.savefig(str(outdir / f"{name}_{metric.lower().replace(' ', '_')}.png"))
        plt.clf()

    # Convert metrics to Python scalars by calling .item(), since each stats value is a
    # 0-D array by default, and JSON cannot serialize 0-D arrays.
    for metric in metrics:
        metrics[metric]["y"] = [
            m if isinstance(m, (int, float)) else m.item() for m in metrics[metric]["y"]
        ]

    # Save metrics to JSON.
    with (outdir / f"{name}_metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)


if __name__ == "__main__":
    fire.Fire(sphere_main)
