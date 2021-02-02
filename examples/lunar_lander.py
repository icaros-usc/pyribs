"""Uses CMA-ME to train linear agents in Lunar Lander.

This script uses the same setup as the tutorial, but it also uses Dask to
parallelize evaluations on a single machine and adds in a CLI. Refer to the
tutorial here: https://docs.pyribs.org/en/latest/tutorials/lunar_lander.html for
more info.

You should not need much familiarity with Dask to read this example. However, if
you would like to know more about Dask, we recommend referring to the quickstart
for Dask distributed: https://distributed.dask.org/en/latest/quickstart.html.

This script creates an output directory (defaults to `lunar_lander_output/`, see
the --outdir flag) with the following files:

    - archive.csv: The CSV representation of the final archive, obtained with
      as_pandas().
    - archive_ccdf.png: A plot showing the (unnormalized) complementary
      cumulative distribution function of objective values in the archive. For
      each objective value p on the x-axis, this plot shows the number of
      solutions that had an objective value of at least p.
    - heatmap.png: A heatmap showing the performance of solutions in the
      archive.
    - metrics.json: Metrics about the run, saved as a mapping from the metric
      name to a list of x values (iteration number) and a list of y values
      (metric value) for that metric.
    - {metric_name}.png: Plots of the metrics, currently just `archive_size` and
      `max_score`.

In evaluation mode (--run-eval flag), the script will read in the archive from
the output directory and simulate 10 random solutions from the archive. It will
write videos of these simulations to a `videos/` subdirectory in the output
directory.

Usage:
    # Basic usage - should take ~1 hour with 4 cores.
    python lunar_lander.py NUM_WORKERS
    # Now open the Dask dashboard at http://localhost:8787 to view worker
    # status.

    # Evaluation mode. If you passed a different outdir and/or env_seed when
    # running the algorithm with the command above, you must pass the same
    # outdir and/or env_seed here.
    python lunar_lander.py --run-eval
Help:
    python lunar_lander.py --help
"""
import json
import time
from pathlib import Path

import fire
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alive_progress import alive_bar
from dask.distributed import Client, LocalCluster

from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap


def simulate(model, seed=None, video_env=None):
    """Simulates the lunar lander model.

    Args:
        model (np.ndarray): The array of weights for the linear policy.
        seed (int): The seed for the environment.
        video_env (gym.Env): If passed in, this will be used instead of creating
            a new env. This is used primarily for recording video during
            evaluation.
    Returns:
        total_reward (float): The reward accrued by the lander throughout its
            trajectory.
        impact_x_pos (float): The x position of the lander when it touches the
            ground for the first time.
        impact_y_vel (float): The y velocity of the lander when it touches the
            ground for the first time.
    """
    if video_env is None:
        # Since we are using multiple processes, it is simpler if each worker
        # just creates their own copy of the environment instead of trying to
        # share the environment. This also makes the function "pure."
        env = gym.make("LunarLander-v2")
    else:
        env = video_env

    if seed is not None:
        env.seed(seed)

    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    model = model.reshape((action_dim, obs_dim))

    total_reward = 0.0
    impact_x_pos = None
    impact_y_vel = None
    all_y_vels = []
    obs = env.reset()
    done = False

    while not done:
        action = np.argmax(model @ obs)  # Linear policy.
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        # Refer to the definition of state here:
        # https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py#L306
        x_pos = obs[0]
        y_vel = obs[3]
        leg0_touch = bool(obs[6])
        leg1_touch = bool(obs[7])
        all_y_vels.append(y_vel)

        # Check if the lunar lander is impacting for the first time.
        if impact_x_pos is None and (leg0_touch or leg1_touch):
            impact_x_pos = x_pos
            impact_y_vel = y_vel

    # If the lunar lander did not land, set the x-pos to the one from the final
    # timestep, and set the y-vel to the max y-vel (we use min since the lander
    # goes down).
    if impact_x_pos is None:
        impact_x_pos = x_pos
        impact_y_vel = min(all_y_vels)

    # Only close the env if it was not a video env.
    if video_env is None:
        env.close()

    return total_reward, impact_x_pos, impact_y_vel


def create_optimizer(seed, n_emitters, sigma0, batch_size):
    """Creates the Optimizer based on given configurations.

    See lunar_lander_main() for description of args.

    Returns:
        A pyribs optimizer set up for CMA-ME (i.e. it has ImprovementEmitter's
        and a GridArchive).
    """
    env = gym.make("LunarLander-v2")
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]

    archive = GridArchive(
        [50, 50],  # 50 bins in each dimension.
        [(-1.0, 1.0), (-3.0, 0.0)],  # (-1, 1) for x-pos and (-3, 0) for y-vel.
        seed=seed,
    )

    # If we create the emitters with identical seeds, they will all output the
    # same initial solutions. The algorithm should still work -- eventually, the
    # emitters will produce different solutions because they get different
    # responses when inserting into the archive. However, using different seeds
    # avoids this problem altogether.
    seeds = ([None] * n_emitters
             if seed is None else [seed + i for i in range(n_emitters)])
    initial_model = np.zeros((action_dim, obs_dim))
    emitters = [
        ImprovementEmitter(
            archive,
            initial_model.flatten(),
            sigma0=sigma0,
            batch_size=batch_size,
            seed=s,
        ) for s in seeds
    ]

    optimizer = Optimizer(archive, emitters)
    return optimizer


def run_search(client, optimizer, env_seed, iterations, log_freq):
    """Runs the QD algorithm for the given number of iterations.

    Args:
        client (Client): A Dask client providing access to workers.
        optimizer (Optimizer): pyribs optimizer.
        env_seed (int): Seed for the environment.
        iterations (int): Iterations to run.
        log_freq (int): Number of iterations to wait before recording metrics.
    Returns:
        dict: A mapping from various metric names to a list of "x" and "y"
        values where x is the iteration and y is the value of the metric. Think
        of each entry as the x's and y's for a matplotlib plot.
    """
    print(
        "> Starting search.\n"
        "  - Open Dask's dashboard at http://localhost:8787 to monitor workers."
    )

    metrics = {
        "Max Score": {
            "x": [],
            "y": [],
        },
        "Archive Size": {
            "x": [0],
            "y": [0],
        },
    }

    start_time = time.time()
    with alive_bar(iterations) as progress:
        for itr in range(1, iterations + 1):
            # Request models from the optimizer.
            sols = optimizer.ask()

            # Evaluate the models and record the objectives and BCs.
            objs, bcs = [], []

            # Ask the Dask client to distribute the simulations among the Dask
            # workers, then gather the results of the simulations.
            futures = client.map(lambda model: simulate(model, env_seed), sols)
            results = client.gather(futures)

            # Process the results.
            for obj, impact_x_pos, impact_y_vel in results:
                objs.append(obj)
                bcs.append([impact_x_pos, impact_y_vel])

            # Send the results back to the optimizer.
            optimizer.tell(objs, bcs)

            # Logging.
            progress()
            if itr % log_freq == 0 or itr == iterations:
                df = optimizer.archive.as_pandas(include_solutions=False)
                elapsed_time = time.time() - start_time
                metrics["Max Score"]["x"].append(itr)
                metrics["Max Score"]["y"].append(df["objective"].max())
                metrics["Archive Size"]["x"].append(itr)
                metrics["Archive Size"]["y"].append(len(df))
                print(f"> {itr} itrs completed after {elapsed_time:.2f} s")
                print(f"  - Max Score: {metrics['Max Score']['y'][-1]}")
                print(f"  - Archive Size: {metrics['Archive Size']['y'][-1]}")

    return metrics


def save_heatmap(archive, filename):
    """Saves a heatmap of the optimizer's archive to the filename.

    Args:
        archive (GridArchive): Archive with results from an experiment.
        filename (str): Path to an image file.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=-300, vmax=300, ax=ax)
    ax.invert_yaxis()  # Makes more sense if larger velocities are on top.
    ax.set_ylabel("Impact y-velocity")
    ax.set_xlabel("Impact x-position")
    fig.savefig(filename)


def save_metrics(outdir, metrics):
    """Saves metrics to png plots and a JSON file.

    Args:
        outdir (Path): output directory for saving files.
        metrics (dict): Metrics as output by run_search.
    """
    # Plots.
    for metric in metrics:
        fig, ax = plt.subplots()
        ax.plot(metrics[metric]["x"], metrics[metric]["y"])
        ax.set_title(metric)
        ax.set_xlabel("Iteration")
        fig.savefig(str(outdir / f"{metric.lower().replace(' ', '_')}.png"))

    # JSON file.
    with (outdir / "metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)


def save_ccdf(archive, filename):
    """Saves a CCDF showing the distribution of the archive's objective values.

    CCDF = Complementary Cumulative Distribution Function (see
    https://en.wikipedia.org/wiki/Cumulative_distribution_function#Complementary_cumulative_distribution_function_(tail_distribution)).
    The CCDF plotted here is not normalized to the range (0,1). This may help
    when comparing CCDF's among archives with different amounts of coverage
    (i.e. when one archive has more cells filled).

    Args:
        archive (GridArchive): Archive with results from an experiment.
        filename (str): Path to an image file.
    """
    fig, ax = plt.subplots()
    ax.hist(
        archive.as_pandas(include_solutions=False)["objective"],
        50,  # Number of bins.
        histtype="step",
        density=False,
        cumulative=-1)  # CCDF rather than CDF.
    ax.set_xlabel("Objective Value")
    ax.set_ylabel("Num. Entries")
    ax.set_title("Distribution of Archive Objective Values")
    fig.savefig(filename)


def run_evaluation(outdir, env_seed):
    """Simulates 10 random archive solutions and saves videos of them.

    Videos are saved to outdir / videos.

    Args:
        outdir (Path): Path object for the output directory from which to
            retrieve the archive and save videos.
        env_seed (int): Seed for the environment.
    """
    df = pd.read_csv(outdir / "archive.csv")
    indices = np.random.permutation(len(df))[:10]

    # Use a single env so that all the videos go to the same directory.
    video_env = gym.wrappers.Monitor(
        gym.make("LunarLander-v2"),
        str(outdir / "videos"),
        force=True,
        # Default is to write the video for "cubic" episodes -- 0,1,8,etc (see
        # https://github.com/openai/gym/blob/master/gym/wrappers/monitor.py#L54).
        # This will ensure all the videos are written.
        video_callable=lambda idx: True,
    )

    for idx in indices:
        model = np.array(df.loc[idx, "solution_0":])
        reward, impact_x_pos, impact_y_vel = simulate(model, env_seed,
                                                      video_env)
        print(f"=== Index {idx} ===\n"
              "Model:\n"
              f"{model}\n"
              f"Reward: {reward}\n"
              f"Impact x-pos: {impact_x_pos}\n"
              f"Impact y-vel: {impact_y_vel}\n")

    video_env.close()


def lunar_lander_main(workers=4,
                      env_seed=1339,
                      iterations=500,
                      log_freq=25,
                      n_emitters=5,
                      batch_size=30,
                      sigma0=1.0,
                      seed=None,
                      outdir="lunar_lander_output",
                      run_eval=False):
    """Uses CMA-ME to train linear agents in Lunar Lander.

    Args:
        workers (int): Number of workers to use for simulations.
        env_seed (int): Environment seed. The default gives the flat terrain
            from the tutorial.
        iterations (int): Number of iterations to run the algorithm.
        log_freq (int): Number of iterations to wait before recording metrics
            and saving heatmap.
        n_emitters (int): Number of emitters.
        batch_size (int): Batch size of each emitter.
        sigma0 (float): Initial step size of each emitter.
        seed (seed): Random seed for the pyribs components.
        outdir (str): Directory for Lunar Lander output.
        run_eval (bool): Pass this flag to run an evaluation of 10 random
            solutions selected from the archive in the `outdir`.
    """
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    if run_eval:
        run_evaluation(outdir, env_seed)
        return

    # Setup Dask. The client connects to a "cluster" running on this machine.
    # The cluster simply manages several concurrent worker processes. If using
    # Dask across many workers, we would set up a more complicated cluster and
    # connect the client to it.
    cluster = LocalCluster(
        processes=True,  # Each worker is a process.
        n_workers=workers,  # Create this many worker processes.
        threads_per_worker=1,  # Each worker process is single-threaded.
    )
    client = Client(cluster)

    # CMA-ME.
    optimizer = create_optimizer(seed, n_emitters, sigma0, batch_size)
    metrics = run_search(client, optimizer, env_seed, iterations, log_freq)

    # Outputs.
    optimizer.archive.as_pandas().to_csv(outdir / "archive.csv")
    save_ccdf(optimizer.archive, str(outdir / "archive_ccdf.png"))
    save_heatmap(optimizer.archive, str(outdir / "heatmap.png"))
    save_metrics(outdir, metrics)


if __name__ == "__main__":
    fire.Fire(lunar_lander_main)
