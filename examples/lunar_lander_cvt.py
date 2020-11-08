"""Trains a linear model with CVT-MAP-Elites in the OpenAI Gym Lunar Lander env.

The model is a simple (action_dim, obs_dim) matrix; the policy at each timestep
is computed as::

    argmax ( model @ obs )

As this example uses CVT-MAP-Elites, we are able to use a very high-dimensional
behavior space, so we use the space of trajectories sampled every 100 timesteps
up to 1000 timesteps -- as each point in the trajectory is (x,y) and we include
the starting point, our trajectories are 22-dimensional. If the episode lasts
shorter than 1000 timesteps, the lander's last position is repeated.

For bounds, each dimension is bounded to (-1,1), as those are the bounds on the
lunar lander's x and y positions.

The CVT archive in this example uses 5000 bins, but the default parameters only
filled up ~200 entries. More iterations are likely needed (much more than 1000),
and some trajectories are physically impossible anyway.

This script uses Dask for parallelization.

Usage:
    # For default parameters, use the following command. This command took ~2
    # hours for me and filled up a very small portion of the archive. It may run
    # faster with more workers (add `--workers N`) and may cover more of the
    # archive with more iterations (add `--iterations N`).
    python lunar_lander_cvt.py

    # For full options, run:
    python lunar_lander_cvt.py --help
"""

import time

import fire
import gym
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster

from ribs.archives import CVTArchive
from ribs.emitters import GaussianEmitter
from ribs.optimizers import Optimizer


def simulate(model, seed=None, render=False, delay=None):
    """Runs the model in the env and returns reward and BCs.

    The model is just a linear model from observations to actions with an argmax
    to select the final action, so it is represented by a single matrix.

    Args:
        model ((action_dim * obs_dim,) array): A 1D array with the model
            weights.
        seed (int): Seed for the environment (makes the environment the same
            between runs).
        render (bool): Whether to render the environment.
        delay (int): Milliseconds to wait between frames when `render` is True.
    Returns:
        reward (float): Total reward of the agent.
        trajectory ((22,) array): The trajectory of the agent every 100
            timesteps -- since there are up to 1000 timesteps, and each point in
            the trajectory is an (x,y) coordinate, and we count the starting
            position, the array's length is 22. Each entry in the trajectory
            runs from -1 to 1. If the trajectory is not long enough (i.e. the
            episode terminates early), the last point in the trajectory is
            repeated until the trajectory is long enough.
    """
    total_reward = 0.0
    env = gym.make("LunarLander-v2")

    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    model = np.reshape(model, (action_dim, obs_dim))

    # Deterministic environment with seed().
    if seed is not None:
        env.seed(seed)
    obs = env.reset()

    timesteps = 0
    trajectory = [obs[0], obs[1]]

    done = False
    while not done:
        if render:
            env.render()
            if delay is not None:
                time.sleep(delay / 1000)

        action = np.argmax(model @ obs)  # Deterministic action selection.
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        timesteps += 1

        if timesteps % 100 == 0:
            trajectory.extend([obs[0], obs[1]])  # (x,y) coordinates.

    env.close()

    while len(trajectory) < 22:
        trajectory.extend(trajectory[-2:])

    return total_reward, np.array(trajectory, dtype=np.float32)


def train_model(create_client, seed, sigma, archive_filename, iterations):
    """Trains models with CVT-MAP-Elites and saves results to a pickle file.

    Args:
        create_client (callable): Function which returns a Dask client when
            called with no parameters.
        seed (int): Seed for environment.
        sigma (int): Standard deviation for the GaussianEmitter.
        archive_filename (str): Pickle file to save the archive to.
        iterations (int): Number of iterations to run CVT-MAP-Elites.
    """
    env = gym.make("LunarLander-v2")
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    env.close()

    print("Constructing CVTArchive -- may take a while...")
    archive = CVTArchive([(-1., 1.)] * 22,
                         5_000,
                         config={
                             "seed": seed,
                             "samples": 100_000,
                         })
    emitter = GaussianEmitter(np.zeros(action_dim * obs_dim),
                              sigma,
                              archive,
                              config={
                                  "seed": seed,
                                  "batch_size": 32,
                              })
    opt = Optimizer(archive, [emitter])

    # Since scipy's k-means uses multiple threads in CVTArchive, it throttles
    # Dask (nothing breaks, but some error messages are thrown, which may be
    # confusing). Hence, we wait until here to create the Dask workers and
    # client.
    client = create_client()

    print("Starting search")
    start_time = time.time()
    for itr in range(iterations):
        sols = opt.ask()

        futures = client.map(lambda sol: simulate(sol, seed), sols)
        results = client.gather(futures)

        objs = []
        bcs = []
        for reward, trajectory in results:
            objs.append(reward)
            bcs.append(trajectory)

        opt.tell(objs, bcs)

        if (itr + 1) % 10 == 0:
            print(f"Completed iteration {itr + 1} after "
                  f"{time.time() - start_time} s")
            archive.as_pandas().to_pickle(archive_filename)

    df = archive.as_pandas()
    df.to_pickle(archive_filename)
    print("=== Done ===\n"
          f"Saved archive to {archive_filename}\n"
          f"Time: {time.time() - start_time}\n"
          f"{len(df)} entries in archive\n"
          "=== Archive head ===\n"
          f"{df.head()}")


def run_evaluation(archive_filename, seed):
    """Runs simulations on 10 random archive solutions."""
    df = pd.read_pickle(archive_filename)
    indices = np.random.permutation(len(df))[:10]
    indices.sort()

    for idx in indices:
        model = df.at[idx, "solution"]
        print(f"=== Index {idx} ===\n"
              "Model:\n"
              f"{model}\n"
              f"Reward: {simulate(model, seed, True, 10)[0]}")


def cvt_map_elites(
    iterations: int = 1000,
    seed: int = 42,
    sigma: float = 10.0,
    workers: int = 4,
    archive_filename: str = "lunar_lander_cvt_archives.pkl",
    run_eval: bool = False,
):
    """Uses CVT-MAP-Elites to train an agent in Lunar Lander.

    Args:
        iterations: Number of iterations to run the algorithm.
        seed: Random seed for environments.
        sigma: Standard deviation for the Gaussian emitter.
        workers: Number of workers to use when running locally.
        archive_filename: Location for CSV file with the archive.
        run_eval: Pass this to run an evaluation in the environment in `env`
            with the model in `archive_filename`.
    """
    # Evaluations do not need Dask.
    if run_eval:
        run_evaluation(archive_filename, seed)
        return

    # A function that initializes on a local machine. We create `workers`
    # processes with one thread per process.
    def create_client():
        cluster = LocalCluster(n_workers=workers,
                               threads_per_worker=1,
                               processes=True)
        client = Client(cluster)  # pylint: disable=unused-variable
        print("Cluster config:")
        print(client.ncores())
        return client

    train_model(create_client, seed, sigma, archive_filename, iterations)


if __name__ == "__main__":
    fire.Fire(cvt_map_elites)
