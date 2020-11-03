"""Learn a linear model with CMA-ES in a discrete OpenAI Gym environment.
Uses Dask for parallelization.
Usage:
    See README.md.
"""

import time

import cma
import fire
import gym
import matplotlib.pyplot
import numpy as np
from dask_jobqueue import SLURMCluster

from dask.distributed import Client, LocalCluster

import matplotlib.pyplot as plt
import seaborn as sns

from ribs.archives import GridArchive
from ribs.optimizers import Optimizer

cma.s.figsave = matplotlib.pyplot.savefig  # See https://github.com/CMA-ES/pycma/issues/131

# pylint: disable = too-many-arguments


def simulate(
    env_name: str,
    model,
    seed: int = None,
    render: bool = False,
    delay: int = 10,
):
    """Runs the model in the env and returns the cumulative cost (negative reward).
    Add the `seed` argument to initialize the environment from the given seed
    (this makes the environment the same between runs).
    The model is just a linear model from input to output with softmax, so it
    is represented by a single (action_dim, obs_dim) matrix.
    Add an integer delay to wait `delay` ms between timesteps.
    """
    total_reward = 0.0
    env = gym.make(env_name)

    # Seeding the environment before each reset ensures that our simulations are
    # deterministic. We cannot vary the environment between the runs because
    # that would confuse CMA-ES. See
    # https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py#L115
    # for the implementation of seed() for LunarLander.
    if seed is not None:
        env.seed(seed)
    obs = env.reset()

    timesteps = 0

    done = False
    while not done:
        if render:
            env.render()
        if delay is not None:
            time.sleep(delay / 1000)
        action = np.argmax(model @ obs)  # Deterministic.
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        time_steps += 1

    env.close()

    return total_reward, obs, timesteps


def save_model(model, filename, verbose=False):
    """Saves the model to the given file."""
    np.save(filename, model)
    if verbose:
        print("Model saved to", filename)


def train_model(
    client: Client,
    env_name: str = "LunarLander-v2",
    seed: int,
    sigma: float,
    model_filename: str,
    plot_filename: str,
    iterations: int,
):
    """Trains a model with CMA-ES and saves it."""
    # Environment properties.
    env = gym.make(env_name)
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]

    config = {
        "seed": seed,
        "batch_size": 64,
    }


    archive = GridArchive((16, 16), [(-4, 4), (-4, 4)], config=config)
    opt = Optimizer(np.zeros(action_dim * obs_dim), sigma, archive, config=config)

    for i in range(0, iterations - 1):

        sols = opt.ask()

        objs = list()
        bcs = list()

        futures = client.map(lambda sol: simulate(env_name, np.reshape(sol, (action_dim, obs_dim)), seed), sols)

        results = client.gather(futures)

        for reward, state, timesteps in results:
            objs.append(reward)
            bcs.append((timesteps, state[0]))

        opt.tell(sols, objs, bcs)


    df = archive.as_pandas()

    df.to_csv(model_filename)

    df = archive.as_pandas()
    df = df.pivot('index-0', 'index-1', 'objective')
    sns.heatmap(df)
    plt.savefig('lunar_landerV2-map-elites.png')


def run_evaluation(model_filename, env_name, seed):
    """Runs a single simulation and displays the results."""
    model = np.load(model_filename)
    print("=== Model ===")
    print(model)
    cost = simulate(env_name, model, seed, True, 10)
    print("Reward:", -cost)


def cma_es_discrete(
    seed: int = 42,
    local_workers: int = 8,
    sigma: float = 10.0,
    slurm: bool = False,
    slurm_workers: int = 2,
    slurm_cpus_per_worker: int = 4,
    plot_filename: str = "lunar_lander_plot.png",
    model_filename: str = "lunar_lander_model.npy",
    run_eval: bool = False,
):
    """Uses CMA-ES to train an agent in an environment with discrete actions.
    Args:
        env: OpenAI Gym environment name. The environment should have a discrete
            action space.
        seed: Random seed for environments.
        sigma: Initial standard deviation for CMA-ES.
        local_workers: Number of workers to use when running locally.
        slurm: Set to True if running on Slurm.
        slurm_workers: Number of workers to start when running on Slurm.
        slurm_cpus_per_worker: Number of CPUs to use on each Slurm worker.
        plot_filename: Location to store plot image.
        model_filename: Location for .npy model file (either for storing or
            reading).
        run_eval: Pass this to run an evaluation in the environment in `env`
            with the model in `model_filename`.
    """
    # Evaluations do not need Dask.
    if run_eval:
        run_evaluation(model_filename, "LunarLander-v2", seed)
        return

        # Initialize on a local machine. See the docs here:
        # https://docs.dask.org/en/latest/setup/single-distributed.html for more
        # info on LocalCluster. Keep in mind that for LocalCluster, the
        # n_workers is the number of processes. Our LunarLander evaluations do
        # not release the GIL (I think), so using threads instead of processes
        # (which we would do by setting n_workers=1 and
        # threads_per_worker=workers) would be very slow, as it would be
        # single-threaded. See here for a bit more info about processes in
        # threads in Dask:
        # https://distributed.dask.org/en/latest/worker.html#thread-pool
        # The link above is for multiple machines (each machine is called a
        # worker, and each workers has processes and threads), but the idea
        # still holds.
    cluster = LocalCluster(n_workers=local_workers,
                           threads_per_worker=1,
                           processes=True)
    client = Client(cluster)  # pylint: disable=unused-variable
    print("Cluster config:")
    print(client.ncores())

    train_model(client, "LunarLander-v2", seed, sigma, model_filename, plot_filename)


if __name__ == "__main__":
    fire.Fire(cma_es_discrete)