"""Learn a linear model with MAP-Elites in a discrete OpenAI Gym environment.

Uses Dask for parallelization.

Usage:
    See README.md.
"""

import time

import fire
import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dask.distributed import Client, LocalCluster

from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.optimizers import Optimizer


def simulate(
    env_name: str,
    model,
    seed: int = None,
    render: bool = False,
    delay: int = 10,
):
    """Runs the model in the env and returns the cumulative reward.

    Add the `seed` argument to initialize the environment from the given seed
    (this makes the environment the same between runs). The model is just a
    linear model from input to output with softmax, so it is represented by a
    single (action_dim, obs_dim) matrix. Add an integer delay to wait `delay` ms
    between timesteps.
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
        # If render is set to True, then a video will appear showing the Lunar
        # Lander taking actions in the environment.
        if render:
            env.render()
            if delay is not None:
                time.sleep(delay / 1000)

        # Deterministic. Here is the action. Multiply observation by policy.
        # Model is the policy and obs is state
        action = np.argmax(model @ obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        timesteps += 1

    env.close()

    return total_reward, obs[0], timesteps


def train_model(
    client: Client,
    seed: int,
    sigma: float,
    model_filename: str,
    plot_filename: str,
    iterations: int,
    env_name: str = "LunarLander-v2",
):
    """Trains a model with MAP-Elites and saves it."""
    # OpenAI Gym environment properties.
    env = gym.make(env_name)
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]

    archive = GridArchive((16, 16), [(0, 1000), (-1., 1.)], seed=seed)
    emitter = GaussianEmitter(archive,
                              np.zeros(action_dim * obs_dim),
                              sigma,
                              batch_size=64)
    opt = Optimizer(archive, [emitter])

    for _ in range(0, iterations - 1):

        # Generating a batch of solutions
        sols = opt.ask()

        objs = list()
        bcs = list()

        # Here, we're running each of the solutions (i.e. policies) we generated
        # above through the simulate() function. simulate() will return the
        # objective value, timesteps to run to completion, and x-position of the
        # lunar lander for each solution we pass in.
        futures = client.map(
            lambda sol: simulate(env_name, np.reshape(sol, (action_dim, obs_dim)
                                                     ), seed), sols)

        results = client.gather(futures)

        # Here we're just constructing a list of objective function evaluations
        # (i.e. objs) and behavior descriptions (i.e. bcs) for each solution.
        # These values were returned by our calls to simulation() above.
        for reward, x_pos, timesteps in results:
            objs.append(reward)
            bcs.append((timesteps, x_pos))

        # We have our Optimizer opt tell our Emitters the objective function
        # evaluations and behavior descriptions of each solution, so that our
        # Emitter emitter and GridArchive archive can decide where and if to
        # store each solution in our GridArchive archive.
        opt.tell(objs, bcs)

    df = archive.as_pandas()

    # Saving our archive to a file.
    df.to_pickle(model_filename)

    df = archive.as_pandas()
    df = df.pivot('index_0', 'index_1', 'objective')

    # Create a heatmap with all of our generated solutions.
    sns.heatmap(df)
    plt.savefig(plot_filename)


def run_evaluation(model_filename, env_name, seed):
    """Runs a single simulation and displays the results."""
    model = np.load(model_filename)
    print("=== Model ===")
    print(model)
    cost = simulate(env_name, model, seed, True, 10)
    print("Reward:", -cost[0])


def map_elites(
    seed: int = 42,
    local_workers: int = 8,
    sigma: float = 10.0,
    plot_filename: str = "lunar_lander_plot.png",
    model_filename: str = "lunar_lander_model.pkl",
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

    train_model(client, seed, sigma, model_filename, plot_filename, 10)


if __name__ == "__main__":
    fire.Fire(map_elites)
