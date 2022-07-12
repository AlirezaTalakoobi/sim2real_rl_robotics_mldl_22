from email import policy
from operator import contains
from unittest.mock import Base
import matplotlib.pyplot as plt

import torch
import gym
import argparse
from sb3_contrib import TRPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor



import os
from env.custom_hopper import *
# from agent import Agent, Policy


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-timesteps", default=100000, type=int, help="Number of training episodes"
    )
    parser.add_argument(
        "--print-every", default=1000, type=int, help="Print info every <> episodes"
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="network device [cpu, cuda]"
    )
    parser.add_argument(
        "--lr", default="1e-3", type=float, help="lrarning rate"
    )

    return parser.parse_args()


args = parse_args()


def main():
    # Create log dir
    log_dir = "/tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)
    # torch.cuda.set_per_process_memory_fraction(0.7, 0)

    env = gym.make("CustomHopper-source-v0")
    env = Monitor(env, log_dir)
    # env = gym.make('CustomHopper-target-v0')

    print("Action space:", env.action_space)
    print("State space:", env.observation_space)
    print("Dynamics parameters:", env.get_parameters())

    """
		Training
	"""

    model = TRPO('MlpPolicy', env,learning_rate = args.lr,device= args.device,tensorboard_log="./a2c_cartpole_tensorboard/")
    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps = args.n_timesteps, callback=callback)

    model.save("model.mdl")

    from stable_baselines3.common import results_plotter

# Helper from the library
    results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "TRPO CustomHopper")


    def moving_average(values, window):

    # Smooth values by doing a moving average
    # :param values: (numpy array)
    # :param window: (int)
    # :return: (numpy array)

        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, 'valid')


    def plot_results(log_folder, title='Learning Curve'):
        """
        plot the results

        :param log_folder: (str) the save location of the results to plot
        :param title: (str) the title of the task to plot
        """
        x, y = ts2xy(load_results(log_folder), 'timesteps')
        y = moving_average(y, window=50)
        # Truncate x
        x = x[len(x) - len(y):]

        fig = plt.figure(title)
        plt.plot(x, y)
        plt.xlabel('Number of Timesteps')
        plt.ylabel('Rewards')
        plt.title(title + " Smoothed")
        print("saved")
        plt.savefig('TRPO.png')
        plt.show()

    plot_results(log_dir)

if __name__ == "__main__":
    main()
