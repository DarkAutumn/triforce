import sys
import numpy as np
import retro
import os
import gymnasium as gym
from random import randint

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

class Frameskip(gym.Wrapper):
    def __init__(self, env, skip_min, skip_max):
        super().__init__(env)
        self._skip_min = skip_min
        self._skip_max = skip_max

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, act):
        total_rew = 0.0
        terminated = False
        truncated = False
        for i in range(randint(self._skip_min, self._skip_max)):
            obs, rew, terminated, truncated, info = self.env.step(act)
            total_rew += rew
            if terminated or truncated:
                break

        return obs, total_rew, terminated, truncated, info

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
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
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True





def test_environment(env, iterations):
    total_reward = 0

    env.reset()
    for _ in range(iterations):
        obs, reward, done, d, info = env.step(env.action_space.sample())
        if reward:
            print(f'reward: {reward} total:{total_reward}')
            total_reward += reward
        env.render()
        if done:
            env.reset()

def evaluate(model, env, episodes=20):
    mean_reward, std_reward = evaluate_policy(model, env, render=True, n_eval_episodes=episodes)
    print(f'Mean reward: {mean_reward} +/- {std_reward}')

def train(env, timesteps= 2_000_000):
    log_path = os.path.join('/output/', f'ppo_combat{timesteps}')


    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_path)

    try:
        callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir='/models/', verbose=True)
        model.learn(timesteps, progress_bar=True, callback=callback)
        
        model_path = os.path.join('/models/', f'ppo_combat{timesteps}.zip')
        model.save(model_path)
    except:
        model_path = os.path.join('/models/', f'ppo_combat{timesteps}_partial.zip')
        model.save(model_path)

def main(action, num):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    retro.data.Integrations.add_custom_path(os.path.join(script_dir, 'custom_integrations'))

    render_mode = None
    if action == 'test' or action == 'evaluate':
        render_mode = 'human'
    
    env = retro.make(game='Zelda-NES', state='120w.state', inttype=retro.data.Integrations.CUSTOM_ONLY, render_mode=render_mode, record=action == 'record')

    env = Frameskip(env, 10, 20)

    try:
        if action == 'test':
            test_environment(env, num)

        elif action == 'train' or action == 'learn':
            train(env, num)

        elif action == 'evaluate' or 'record':
            if num:
                model_path = os.path.join('/models/', f'ppo_combat{num}.zip')
                model = PPO.load(model_path)
            evaluate(model, env)

        train(env, num)
    finally:
        env.close()


if __name__ == '__main__':
    # take in the number of iterations as the first parameter
    action = sys.argv[1]
    if len(sys.argv) > 2:
        timesteps = int(sys.argv[2])
    else:
        timesteps = None
    main(action, timesteps)
