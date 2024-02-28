# All code related to actual machine learning is contained in this file, hidden behind the "MachineLearningModel"
# class.  This makes it easier to swap in a brand new machine learning library or a completely different algorithm
# without having to change any other part of the code.

import os
import json
from collections import Counter

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from .models_and_scenarios import ZeldaModelDefinition
from .zelda_env import make_zelda_env

class ZeldaAI:
    """The raw implementation of the machine learning agent which plays the game."""
    def __init__(self, model_definition: ZeldaModelDefinition, device='auto', ent_coef=0.01, verbose=False):
        self.model_definition = model_definition
        self.device = device
        self.ent_coef = ent_coef
        self.verbose = verbose
        self._model = None

    def load(self, path) -> PPO:
        """Loads the model from the specified path."""
        self._model = PPO.load(path, device=self.device)
        return self._model

    def save(self, path):
        """Saves the model to the specified path."""
        self._model.save(path)

    @property
    def num_timesteps(self):
        """Returns the number of timesteps the model has been trained for."""
        return self._model.num_timesteps if self._model is not None else 0

    def predict(self, obs, deterministic = False):
        """Predicts the action to take based on the observation."""
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return action

    def train(self, output_path = None, iterations = None, parallel = None, *,
              grayscale = True, framestack = 1, obs_kind = 'viewport'):
        """
        Trains this model and saves the result to the output path.  The directory structure must be:
            output_path/{self.model_definition.name}/
                                                     model1.zip
                                                     model2.zip
                                                        ...
        """
        model = self.model_definition

        model_dir = os.path.join(output_path, model.name)
        os.makedirs(model_dir, exist_ok=True)

        iterations = model.iterations if iterations is None else iterations

        def make_env():
            return make_zelda_env(model.training_scenario, model.action_space, grayscale=grayscale,
                                  framestack=framestack, obs_kind=obs_kind)

        if parallel is not None and parallel > 1:
            env = make_vec_env(make_env, n_envs=parallel, vec_env_cls=SubprocVecEnv)
        else:
            env = make_env()

        try:
            print()
            print(f"Training model: {model.name}")
            print(f"Scenario:       {model.training_scenario.name}")
            print(f"Path:           {model_dir}")
            ml_model = self._create(env, tensorboard_log=os.path.join(model_dir, 'logs'))
            ml_model.learn(iterations, progress_bar=True, callback=LogRewardCallback(ml_model, model_dir))
            ml_model.save(os.path.join(model_dir, 'last.zip'))

        finally:
            env.close()

    def _create(self, env, **kwargs) -> PPO:
        return PPO('MultiInputPolicy', env, device=self.device, ent_coef=self.ent_coef, verbose=self.verbose, **kwargs)

class LogRewardCallback(BaseCallback):
    """A callback to log reward values to tensorboard and save the best models."""
    def __init__(self, model : PPO, save_dir : str, last_model_freq = 500_000):
        super().__init__()
        self.model = model
        self.next_save = model.n_steps
        self.last_model_freq = last_model_freq
        self.last_model_next_save = self.last_model_freq

        self.best_score = -np.inf
        self.best_reward = -np.inf

        self.save_dir = save_dir

        self._frames_per_iteration = []
        self._rewards = {}
        self._endings = []
        self._evaluation = []
        self._success_rate = []

    def _on_step(self) -> bool:
        self._update_stats()

        if self.n_calls > self.next_save:
            self.next_save += self.model.n_steps

            if self._frames_per_iteration:
                frame_mean = np.mean(self._frames_per_iteration)
                self.logger.record('rollout/frames-per-iteration', frame_mean)

            rew_mean = np.mean(list(self._rewards.values()))
            for kind, rew in self._rewards.items():
                split = kind.split('-', 1)
                name = f"{split[0]}/{split[1]}"
                self.logger.record(name, rew)

            ends = Counter(self._endings)
            for ending, count in ends.items():
                self.logger.record('end/' + ending, count)

            success_rate = np.mean(self._success_rate) if self._success_rate else 0.0
            self.logger.record('evaluation/success-rate', success_rate)

            score_mean = None
            if self._evaluation:
                score_mean = np.mean(self._evaluation)
                self.logger.record('evaluation/score', score_mean)

                if score_mean > self.best_score:
                    self.best_score = score_mean
                    self._save_best(score_mean, rew_mean, os.path.join(self.save_dir, 'best_score.zip'))

            if rew_mean > self.best_reward:
                self.best_reward = rew_mean
                self._save_best(score_mean, rew_mean, os.path.join(self.save_dir, 'best_reward.zip'))

            if self.model.num_timesteps >= self.last_model_next_save:
                self.last_model_next_save += self.last_model_freq
                self._save_best(score_mean, rew_mean, os.path.join(self.save_dir,
                                                                   f'model_{self.model.num_timesteps}.zip'))

            self._frames_per_iteration.clear()
            self._rewards.clear()
            self._endings.clear()
            self._evaluation.clear()

        return True

    def _update_stats(self):
        log_frames = self.n_calls % 10 == 1
        for info in self.locals['infos']:
            if 'rewards' in info:
                for kind, rew in info['rewards'].items():
                    self._rewards[kind] = rew + self._rewards.get(kind, 0)

            if 'end' in info:
                ending = info['end']
                self._endings.append(ending)
                if ending.startswith('success'):
                    self._success_rate.append(1)
                else:
                    self._success_rate.append(0)

            if 'final-score' in info:
                self._evaluation.append(info['final-score'])

            if log_frames and 'total_frames' in info:
                self._frames_per_iteration.append(info['total_frames'])

    def _save_best(self, score, reward, save_path):
        self.model.save(save_path)

        metadata = { "iterations" : self.num_timesteps, 'reward' : reward}
        if score is not None:
            metadata['score'] = score

        with open(save_path + '.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
