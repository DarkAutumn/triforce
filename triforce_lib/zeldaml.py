"""The overall entry point for the ML agent."""

from collections import Counter
import json
import os
from typing import List
import retro
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from .models_and_scenarios import ZeldaAIModel
from .objective_selector import ObjectiveSelector
from .zelda_wrapper import ZeldaGameWrapper
from .action_space import ZeldaActionSpace
from .zelda_observation_wrapper import FrameCaptureWrapper, ZeldaObservationWrapper
from .zelda_game_features import ZeldaGameFeatures
from .scenario_wrapper import ScenarioWrapper
from .models_and_scenarios import ZeldaScenario

class ZeldaML:
    """A class to train zelda models or to create environments."""
    def __init__(self, color, framestack = 1, **kwargs):
        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
            del kwargs['verbose']
        else:
            self.verbose = 0

        if 'device' in kwargs:
            self.device = kwargs['device']
            del kwargs['device']
        else:
            self.device = "auto"

        if 'ent_coef' in kwargs:
            self.ent_coef = kwargs['ent_coef']
            del kwargs['ent_coef']
        else:
            self.ent_coef = 0.0

        if 'obs_kind' in kwargs:
            self.obs_kind = kwargs['obs_kind']
            del kwargs['obs_kind']
        else:
            self.obs_kind = 'viewport'

        self.__extra_args = kwargs

        self.rgb_deque = None
        self.rgb_render = False
        if 'render_mode' in kwargs and kwargs['render_mode'] == 'rgb_array':
            self.rgb_render = True

        self.color = color
        self.framestack = framestack

    def make_env(self, scenario : ZeldaScenario, action_space = "all", parallel = 1):
        """Creates a Zelda retro environment for the given scenario."""

        def make_env_func():
            # create the environment
            env = retro.make(game='Zelda-NES', state=scenario.start[0], inttype=retro.data.Integrations.CUSTOM_ONLY,
                             **self.__extra_args)

            # Capture the raw observation frames into a deque.
            env = FrameCaptureWrapper(env, self.rgb_render)
            captured_frames = env.frames
            if self.rgb_render:
                self.rgb_deque = env.rgb_deque

            # Wrap the game to produce new info about game state and to hold the button down after the action is
            # taken to achieve the desired number of actions per second.
            env = ZeldaGameWrapper(env)

            # The AI orchestration piece.  This is responsible for selecting the model to use and the target
            # objective.
            env = ObjectiveSelector(env)

            # Frame stack and convert to grayscale if requested
            env = ZeldaObservationWrapper(env, captured_frames, not self.color, kind=self.obs_kind,
                                          framestack=self.framestack)

            # Reduce the action space to only the actions we want the model to take (no need for A+B for example,
            # since that doesn't make any sense in Zelda)
            env = ZeldaActionSpace(env, action_space)

            # Extract features from the game for the model, like whether link has beams or has keys and expose
            # these as observations.
            env = ZeldaGameFeatures(env)

            # Activate the scenario.  This is where rewards and end conditions are checked, using some of the new
            # info state provded by ZeldaGameWrapper above.
            env = ScenarioWrapper(env, scenario)

            return env

        if parallel and parallel > 1:
            env = make_vec_env(make_env_func, n_envs=parallel, vec_env_cls=SubprocVecEnv)
        else:
            env = make_env_func()

        return env

    def train(self, models : List[ZeldaAIModel], output_path = None, iteration_override = None, parallel = None):
        """Trains the given models."""

        if output_path is None:
            output_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            output_path = os.path.join(output_path, 'training')

        print("Writing to:", output_path)

        for zelda_ai_model in models:
            model_dir = os.path.join(output_path, zelda_ai_model.name)
            os.makedirs(model_dir, exist_ok=True)

            iterations = zelda_ai_model.iterations if iteration_override is None else iteration_override

            log_path = os.path.join(model_dir, 'logs')

            scenario = ZeldaScenario.get(zelda_ai_model.training_scenario)
            env = self.make_env(scenario, zelda_ai_model.action_space, parallel)
            try:
                print()
                print(f"Training model: {zelda_ai_model.name}")
                print(f"Scenario:       {zelda_ai_model.training_scenario}")
                print(f"Path:           {model_dir}")
                model = zelda_ai_model.create(env=env, verbose=self.verbose, tensorboard_log=log_path,
                                              ent_coef=self.ent_coef, device=self.device)
                callback = LogRewardCallback(model, model_dir)
                model.learn(iterations, progress_bar=True, callback=callback)
                model.save(os.path.join(model_dir, 'last.zip'))

            finally:
                env.close()

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

        self._rewards = {}
        self._endings = []
        self._evaluation = []
        self._success_rate = []

    def _on_step(self) -> bool:
        self._update_stats()

        if self.n_calls > self.next_save:
            self.next_save += self.model.n_steps

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

            self._rewards.clear()
            self._endings.clear()
            self._evaluation.clear()

        return True

    def _update_stats(self):
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

    def _save_best(self, score, reward, save_path):
        self.model.save(save_path)

        metadata = { "iterations" : self.num_timesteps, 'reward' : reward}
        if score is not None:
            metadata['score'] = score

        with open(save_path + '.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)

__all__ = ['ZeldaML']
