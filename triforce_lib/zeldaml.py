from collections import Counter
import datetime
import json
import os
import retro
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from .models import ZeldaModel
from .objective_selector import ObjectiveSelector
from .zelda_wrapper import ZeldaGameWrapper
from .action_space import ZeldaActionSpace
from .zelda_observation_wrapper import FrameCaptureWrapper, ZeldaObservationWrapper
from .zelda_game_features import ZeldaGameFeatures
from .scenario import ZeldaScenario

class ZeldaML:
    def __init__(self, frame_stack, color, **kwargs):
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

        self.rgb_render = False
        if 'render_mode' in kwargs and kwargs['render_mode'] == 'rgb_array':
            self.rgb_render = True

        self.color = color
        self.frame_stack = 1 if not isinstance(frame_stack, int) or frame_stack < 2 else frame_stack

    def make_env(self, scenario, action_space = "all", parallel = 1):
        def make_env_func():
            # create the environment
            env = retro.make(game='Zelda-NES', state=scenario.all_start_states[0], inttype=retro.data.Integrations.CUSTOM_ONLY, **self.__extra_args)

            # Capture the raw observation frames into a deque.  Since we are skipping frames and not acting on every frame, we need to save
            # the last 'frame_stack' frames so that we can give the model a sense of motion without it being affected by the skipped frames.
            env = FrameCaptureWrapper(env, self.rgb_render)
            captured_frames = env.frames
            if self.rgb_render:
                self.rgb_deque = env.rgb_deque

            # Wrap the game to produce new info about game state and to hold the button down after the action is taken to achieve the desired
            # number of actions per second.
            env = ZeldaGameWrapper(env)

            # The AI orchestration piece.  This is responsible for selecting the model to use and the target
            # objective.
            env = ObjectiveSelector(env)
            orchestrator = env
            
            # Frame stack and convert to grayscale if requested
            env = ZeldaObservationWrapper(env, captured_frames, not self.color, kind=self.obs_kind)

            # Reduce the action space to only the actions we want the model to take (no need for A+B for example,
            # since that doesn't make any sense in Zelda)
            env = ZeldaActionSpace(env, action_space)

            # extract features from the game for the model, like whether link has beams or has keys and expose these as observations
            env = ZeldaGameFeatures(env)

            # Activate the scenario.  This is where rewards and end conditions are checked, using some of the new info state provded
            # by ZeldaGameWrapper above.
            env = scenario.activate(env)

            return env

        if parallel and parallel > 1:
            env = make_vec_env(make_env_func, n_envs=parallel, vec_env_cls=SubprocVecEnv)
        else:
            env = make_env_func()

        return env
        
    def train(self, output_path = None, model_names = None, iteration_override = None, parallel = None, progress_bar = True):
        if model_names is None:
            models = ZeldaModel.get_model_info()
        else:
            models = [x for x in ZeldaModel.get_model_info() if x.name in model_names]
            if len(models) != len(model_names):
                raise Exception(f'Could not find all models requested: {model_names} missing: {set(model_names) - set([x.name for x in models])}')

        if output_path is None:
            output_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            output_path = os.path.join(output_path, 'training')

        print("Writing to:", output_path)

        for model_info in models:
            model_dir = os.path.join(output_path, model_info.name)
            os.makedirs(model_dir, exist_ok=True)

            iterations = model_info.iterations if iteration_override is None else iteration_override

            model_path = os.path.join(model_dir, 'final.zip')
            log_path = os.path.join(model_dir, 'logs')

            scenario = ZeldaScenario.get(model_info.training_scenario)
            env = self.make_env(scenario, model_info.action_space, parallel)
            try:
                print()
                print(f"Training model: {model_info.name}")
                print(f"Scenario:       {model_info.training_scenario}")
                print(f"Path:           {model_path}")
                model = self._create_model(env, log_path)
                callback = LogRewardCallback(model.save, model_dir)
                model.learn(iterations, progress_bar=progress_bar, callback=callback)
                model.save(model_path)

            finally:
                env.close()

    def load_models(self, path):
        return ZeldaModel.load_models(path, verbose=self.verbose, ent_coef=self.ent_coef, device=self.device)

    def _create_model(self, env, log_dir):
        return PPO('MultiInputPolicy', env, verbose=self.verbose, tensorboard_log=log_dir, ent_coef=self.ent_coef, device=self.device)
    

class LogRewardCallback(BaseCallback):
    def __init__(self, save_model, save_dir : str, save_freq : int = 4096):
        super(LogRewardCallback, self).__init__()
        self.log_reward_freq = save_freq

        self.best_score = -np.inf
        self.best_reward = -np.inf

        self.save_dir = save_dir
        self.save_model = save_model

        self._rewards = {}
        self._endings = []
        self._evaluation = []

    def _on_step(self) -> bool:
        infos = self.locals['infos']
        for info in infos:
            if 'rewards' in info:
                for kind, rew in info['rewards'].items():
                    self._rewards[kind] = rew + self._rewards.get(kind, 0)
            
            if 'end' in info:
                self._endings.append(info['end'])

            if 'final-score' in info:
                self._evaluation.append(info['final-score'])

        if self.n_calls % self.log_reward_freq == 0:
            # rewards and ends tend to be pretty wild at the beginning of training, so only log them after a certain threshold
            if self.n_calls >= 2048:
                rew_mean = np.mean(list(self._rewards.values()))
                for kind, rew in self._rewards.items():
                    split = kind.split('-', 1)
                    name = f"{split[0]}/{split[1]}"
                    self.logger.record(name, rew)

                ends = Counter(self._endings)
                for ending, count in ends.items():
                    self.logger.record('end/' + ending, count)

                score_mean = None
                if self._evaluation:
                    score_mean = np.mean(self._evaluation)
                    self.logger.record('evaluation/score', score_mean)

                    if score_mean > self.best_score:
                        self.best_score = score_mean
                        self.save_best(score_mean, rew_mean, os.path.join(self.save_dir, 'best_score.zip'))
                
                if rew_mean > self.best_reward:
                    self.best_reward = rew_mean
                    self.save_best(score_mean, rew_mean, os.path.join(self.save_dir, 'best_reward.zip'))

            self._rewards.clear()
            self._endings.clear()
            self._evaluation.clear()

        return True

    def save_best(self, score, reward, save_path):
        self.save_model(save_path)

        metadata = { "iterations" : self.num_timesteps, 'reward' : reward}
        if score is not None:
            metadata['score'] = score

        with open(save_path + '.json', 'w') as f:
            json.dump(metadata, f, indent = 4)

__all__ = ['ZeldaML']