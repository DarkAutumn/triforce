from collections import Counter
import os
import retro
import numpy as np

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from .zelda_wrapper import ZeldaGameWrapper
from .action_space import ZeldaAttackOnlyActionSpace
from .zelda_observation_wrapper import FrameCaptureWrapper, ZeldaObservationWrapper
from .zelda_game_features import ZeldaGameFeatures
from .scenario import ZeldaScenario

class ZeldaML:
    """The model and algorithm used to train the agent"""
    def __init__(self, model_dir, scenario, frame_stack, color, parallel, **kwargs):
        """
        arguments:
            model_dir -- the directory to save the model
            color -- whether to use color or not (False = grayscale)
            frame_stack -- number of frames to stack in the observation
            kwargs -- additional arguments to pass to the environment creation, such as render_mode, etc
        """
        algorithm = "ppo"
        
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

        if not isinstance(frame_stack, int) or frame_stack < 2:
            frame_stack = 1

        if isinstance(scenario, str):
            scenario = ZeldaScenario.get(scenario)
        elif not isinstance(scenario, ZeldaScenario):
            raise Exception('scenario must be a ZeldaScenario or the name of a well-known scenario')
        
        if 'debug_scenario' in kwargs:
            scenario.debug(kwargs['debug_scenario'])
            del kwargs['debug_scenario']

        self.algorithm = algorithm
        self.color = color
        self.frame_stack = frame_stack
        self.scenario = scenario

        # set up directories
        self.model_base_dir = os.path.join(model_dir, scenario.name, f"{algorithm.lower()}_{frame_stack}x_{'color' if color else 'grayscale'}")
        self.model_file = os.path.join(self.model_base_dir, 'model.zip')
        self.best_file = os.path.join(self.model_base_dir, 'best.zip')
        self.log_dir = os.path.join(self.model_base_dir, 'logs')
        self.log_file = os.path.join(self.log_dir, 'monitor.csv')

        os.makedirs(self.model_base_dir, exist_ok=True)

        def make_env():
            # create the environment
            env = retro.make(game='Zelda-NES', state=self.scenario.all_start_states[0], inttype=retro.data.Integrations.CUSTOM_ONLY, **kwargs)

            # Capture the raw observation frames into a deque.  Since we are skipping frames and not acting on every frame, we need to save
            # the last 'frame_stack' frames so that we can give the model a sense of motion without it being affected by the skipped frames.
            env = FrameCaptureWrapper(env)
            captured_frames = env.frames

            # Wrap the game to produce new info about game state and to hold the button down after the action is taken to achieve the desired
            # number of actions per second.
            env = ZeldaGameWrapper(env)
            
            # Frame stack and convert to grayscale if requested
            env = ZeldaObservationWrapper(env, captured_frames, self.frame_stack, not self.color, gameplay_only=True)
            
            # Reduce the action space to only the actions we want the model to take (no need for A+B for example,
            # since that doesn't make any sense in Zelda)
            env = ZeldaAttackOnlyActionSpace(env)

            # extract features from the game for the model, like whether link has beams or has keys and expose these as observations
            env = ZeldaGameFeatures(env)

            # Activate the scenario.  This is where rewards and end conditions are checked, using some of the new info state provded
            # by ZeldaGameWrapper above.
            env = self.scenario.activate(env)

            return env

        if parallel and parallel > 1:
            env = make_vec_env(make_env, n_envs=parallel, vec_env_cls=SubprocVecEnv)
        else:
            env = make_env()
        
        self.env = env
        self.model = None

    def close(self):
        self.env.close()

    def evaluate(self, episodes, **kwargs):
        return evaluate_policy(self.model, self.env, n_eval_episodes=episodes, **kwargs)
        
    def learn(self, iterations, progress_bar = True, save_best = True):
        if not iterations:
            raise Exception('Must specify number of iterations to learn')
        
        if not self.model:
            self.model = self._create_model()

        callback = LogRewardCallback(save_freq=4096, best_check_freq=2048, zeldaml=self) if save_best else None
        self.model.learn(iterations, progress_bar=progress_bar, callback=callback)
        self.save(self.model_file)

    def load(self, path=None, best = None):
        if path and best:
            raise Exception('Cannot specify both path and best')
        
        if not path:
            if best and os.path.exists(self.best_file):
                path = self.best_file
            elif os.path.exists(self.model_file):
                path = self.model_file

        if not path or not os.path.exists(path):
            return False
        
        if self.algorithm == 'ppo':
            self.model = PPO.load(path, self.env, verbose=self.verbose, tensorboard_log=self.log_dir, ent_coef=self.ent_coef, device=self.device)
        elif self.algorithm == 'a2c':
            self.model = A2C.load(path, self.env, verbose=self.verbose, tensorboard_log=self.log_dir, ent_coef=self.ent_coef, device=self.device)
        else:
            raise Exception(f'Unsupported algorithm: {self.algorithm}')

        return True
    
    def save(self, path):
        if not path:
            raise Exception('Must specify path to save model to')
        
        self.model.save(path)

    def _create_model(self):
        tensorboard_log=self.log_dir

        if self.algorithm == 'ppo':
            return PPO('MultiInputPolicy', self.env, verbose=self.verbose, tensorboard_log=tensorboard_log, ent_coef=self.ent_coef, device=self.device)
        
        raise Exception(f'Unsupported algorithm: {self.algorithm}')
    

class LogRewardCallback(BaseCallback):
    def __init__(self, save_freq : int, best_check_freq: int, zeldaml : ZeldaML):
        super(LogRewardCallback, self).__init__(zeldaml.verbose)
        self.log_reward_freq = save_freq
        self.zeldaml = zeldaml
        self.best_mean_reward = -np.inf
        self.best_timestamp = -np.inf

        self._rewards = {}
        self._endings = []

    def _on_step(self) -> bool:
        infos = self.locals['infos']
        for info in infos:
            if 'rewards' in info:
                for kind, rew in info['rewards'].items():
                    self._rewards[kind] = rew + self._rewards.get(kind, 0)
            
            if 'end' in info:
                self._endings.append(info['end'])

        if self.n_calls % self.log_reward_freq == 0:
            # Save the model as save_dir/model_{iterations}.zip
            path = os.path.join(self.zeldaml.model_base_dir, f"model_{self.num_timesteps}.zip")
            self.zeldaml.save(path)

            # rewards and ends tend to be pretty wild at the beginning of training, so only log them after a certain threshold
            if self.n_calls >= 2048:
                for kind, rew in self._rewards.items():
                    self.logger.record('reward/' + kind, rew)

                ends = Counter(self._endings)
                for ending, count in ends.items():
                    self.logger.record('end/' + ending, count)

            self._rewards.clear()
            self._endings.clear()

        return True

__all__ = ['ZeldaML']