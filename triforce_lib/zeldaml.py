import os
import retro
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor

from .scenario import ZeldaScenario

class ZeldaML:
    """The model and algorithm used to train the agent"""
    def __init__(self, model_dir, scenario, algorithm, frame_stack, color, **kwargs):
        """
        arguments:
            model_dir -- the directory to save the model
            algorithm -- the algorithm to use (ppo, a2c, etc)
            color -- whether to use color or not (False = grayscale)
            frame_stack -- number of frames to stack in the observation
            kwargs -- additional arguments to pass to the environment creation, such as render_mode, etc
        """
        algorithm = algorithm.lower()
        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
            del kwargs['verbose']
        else:
            self.verbose = 0


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

        os.makedirs(self.model_base_dir, exist_ok=True)

        # create the environment
        env = retro.make(game='Zelda-NES', state=self.scenario.start_state, inttype=retro.data.Integrations.CUSTOM_ONLY, **kwargs)

#        if self.frame_stack > 1:
#            env = VecFrameStack(env, n_stack=self.frame_stack)

#        if not self.color:
#            env = GrayscaleObservation(env)
        
        env = self.scenario.activate(env)
        self.env = Monitor(env, self.log_dir)
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

        callback = SaveBestModelCallback(check_freq=4096, save_func=self.save, log_dir=self.model_base_dir, verbose=self.verbose) if save_best else None
        self.model.learn(iterations, progress_bar=progress_bar, callback=callback)
        self.save()

    def load(self, path=None, best = True):
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
            self.model = PPO.load(path, self.env, verbose=self.verbose)
        elif self.algorithm == 'a2c':
            self.model = A2C.load(path, self.env, verbose=self.verbose)
        else:
            raise Exception(f'Unsupported algorithm: {self.algorithm}')

        return True
    
    def save(self, path = None, best = False):
        if not path:
            path = self.best_file if best else self.model_file
        
        self.model.save(path)

    def _create_model(self):
        tensorboard_log=self.log_dir

        if self.algorithm == 'ppo':
            return PPO('CnnPolicy', self.env, verbose=self.verbose, tensorboard_log=tensorboard_log)
        
        elif self.algorithm == 'a2c':
            return A2C('CnnPolicy', self.env, verbose=self.verbose, tensorboard_log=tensorboard_log)
        
        raise Exception(f'Unsupported algorithm: {self.algorithm}')
    

class SaveBestModelCallback(BaseCallback):
    def __init__(self, check_freq: int, save_func, log_dir: str, verbose=0):
        super(SaveBestModelCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf
        self.save_func = save_func

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(f"New best mean reward: {mean_reward:.2f} (last: {self.best_mean_reward:.2f})")

                if mean_reward > self.best_mean_reward:
                    if self.verbose > 0:
                        print("Saving new best model.")

                    self.best_mean_reward = mean_reward
                    self.save_func(best=True)

        return True

class GrayscaleObservation(gym.ObservationWrapper):
    """Converts the observation to grayscale to make processing easier"""
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        grayscale_obs = np.dot(observation[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        return grayscale_obs

__all__ = ['ZeldaML']