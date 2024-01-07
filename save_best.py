import json
import os
import numpy as np

from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path, log_dir: str, verbose=1, model_info=None):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.model_info = model_info

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

                if mean_reward > self.best_mean_reward:
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))

                    self.best_mean_reward = mean_reward
                    self.model.save(self.save_path)


                    if self.model_info:
                        info = dict(info)
                    else:
                        info = {}

                    info['timesteps'] = self.num_timesteps
                    info['mean_reward'] = mean_reward
                    
                    with open(self.save_path + '.json', 'w') as f:
                        f.write(json.dumps(info, indent=4))

        return True
