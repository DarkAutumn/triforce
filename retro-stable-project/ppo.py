import sys
import numpy as np
import retro
import os

from frameskip import Frameskip
from save_best import SaveOnBestTrainingRewardCallback

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

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

def get_model_name(algorithm, policy, scenario, iterations):
    return f'{algorithm}_{policy}_{scenario}_{iterations}.zip'

def main(action, num, load=None):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    retro.data.Integrations.add_custom_path(os.path.join(script_dir, 'custom_integrations'))

    log_dir = '/output/'
    model_dir = '/models/'
    model_name = get_model_name('ppo', 'cnn', 'gauntlet', timesteps)
    model_path = os.path.join(model_dir, model_name)
    best_dir = os.path.join(model_dir, 'best')
    best_path = os.path.join(best_dir, model_name)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    render_mode = None
    if action == 'test' or action == 'evaluate':
        render_mode = 'human'
    
    env = retro.make(game='Zelda-NES', state='120w.state', inttype=retro.data.Integrations.CUSTOM_ONLY, render_mode=render_mode, record=action == 'record')

    try:
        # We only take action every so many frames, not every single frame.
        env = Frameskip(env, 10, 20)

        if action == 'test':
            test_environment(env, num)

        else:
            env = Monitor(env, log_dir)
            model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_dir)

            if action == 'train' or action == 'learn':
                callback = SaveOnBestTrainingRewardCallback(check_freq=1000, save_path=best_path, log_dir=log_dir, verbose=True)
                model.learn(timesteps, progress_bar=True, callback=callback)
                model.save(model_path)

            elif action == 'evaluate' or action == 'record':
                if num:
                    model = PPO.load(load)
                            
                mean_reward, std_reward = evaluate_policy(model, env, render=True, n_eval_episodes=num)
                print(f'Mean reward: {mean_reward} +/- {std_reward}')

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
