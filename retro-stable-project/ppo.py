import sys
import retro
import os
import gymnasium as gym
from random import randint

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback

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
        model.learn(timesteps, progress_bar=True)
        
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

    #eval_callback = EvalCallback(env, best_model_save_path='/models/partial/', eval_freq=100000)

    try:
        if action == 'test':
            test_environment(env, num)

        elif action == 'train':
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
