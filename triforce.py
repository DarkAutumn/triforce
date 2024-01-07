import sys
import numpy as np
import os

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from triforce_lib import *

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

def parse_args():
    parser = argparse.ArgumentParser(description="Parse command line arguments for training, testing, evaluating, or recording.")

    parser.add_argument("action", choices=['train', 'test', 'evaluate', 'record'], help="Action to perform: train, test, evaluate, or record.")
    parser.add_argument("scenario", choices=['gauntlet'], help="The scenario to run: guantlet.")
    parser.add_argument("iterations", type=int, help="Number of iterations to run.")

    parser.add_argument("--algorithm", choices=['ppo', 'a2c', 'dqn', 'sac', 'td3', 'ddpg'], help="The algorithm to use (ppo, a2c, dqn, sac, td3, ddpg).")
    parser.add_argument("--scenario", help="The scenario to use (e.g., gauntlet).")
    parser.add_argument("--load", help="Loads the given model.")
    parser.add_argument("--render", action='store_true', help="Render the environment.")

    return parser.parse_args()


def main():
    args = parse_args()

    # load scenario and create directories
    scenario = load_scenario(args.scenario)
    
    log_dir = '/output/'
    model_dir = '/models/'
    model_name = scenario.get_model_name(args.iterations)
    model_path = os.path.join(model_dir, model_name)
    best_dir = os.path.join(model_dir, 'best')
    best_path = os.path.join(best_dir, model_name)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    # create the environment
    render_mode = None
    render_mode = 'human' if args.action == 'test' or args.action == 'evaluate' or args.render else None
    record = args.action == 'record'

    env = scenario.create_env(render_mode=render_mode, record=record)

    # run the scenario
    try:

        if args.action == 'test':
            test_environment(env, args.iterations)

        else:
            env = Monitor(env, log_dir)
            
            if args.load:
                model = scenario.load_model(args.load)
            else:
                model = scenario.create_model(env, verbose=1, tensorboard_log=log_dir)

            if args.action == 'train' or args.action == 'learn':
                callback = SaveBestModelCallback(check_freq=1000, save_path=best_path, log_dir=log_dir, verbose=True)
                model.learn(args.iterations, progress_bar=True, callback=callback)
                model.save(model_path)

            elif args.action == 'evaluate' or args.action == 'record':
                if not args.load:
                    raise Exception('Must specify model to evaluate or record.')
                
                mean_reward, std_reward = evaluate_policy(model, env, render=True, n_eval_episodes=args.iterations)
                print(f'Mean reward: {mean_reward} +/- {std_reward}')

    finally:
        env.close()

import argparse

if __name__ == '__main__':
    main()
