#! /usr/bin/python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import warnings
warnings.filterwarnings('ignore')

import argparse
from multiprocessing import Value, Pool
import multiprocessing
import pandas as pd
from tqdm import tqdm
from triforce_lib import ZeldaML, ZeldaModel, ZeldaScenario
import tensorflow as tf

def run_one_scenario(args, model_name, model_kind, zelda_ml=None):
    if zelda_ml is None:
        zelda_ml = create_zeldaml(args)

    model = ZeldaModel.get(model_name)
    loaded_model = model.get_model_by_kind(model_kind)

    ep_result = []

    env = zelda_ml.make_env(ZeldaScenario.get(model.training_scenario), model.action_space, 1)

    for ep in range(args.episodes):
        obs, info = env.reset()

        episode_rewards = 0
        episode_penalties = 0
        episode_total_reward = 0
        episode_score = 0
        success = False

        terminated = truncated = False

        while not terminated and not truncated:
            action, _ = loaded_model.predict(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_total_reward += reward

            if 'score' in info:
                episode_score = info['score']

            if 'rewards' in info:
                for kind, rew in info['rewards'].items():
                    rew_kind = kind.split('-', 1)[0]
                    if rew_kind == 'reward':
                        episode_rewards += abs(rew)
                    elif rew_kind == 'penalty':
                        episode_penalties -= abs(rew)
                    else:
                        raise ValueError(f"Unknown reward kind: {kind}")

            if 'end' in info:
                success = info['end'].startswith("success")

        ep_result.append((ep, success, episode_score, episode_total_reward, episode_rewards, episode_penalties))

        if args.verbose:
            print(f"Episode {ep}: {'Success' if success else 'Failure'} - Score: {episode_score} - Total Reward: {episode_total_reward} - Rewards: {episode_rewards} - Penalties: {episode_penalties}")

        global counter
        with counter.get_lock():
            counter.value += 1

    env.close()

    success_rate = round(100 * sum([1 for x in ep_result if x[1]]) / len(ep_result), 1)
    score = round(sum([x[2] for x in ep_result]) / len(ep_result), 1)
    total_reward = round(sum([x[3] for x in ep_result]) / len(ep_result), 1)
    rewards = round(sum([x[4] for x in ep_result]) / len(ep_result), 1)
    penalties = round(sum([x[5] for x in ep_result]) / len(ep_result), 1)

    return (model_name, model_kind, success_rate, score, total_reward, rewards, penalties)


def create_zeldaml(args):
    render_mode = 'human' if args.render else None
    model_path = get_model_path(args)

    zelda_ml = ZeldaML(args.color, args.frame_stack, render_mode=render_mode, verbose=args.verbose, ent_coef=args.ent_coef, device="cuda", obs_kind=args.obs_kind)
    zelda_ml.load_models(model_path)
    return zelda_ml

def get_model_path(args):
    model_path = args.model_path[0] if args.model_path else os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')
    return model_path

def init_pool(args):
    global counter
    counter = args

def main(args):
    multiprocessing.set_start_method('spawn')

    global counter
    counter = Value('i', 0)

    # loads models
    zelda_ml = create_zeldaml(args)

    all_scenarios = []
    for model in ZeldaModel.get_loaded_models():
        if not args.models or model.name in args.models:
            for i in range(len(model.models)):
                all_scenarios.append((args, model.name, model.model_kinds[i]))

    total_count = len(all_scenarios) * args.episodes

    if args.parallel > 1:
        with Pool(args.parallel, initializer=init_pool, initargs=(counter,)) as pool:
            result = pool.starmap_async(run_one_scenario, all_scenarios)

            with tqdm(total=total_count) as progress:
                while not result.ready():
                    result.wait(1)

                    with counter.get_lock():
                        progress.n = counter.value

                    progress.refresh()

            results = result.get()

    else:
        results = []
        for scenario in tqdm(all_scenarios, total=len(all_scenarios)):
            results.append(run_one_scenario(*scenario, zelda_ml=zelda_ml))

    columns = ['Model', 'Kind', 'Success%', 'Score', 'Total Reward', 'Rewards', 'Penalties']
    data_frame = pd.DataFrame(results, columns=columns)
    print(data_frame.to_string(index=False))
    data_frame.to_csv(os.path.join(get_model_path(args), 'evaluation.csv'), index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="ZeldaML - An ML agent to play The Legned of Zelda (NES).")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity.")
    parser.add_argument("--ent-coef", type=float, default=0.001, help="Entropy coefficient for the PPO algorithm.")
    parser.add_argument("--color", action='store_true', help="Give the model a color version of the game (instead of grayscale).")
    parser.add_argument("--obs-kind", choices=['gameplay', 'viewport', 'full'], default='viewport', help="The kind of observation to use.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to test.")
    parser.add_argument("--parallel", type=int, default=1, help="Use parallel environments to evaluate the models.")
    parser.add_argument("--render", action='store_true', help="Render the game while evaluating the models.")
    parser.add_argument("--frame-stack", type=int, default=1, help="Number of frames to stack together.")

    parser.add_argument('model_path', nargs=1, help='The director containing the models to evaluate')
    parser.add_argument('models', nargs='*', help='The director containing the models to evaluate')

    try:
        args = parser.parse_args()
        return args
    except Exception as e:
        print(e)
        parser.print_help()
        exit(0)

if __name__ == '__main__':
    args = parse_args()

    # if model path is actually a .csv that exists on disk, print that instead
    if args.model_path and args.model_path[0].endswith('.csv') and os.path.exists(args.model_path[0]):
        print(pd.read_csv(args.model_path[0]).to_string(index=False))
    else:
        main(args)
