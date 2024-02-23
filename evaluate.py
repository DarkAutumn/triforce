#! /usr/bin/python
"""Evaluates the result of trainined models."""

import sys
import os
import argparse
from multiprocessing import Value, Pool
import multiprocessing
import pandas as pd
from tqdm import tqdm
from triforce import ZeldaEnv, ZeldaAIModel

# pylint: disable=global-statement,global-variable-undefined

def run_one_scenario(args, model_name, model_kind):
    """Runs a single scenario."""
    # pylint: disable=redefined-outer-name,too-many-locals

    zelda_ml = create_zeldaml(args)

    model_dir = get_model_path(args)
    models = ZeldaAIModel.initialize(model_dir)
    model = next(x for x in models if x.name == model_name)
    loaded_model = model.load(model_kind)

    ep_result = []

    env = zelda_ml.make_env(model.training_scenario, model.action_space, 1)

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
            obs, reward, terminated, truncated, info = env.step(action) # pylint: disable=unbalanced-tuple-unpacking
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
            # pylint: disable=line-too-long
            print(f"Episode {ep}: {'Success' if success else 'Failure'} - Score: {episode_score} - Total Reward: {episode_total_reward} - Rewards: {episode_rewards} - Penalties: {episode_penalties}")

        with COUNTER.get_lock():
            COUNTER.value += 1

    env.close()

    success_rate = round(100 * sum(1 for x in ep_result if x[1]) / len(ep_result), 1)
    score = round(sum(x[2] for x in ep_result) / len(ep_result), 1)
    total_reward = round(sum(x[3] for x in ep_result) / len(ep_result), 1)
    rewards = round(sum(x[4] for x in ep_result) / len(ep_result), 1)
    penalties = round(sum(x[5] for x in ep_result) / len(ep_result), 1)

    return (model_name, model_kind, success_rate, score, total_reward, rewards, penalties)


def create_zeldaml(args):
    """Creates a ZeldaML instance."""
    render_mode = 'human' if args.render else None
    zelda_ml = ZeldaEnv(args.color, args.frame_stack, render_mode=render_mode, verbose=args.verbose,
                       ent_coef=args.ent_coef, device="cuda", obs_kind=args.obs_kind)
    return zelda_ml

def get_model_path(args):
    """Gets the model path."""
    return args.model_path[0] if args.model_path else \
                    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')

def init_pool(args):
    """Initializes the pool."""
    global COUNTER
    COUNTER = args

def main():
    """Main entry point."""
    args = parse_args()

    # if model path is actually a .csv that exists on disk, print that instead
    if args.model_path and args.model_path[0].endswith('.csv') and os.path.exists(args.model_path[0]):
        print(pd.read_csv(args.model_path[0]).to_string(index=False))
        return

    multiprocessing.set_start_method('spawn')

    global COUNTER
    COUNTER = Value('i', 0)

    models = ZeldaAIModel.initialize(get_model_path(args))

    all_scenarios = []
    for model in models:
        if not args.models or model.name in args.models:
            # For inteveral saved models,  only evaluate the last 3
            available_models = sorted([int(x) for x in model.available_models.keys() if isinstance(x, int)])[-3:]
            available_models += [x for x in model.available_models.keys() if not isinstance(x, int)]
            for model_kind in available_models:
                all_scenarios.append((args, model.name, model_kind))

    total_count = len(all_scenarios) * args.episodes

    if args.parallel > 1:
        with Pool(args.parallel, initializer=init_pool, initargs=(COUNTER,)) as pool:
            result = pool.starmap_async(run_one_scenario, all_scenarios)

            with tqdm(total=total_count) as progress:
                while not result.ready():
                    result.wait(1)

                    with COUNTER.get_lock():
                        progress.n = COUNTER.value

                    progress.refresh()

            results = result.get()

    else:
        results = []
        for scenario in tqdm(all_scenarios, total=len(all_scenarios)):
            results.append(run_one_scenario(*scenario))

    columns = ['Model', 'Kind', 'Success%', 'Score', 'Total Reward', 'Rewards', 'Penalties']
    data_frame = pd.DataFrame(results, columns=columns)
    print(data_frame.to_string(index=False))
    data_frame.to_csv(os.path.join(get_model_path(args), 'evaluation.csv'), index=False)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ZeldaML - An ML agent to play The Legned of Zelda (NES).")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity.")
    parser.add_argument("--ent-coef", type=float, default=0.001, help="Entropy coefficient for the PPO algorithm.")
    parser.add_argument("--color", action='store_true',
                        help="Give the model a color version of the game (instead of grayscale).")
    parser.add_argument("--obs-kind", choices=['gameplay', 'viewport', 'full'], default='viewport',
                        help="The kind of observation to use.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to test.")
    parser.add_argument("--parallel", type=int, default=1, help="Use parallel environments to evaluate the models.")
    parser.add_argument("--render", action='store_true', help="Render the game while evaluating the models.")
    parser.add_argument("--frame-stack", type=int, default=1, help="Number of frames to stack together.")

    parser.add_argument('model_path', nargs=1, help='The director containing the models to evaluate')
    parser.add_argument('models', nargs='*', help='The director containing the models to evaluate')

    try:
        args = parser.parse_args()
        return args

    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(e)
        parser.print_help()
        sys.exit(0)

if __name__ == '__main__':
    main()
