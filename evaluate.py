#! /usr/bin/python
"""Evaluates the result of trainined models."""

from collections import Counter
import sys
import os
import json
import argparse
from multiprocessing import Value, Pool
import multiprocessing
from typing import Any, Dict, Optional
import pandas as pd
from tqdm import tqdm
from triforce import ZeldaModelDefinition, make_zelda_env, ZELDA_MODELS, ZeldaAI
from triforce.rewards import Penalty, Reward, StepRewards

# pylint: disable=global-statement,global-variable-undefined

def run_one_scenario(args, model_name, model_path):
    """Runs a single scenario."""
    # pylint: disable=redefined-outer-name,too-many-locals

    model = ZELDA_MODELS[model_name]
    ai = ZeldaAI(model, ent_coef=args.ent_coef, verbose=args.verbose)
    ai.load(model_path)
    print(f"{model_name} {ai.num_timesteps:,} timesteps")

    ep_result = []
    endings = []

    env = make_zelda_env_from_args(model, args)

    for ep in range(args.episodes):
        obs, info = env.reset()

        episode_rewards = 0
        episode_penalties = 0
        episode_total_reward = 0
        episode_score = 0
        success = False

        terminated = truncated = False

        while not terminated and not truncated:
            action = ai.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action) # pylint: disable=unbalanced-tuple-unpacking
            episode_total_reward += reward

            if 'rewards' in info:
                rewards : StepRewards = info['rewards']
                episode_score = rewards.score if rewards.score is not None else episode_score
                for outcome in rewards:
                    if isinstance(outcome, Penalty):
                        episode_penalties += outcome.value
                    elif isinstance(outcome, Reward):
                        episode_rewards += outcome.value

                end = rewards.ending
                success = end is not None and end.startswith("success")
                endings.append(end)

        episode_score = episode_score if success else None
        ep_result.append((ep, success, episode_score, episode_total_reward, episode_rewards, episode_penalties))

        if args.verbose:
            # pylint: disable=line-too-long
            print(f"Episode {ep}: {'Success' if success else 'Failure'} - Score: {episode_score} - Total Reward: {episode_total_reward} - Rewards: {episode_rewards} - Penalties: {episode_penalties}")

        with COUNTER.get_lock():
            COUNTER.value += 1

    env.close()

    scores = [x[2] for x in ep_result if x[2] is not None]
    score = round(sum(scores) / len(scores), 1) if scores else None

    success_rate = round(100 * sum(1 for x in ep_result if x[1]) / len(ep_result), 1)
    total_reward = round(sum(x[3] for x in ep_result) / len(ep_result), 1)
    rewards = round(sum(x[4] for x in ep_result) / len(ep_result), 1)
    penalties = round(sum(x[5] for x in ep_result) / len(ep_result), 1)

    endings = dict(Counter(endings))

    result = {
        'model': model_name,
        'model_path': model_path,
        'success_rate': success_rate,
        'score': score,
        'total_reward': total_reward,
        'rewards': rewards,
        'penalties': penalties,
        'endings' : endings,
        'episodes': args.episodes,
    }

    return result


def make_zelda_env_from_args(model : ZeldaModelDefinition, args):
    """Creates a ZeldaML instance."""
    render_mode = 'human' if args.render else None
    return make_zelda_env(model.training_scenario, model.action_space, render_mode=render_mode, obs_kind=args.obs_kind)

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

    results, all_scenarios = create_scenarios(args)

    if args.parallel > 1:
        with Pool(args.parallel, initializer=init_pool, initargs=(COUNTER,)) as pool:
            result = pool.starmap_async(run_one_scenario, all_scenarios)

            with tqdm(total=len(all_scenarios) * args.episodes) as progress:
                while not result.ready():
                    result.wait(1)

                    with COUNTER.get_lock():
                        progress.n = COUNTER.value

                    progress.refresh()

            for item in result.get():
                save_result(item)
                results.append(item)


    else:
        for scenario in tqdm(all_scenarios, total=len(all_scenarios)):
            result = run_one_scenario(*scenario)
            save_result(result)
            results.append(result)

    print_and_save(get_model_path(args), results)

def create_scenarios(args):
    """Finds all scenarios to be executed.  Also returns the results of any previous evaluations."""
    model_path = get_model_path(args)
    models = args.models if args.models else ZELDA_MODELS.keys()

    results = []
    all_scenarios = []
    for model_name in models:
        if not args.models or model_name in args.models:
            available_models = ZELDA_MODELS[model_name].find_available_models(model_path)
            models_to_evaluate = sorted([int(x) for x in available_models.keys() if isinstance(x, int)])
            models_to_evaluate += [x for x in available_models.keys() if not isinstance(x, int)]
            for key in models_to_evaluate:
                path = available_models[key]

                result : Optional[Dict[str, Any]] = None
                if os.path.exists(path + '.evaluation.json'):
                    with open(path + '.evaluation.json', 'r', encoding="utf8") as f:
                        result = json.load(f)
                        results.append(result)

                if result and 'episodes' in result and result['episodes'] >= args.episodes:
                    results.append(result)
                else:
                    all_scenarios.append((args, model_name, path))

    if args.limit > 0:
        all_scenarios = all_scenarios[-args.limit:]

    return results, all_scenarios

def save_result(result):
    """Saves the result of an evaluation."""
    with open(result['model_path'] + '.evaluation.json', 'w', encoding="utf8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

def print_and_save(model_path, results):
    """Prints the result and saves it to evaluation.csv."""

    results = [x.copy() for x in results]
    for result in results:
        del result['endings']
    results = [tuple(x.values()) for x in results]

    columns = ['Model', 'Kind', 'Success%', 'Score', 'Total Reward', 'Rewards', 'Penalties', 'Episodes']
    data_frame = pd.DataFrame(results, columns=columns)
    print(data_frame.to_string(index=False))
    data_frame.to_csv(os.path.join(model_path, 'evaluation.csv'), index=False)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="evaluate - Evaluate Zelda ML models.")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity.")
    parser.add_argument("--ent-coef", type=float, default=0.001, help="Entropy coefficient for the PPO algorithm.")
    parser.add_argument("--obs-kind", choices=['gameplay', 'viewport', 'full'], default='viewport',
                        help="The kind of observation to use.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to test.")
    parser.add_argument("--parallel", type=int, default=1, help="Use parallel environments to evaluate the models.")
    parser.add_argument("--render", action='store_true', help="Render the game while evaluating the models.")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit the number of models to evaluate.")

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
