#! /usr/bin/python
"""Evaluates the result of trainined models."""
from multiprocessing.sharedctypes import Synchronized
import sys
import os
import argparse
from tqdm import tqdm
from triforce import ZeldaModelDefinition, make_zelda_env, ZELDA_MODELS, Network
from triforce.rewards import TotalRewards, RewardStats

# pylint: disable=global-statement,global-variable-undefined

def _print_stat_header():
    print(f"{'Model':<20} {'Filename':<25} {'Steps':>9} {'Total Reward':>13} {'Score':>10} {'Duration':>9} "
          f"{'Success Rate':>12}")

def _print_stat_row(model_name, filename, steps_trained, stats : RewardStats):
    print(f"{model_name:<20} {filename:<25} {steps_trained:9,} {stats.rewards:13.1f} {stats.scores:10.1f} "
          f"{int(stats.total_steps / 60.1):9} {stats.success_rate * 100:11.1f}%")

def run_one_scenario(args, model_name, model_path, counter_or_callback):
    """Runs a single scenario."""
    # pylint: disable=redefined-outer-name,too-many-locals
    model_def = ZELDA_MODELS[model_name]
    env = make_zelda_env_from_args(model_def, args)
    network : Network = model_def.neural_net(env.observation_space, env.action_space)
    network.load(model_path)

    total = TotalRewards()
    for _ in range(args.episodes):
        obs, _ = env.reset()

        terminated = truncated = False
        while not terminated and not truncated:
            action = network.get_action(obs)
            obs, _, terminated, truncated, info = env.step(action) # pylint: disable=unbalanced-tuple-unpacking

        total.add(info['episode_rewards'])
        if isinstance(counter_or_callback, Synchronized):
            with counter_or_callback.get_lock():
                counter_or_callback.value += 1
        else:
            counter_or_callback()

    env.close()

    stats = total.stats
    network.stats = stats
    network.stats.evaluated = True
    network.save(model_path)

def make_zelda_env_from_args(model : ZeldaModelDefinition, args):
    """Creates a ZeldaML instance."""
    render_mode = 'human' if args.render else None
    return make_zelda_env(model.training_scenario, model.action_space, render_mode=render_mode, obs_kind=args.obs_kind)

def get_model_path(args):
    """Gets the model path."""
    return args.model_path[0] if args.model_path else \
                    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')

def main():
    """Main entry point."""
    args = parse_args()

    all_scenarios = create_scenarios(args)
    total_episodes = sum(args.episodes for x in all_scenarios if x[-1])
    with tqdm(total=total_episodes) as progress:
        def update_progress():
            progress.update(1)

        for args, model_name, path, process in all_scenarios:
            if process:
                run_one_scenario(args, model_name, path, update_progress)

    _print_stat_header()
    env = None
    for args, model_name, path, _ in all_scenarios:
        model_def = ZELDA_MODELS[model_name]
        if env is None:
            env = make_zelda_env_from_args(model_def, args)

        network = model_def.neural_net(env.observation_space, env.action_space)
        network.load(path)
        filename = os.path.basename(path)
        _print_stat_row(model_name, filename, network.steps_trained, network.stats)

    if env:
        env.close()

def create_scenarios(args):
    """Finds all scenarios to be executed.  Also returns the results of any previous evaluations."""
    model_path = get_model_path(args)
    models = args.models if args.models else ZELDA_MODELS.keys()

    all_scenarios = []
    for model_name in models:
        if not args.models or model_name in args.models:
            process = True
            available_models = ZELDA_MODELS[model_name].find_available_models(model_path)
            models_to_evaluate = sorted([int(x) for x in available_models.keys() if isinstance(x, int)])
            models_to_evaluate += [x for x in available_models.keys() if not isinstance(x, int)]
            for key in models_to_evaluate:
                path = available_models[key]
                stats = Network.load_stats(path)
                evaluated = stats.evaluated if stats and hasattr(stats, 'evaluated') else False
                if not args.reprocess and evaluated and stats.episodes >= args.episodes:
                    process = False

                all_scenarios.append((args, model_name, path, process))

    if args.limit > 0:
        all_scenarios = all_scenarios[-args.limit:]

    return all_scenarios

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
    parser.add_argument('--reprocess', action='store_true', help='Reprocess the models')

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
