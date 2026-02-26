#!/usr/bin/env python3
"""Evaluates the result of trainined models."""
from multiprocessing.sharedctypes import Synchronized
import sys
import os
import argparse
import shutil
from typing import Dict
from tqdm import tqdm
from triforce import ModelDefinition, make_zelda_env, Network, TrainingScenarioDefinition,  MetricTracker

def _print_stat_header(metrics: Dict[str, float]):
    terminal_width = shutil.get_terminal_size((80, 20)).columns
    header_result = f"{'Filename':<32} {'Steps':>9} "
    metric_columns = []

    for metric_name in metrics.keys():
        if len(header_result) + 12 > terminal_width:
            break

        metric_columns.append(metric_name)
        header_result += f"{metric_name[-12:]:>12} "

    print(header_result)
    return metric_columns

def _print_stat_row(filename, steps_trained, metrics: Dict[str, float], metric_columns):
    result = f"{filename:<32} {steps_trained:>9,} "
    for key in metric_columns:
        result += f"{metrics[key]:>12.2f} "

    print(result)

def evaluate_one_model(make_env, network, episodes, counter_or_callback) -> MetricTracker:
    """Runs a single scenario."""
    # pylint: disable=redefined-outer-name,too-many-locals
    env = make_env()
    try:
        for _ in range(episodes):
            obs, info = env.reset()

            terminated = truncated = False
            while not terminated and not truncated:
                action_mask = info.get('action_mask', None)
                action_mask = action_mask.unsqueeze(0) if action_mask is not None else None
                action = network.get_action(obs, action_mask)
                obs, _, terminated, truncated, info = env.step(action)

            if isinstance(counter_or_callback, Synchronized):
                with counter_or_callback.get_lock():
                    counter_or_callback.value += 1
            else:
                counter_or_callback()

        return MetricTracker.get_metrics_and_clear()
    finally:
        env.close()

def get_model_path(args):
    """Gets the model path."""
    return args.model_path[0] if args.model_path else \
                    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')

def main():
    """Main entry point."""
    # pylint: disable=too-many-locals
    args = parse_args()
    scenario_def = TrainingScenarioDefinition.get(args.scenario)
    def make_env():
        """Creates the environment."""
        model_def = ModelDefinition.get(args.model)
        render_mode = 'human' if args.render else None
        return make_zelda_env(scenario_def, model_def.action_space, render_mode=render_mode,
                              frame_stack=args.frame_stack)

    observation_space, action_space = None, None

    networks = []

    all_scenarios = create_scenarios(args)
    all_scenarios.reverse()
    total_episodes = sum(args.episodes for x in all_scenarios if x[-1])
    with tqdm(total=total_episodes) as progress:
        def update_progress():
            progress.update(1)

        for model_name, path, process in all_scenarios:
            if observation_space is None:
                observation_space, action_space = Network.load_spaces(path)

            model_def = ModelDefinition.get(model_name)
            network : Network = model_def.neural_net(observation_space, action_space)
            network.load(path)
            networks.append((network, path))
            if process:
                if metrics := evaluate_one_model(make_env, network, args.episodes, update_progress):
                    network.metrics = metrics
                    network.episodes_evaluated = args.episodes
                    network.save(path)

    if networks:
        columns = _print_stat_header(network.metrics)
        for network, path in networks:
            _print_stat_row(os.path.basename(path), network.steps_trained, network.metrics, columns)

def create_scenarios(args):
    """Finds all scenarios to be executed.  Also returns the results of any previous evaluations."""
    model_path = get_model_path(args)
    model_name = args.model

    all_scenarios = []

    process = True
    available_models = ModelDefinition.get(model_name).find_available_models(model_path)
    models_to_evaluate = sorted([int(x) for x in available_models.keys() if isinstance(x, int)])
    models_to_evaluate += [x for x in available_models.keys() if not isinstance(x, int)]
    for key in models_to_evaluate:
        path = available_models[key]
        _, episodes_evaluated = Network.load_metrics(path)
        process = args.reprocess or episodes_evaluated < args.episodes
        all_scenarios.append((model_name, path, process))

    if args.limit > 0:
        all_scenarios = all_scenarios[-args.limit:]

    return all_scenarios

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="evaluate - Evaluate Zelda ML models.")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity.")
    parser.add_argument("--ent-coef", type=float, default=0.001, help="Entropy coefficient for the PPO algorithm.")
    parser.add_argument("--frame-stack", type=int, default=3, help="Number of frames to stack in the observation.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to test.")
    parser.add_argument("--parallel", type=int, default=1, help="Use parallel environments to evaluate the models.")
    parser.add_argument("--render", action='store_true', help="Render the game while evaluating the models.")
    parser.add_argument("--limit", type=int, default=-1, help="Limit the number of models to evaluate.")

    parser.add_argument('model_path', nargs=1, help='The directory containing the models to evaluate')
    parser.add_argument('model', type=str, help='The model to evaluate.')
    parser.add_argument('scenario', type=str, help='The scenario to evaluate.')
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
