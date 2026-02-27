#!/usr/bin/env python3
"""Evaluates the result of trainined models."""
from collections import Counter
from math import ceil
from multiprocessing.sharedctypes import Synchronized
import sys
import os
import argparse
import json
import shutil
from typing import Dict, List
from tqdm import tqdm
from triforce import ModelDefinition, make_zelda_env, Network, TrainingScenarioDefinition,  MetricTracker


def write_progress_markdown(md_path, progress_values, max_progress, episodes, scenario_name, model_name=None):
    """Writes a progress report as a markdown file."""
    if not progress_values:
        return

    sorted_vals = sorted(progress_values)
    n = len(sorted_vals)
    success_count = sum(1 for v in sorted_vals if v >= max_progress)
    counts = Counter(sorted_vals)
    max_count = max(counts.values()) if counts else 1

    with open(md_path, 'w', encoding='utf-8') as f:
        title = f"Evaluation: {scenario_name}"
        if model_name:
            title += f" ({model_name})"
        f.write(f"# {title}\n\n")
        f.write(f"- **Episodes**: {episodes}\n")
        f.write(f"- **Success rate**: {success_count}/{n} ({100*success_count/n:.0f}%)"
                f" (reached milestone {max_progress})\n")
        f.write(f"- **Median progress**: {sorted_vals[n//2]}/{max_progress}\n")
        f.write(f"- **P25**: {sorted_vals[max(0, ceil(n*0.25)-1)]}  "
                f"**P50**: {sorted_vals[n//2]}  "
                f"**P75**: {sorted_vals[max(0, ceil(n*0.75)-1)]}  "
                f"**P90**: {sorted_vals[max(0, ceil(n*0.90)-1)]}\n\n")

        f.write("## Milestone Histogram\n\n")
        f.write("| Milestone | Count | Distribution |\n")
        f.write("|----------:|------:|:-------------|\n")
        for milestone in range(max_progress + 1):
            count = counts.get(milestone, 0)
            bar_len = round(count / max_count * 20) if max_count > 0 else 0
            bar_str = '█' * bar_len
            f.write(f"| {milestone} | {count} | {bar_str} |\n")


def convert_eval_json_to_md(json_path):
    """Converts a .eval.json file to a .eval.md file alongside it."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    md_path = json_path.rsplit('.json', 1)[0] + '.md'
    model_name = os.path.basename(json_path).rsplit('.eval.json', 1)[0]
    write_progress_markdown(
        md_path,
        data['progress_values'],
        data['max_progress'],
        data['episodes'],
        data['scenario'],
        model_name=model_name,
    )
    return md_path


def print_progress_report(progress_values : List[int], max_progress : int,
                          episodes : int, scenario_name : str):
    """Prints a progress-focused evaluation report with percentiles and histogram."""
    if not progress_values:
        print("No progress data collected.")
        return

    sorted_vals = sorted(progress_values)
    n = len(sorted_vals)
    success_count = sum(1 for v in sorted_vals if v >= max_progress)

    print(f"\n{'='*60}")
    print(f"  Evaluation: {episodes} episodes of {scenario_name}")
    print(f"{'='*60}")
    print(f"  Success rate: {success_count}/{n} ({100*success_count/n:.0f}%)"
          f"  (reached milestone {max_progress})")
    print(f"  Median progress: {sorted_vals[n//2]}/{max_progress}")
    print(f"  P25: {sorted_vals[max(0, ceil(n*0.25)-1)]:>3}  "
          f"P50: {sorted_vals[n//2]:>3}  "
          f"P75: {sorted_vals[max(0, ceil(n*0.75)-1)]:>3}  "
          f"P90: {sorted_vals[max(0, ceil(n*0.90)-1)]:>3}")
    print()

    # Histogram
    counts = Counter(sorted_vals)
    max_count = max(counts.values()) if counts else 1
    bar_width = 30

    print("  Milestone histogram:")
    for milestone in range(max_progress + 1):
        count = counts.get(milestone, 0)
        bar_len = round(count / max_count * bar_width) if max_count > 0 else 0
        bar_str = '█' * bar_len
        print(f"    {milestone:>3}: {count:>4}  {bar_str}")
    print(f"{'='*60}\n")

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

def evaluate_one_model(make_env, network, episodes, counter_or_callback):
    """Runs a single scenario.  Returns (metrics_dict, progress_metric_or_none)."""
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

        tracker = MetricTracker.get_instance()
        progress_metric = tracker.get_progress_metric() if tracker else None

        # Copy progress data before clearing
        progress_values = list(progress_metric.episode_values) if progress_metric else None
        max_progress = progress_metric.max_progress if progress_metric else 0

        metrics = MetricTracker.get_metrics_and_clear()
        return metrics, progress_values, max_progress
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
    total_episodes = sum(args.episodes for x in all_scenarios if x[-1])
    last_progress = None
    last_max_progress = 0
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
                result = evaluate_one_model(make_env, network, args.episodes, update_progress)
                metrics, progress_values, max_progress = result
                if metrics:
                    network.metrics = metrics
                    network.episodes_evaluated = args.episodes
                    network.save(path)

                if progress_values is not None:
                    last_progress = progress_values
                    last_max_progress = max_progress

                    # Save progress data as JSON sidecar
                    json_path = path.rsplit('.', 1)[0] + '.eval.json'
                    eval_data = {
                        'episodes': args.episodes,
                        'scenario': args.scenario,
                        'progress_values': progress_values,
                        'max_progress': max_progress,
                        'metrics': metrics,
                    }
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(eval_data, f, indent=2)

                    convert_eval_json_to_md(json_path)

    # Print progress report for the last evaluated model
    if last_progress is not None:
        print_progress_report(last_progress, last_max_progress, args.episodes, args.scenario)

    # Print the detailed metrics table
    if networks and args.verbose > 0:
        columns = _print_stat_header(network.metrics)
        for network, path in networks:
            _print_stat_row(os.path.basename(path), network.steps_trained, network.metrics, columns)

def create_scenarios(args):
    """Finds all scenarios to be executed.  Also returns the results of any previous evaluations.
    Step-count models are processed in descending order (most trained first).
    Non-step-count models are processed last."""
    model_path = get_model_path(args)
    model_name = args.model

    all_scenarios = []

    process = True
    available_models = ModelDefinition.get(model_name).find_available_models(model_path)

    # Step-count models in descending order (most steps first)
    step_count_keys = sorted([int(x) for x in available_models.keys() if isinstance(x, int)], reverse=True)
    # Non-step-count models last, any order
    other_keys = [x for x in available_models.keys() if not isinstance(x, int)]

    for key in list(step_count_keys) + other_keys:
        path = available_models[key]
        _, episodes_evaluated = Network.load_metrics(path)
        process = args.reprocess or episodes_evaluated < args.episodes
        all_scenarios.append((model_name, path, process))

    if args.limit > 0:
        all_scenarios = all_scenarios[:args.limit]

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
