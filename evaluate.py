#!/usr/bin/env python3
"""Evaluates the result of trainined models."""
from collections import Counter
from math import ceil
import sys
import os
import argparse
import json
import shutil
from typing import Dict, List
from scipy.stats import mannwhitneyu
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


def print_learning_curve(eval_dir, scenario_name=None):
    """Prints a learning curve table from all .eval.json files in a directory, showing progress over training steps."""
    # Collect all eval.json files with step counts
    entries = []
    for filename in os.listdir(eval_dir):
        if not filename.endswith('.eval.json'):
            continue

        # Extract step count from filename pattern: name_STEPS.eval.json
        base = filename.rsplit('.eval.json', 1)[0]
        parts = base.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            steps = int(parts[1])
        else:
            continue  # skip non-checkpoint files (e.g. final model)

        filepath = os.path.join(eval_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        progress = sorted(data['progress_values'])
        n = len(progress)
        max_prog = data['max_progress']
        success = sum(1 for v in progress if v >= max_prog)

        entries.append({
            'steps': steps,
            'median': progress[n // 2],
            'p75': progress[max(0, ceil(n * 0.75) - 1)],
            'p90': progress[max(0, ceil(n * 0.90) - 1)],
            'max': progress[-1],
            'success': f"{success}/{n}",
            'mean': sum(progress) / n,
            'max_progress': max_prog,
        })

    if not entries:
        print("No checkpoint evaluation data found.")
        return

    entries.sort(key=lambda e: e['steps'])
    max_prog = entries[0]['max_progress']

    title = f"Learning Curve: {scenario_name}" if scenario_name else "Learning Curve"
    print(f"\n{'='*72}")
    print(f"  {title}  (max milestone: {max_prog})")
    print(f"{'='*72}")
    print(f"  {'Steps':>10}  {'Mean':>6}  {'Median':>6}  {'P75':>5}  {'P90':>5}  {'Max':>5}  {'Success':>9}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*9}")

    for e in entries:
        print(f"  {e['steps']:>10,}  {e['mean']:>6.2f}  {e['median']:>6}  {e['p75']:>5}  "
              f"{e['p90']:>5}  {e['max']:>5}  {e['success']:>9}")
    print(f"{'='*72}\n")


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

def _percentile(sorted_vals, pct):
    """Returns the value at the given percentile from a sorted list."""
    return sorted_vals[max(0, ceil(len(sorted_vals) * pct) - 1)]


def compare_models(path_a, path_b):
    """Compares two evaluation results and prints a human-readable statistical comparison."""
    # pylint: disable=too-many-locals,too-many-statements
    with open(path_a, 'r', encoding='utf-8') as f:
        data_a = json.load(f)
    with open(path_b, 'r', encoding='utf-8') as f:
        data_b = json.load(f)

    name_a = os.path.basename(path_a).rsplit('.eval.json', 1)[0] or path_a
    name_b = os.path.basename(path_b).rsplit('.eval.json', 1)[0] or path_b

    vals_a = data_a['progress_values']
    vals_b = data_b['progress_values']
    max_a = data_a['max_progress']
    max_b = data_b['max_progress']
    sorted_a = sorted(vals_a)
    sorted_b = sorted(vals_b)

    # Summary stats
    mean_a, mean_b = sum(vals_a) / len(vals_a), sum(vals_b) / len(vals_b)
    median_a, median_b = sorted_a[len(sorted_a) // 2], sorted_b[len(sorted_b) // 2]
    success_a = sum(1 for v in vals_a if v >= max_a)
    success_b = sum(1 for v in vals_b if v >= max_b)

    # Mann-Whitney U test (two-sided)
    stat, p_value = mannwhitneyu(vals_a, vals_b, alternative='two-sided')

    # Stochastic dominance: P(A > B) from the U statistic
    n_a, n_b = len(vals_a), len(vals_b)
    prob_a_wins = stat / (n_a * n_b)

    # Determine winner
    if p_value < 0.05:
        winner = name_a if mean_a > mean_b else name_b
        verdict = f"{winner} is significantly better (p={p_value:.4f})"
    else:
        verdict = f"No significant difference (p={p_value:.4f})"

    # Print report
    col_w = max(len(name_a), len(name_b), 10)
    print(f"\n{'='*70}")
    print("  Model Comparison")
    print(f"{'='*70}")
    print(f"  {'':>{col_w}}   {'A: ' + name_a:>20}   {'B: ' + name_b:>20}")
    print(f"  {'Episodes':>{col_w}}   {len(vals_a):>20}   {len(vals_b):>20}")
    print(f"  {'Mean':>{col_w}}   {mean_a:>20.2f}   {mean_b:>20.2f}")
    print(f"  {'Median':>{col_w}}   {median_a:>20}   {median_b:>20}")
    print(f"  {'P25':>{col_w}}   {_percentile(sorted_a, 0.25):>20}   {_percentile(sorted_b, 0.25):>20}")
    print(f"  {'P75':>{col_w}}   {_percentile(sorted_a, 0.75):>20}   {_percentile(sorted_b, 0.75):>20}")
    print(f"  {'P90':>{col_w}}   {_percentile(sorted_a, 0.90):>20}   {_percentile(sorted_b, 0.90):>20}")
    print(f"  {'Success':>{col_w}}   "
          f"{success_a:>3}/{len(vals_a)} ({100*success_a/len(vals_a):.0f}%)          "
          f"{success_b:>3}/{len(vals_b)} ({100*success_b/len(vals_b):.0f}%)")
    print()

    # Side-by-side histogram
    max_progress = max(max_a, max_b)
    counts_a = Counter(vals_a)
    counts_b = Counter(vals_b)
    max_count = max(max(counts_a.values(), default=1), max(counts_b.values(), default=1))  # pylint: disable=nested-min-max
    bar_w = 15

    print("  Milestone histogram (A | B):")
    for milestone in range(max_progress + 1):
        ca = counts_a.get(milestone, 0)
        cb = counts_b.get(milestone, 0)
        bar_a = '█' * round(ca / max_count * bar_w) if max_count > 0 else ''
        bar_b = '█' * round(cb / max_count * bar_w) if max_count > 0 else ''
        print(f"    {milestone:>3}: {ca:>4} {bar_a:<{bar_w}} | {cb:>4} {bar_b:<{bar_w}}")
    print()

    # Statistical tests
    print("  Statistical Analysis:")
    print(f"    Mann-Whitney U statistic: {stat:.1f}")
    print(f"    p-value: {p_value:.6f}" + ("  (significant)" if p_value < 0.05 else "  (not significant)"))
    print(f"    P(A > B): {100*prob_a_wins:.1f}%   P(B > A): {100*(1-prob_a_wins):.1f}%")
    print()
    print(f"  Verdict: {verdict}")
    print(f"{'='*70}\n")


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

def evaluate_one_model(make_env, network, episodes, progress_callback):
    """Runs a single scenario.  Returns (metrics_dict, progress_values, max_progress)."""
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

            progress_callback()

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
    return args.model_path if args.model_path else \
                    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')

def main():
    """Main entry point."""
    # pylint: disable=too-many-locals
    args = parse_args()

    if args.compare:
        compare_models(args.compare[0], args.compare[1])
        return

    if args.summary:
        _print_summary(args)
        return

    if not args.model_path or not args.model or not args.scenario:
        print("Error: model_path, model, and scenario are required when not using --compare.")
        sys.exit(1)

    all_scenarios = create_scenarios(args)
    to_process = [(name, path) for name, path, process in all_scenarios if process]
    total_episodes = len(to_process) * args.episodes

    if not to_process:
        print("No models to evaluate.")
        return

    _run_sequential(args, to_process, total_episodes)

    # Print learning curve across all checkpoints
    model_dir = os.path.join(get_model_path(args), args.model)
    if os.path.isdir(model_dir):
        print_learning_curve(model_dir, args.scenario)

    # Print progress report for the best-trained model (first = most steps)
    if to_process:
        json_path = to_process[0][1].rsplit('.', 1)[0] + '.eval.json'
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print_progress_report(data['progress_values'], data['max_progress'],
                                  data['episodes'], args.scenario)


def _print_summary(args):
    """Prints the learning curve from existing .eval.json files without running evaluation."""
    model_path = get_model_path(args)
    model_dir = os.path.join(model_path, args.model) if args.model else model_path
    if not os.path.isdir(model_dir):
        print(f"Error: directory not found: {model_dir}")
        sys.exit(1)
    print_learning_curve(model_dir, args.scenario)


def _save_results(path, metrics, progress_values, max_progress, episodes, scenario, model_name):
    """Saves evaluation results: updates model .pt, writes .eval.json and .eval.md."""
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    if metrics:
        obs_space, act_space = Network.load_spaces(path)
        model_def = ModelDefinition.get(model_name)
        network = model_def.neural_net(obs_space, act_space)
        network.load(path)
        network.metrics = metrics
        network.episodes_evaluated = episodes
        network.save(path)

    if progress_values is not None:
        json_path = path.rsplit('.', 1)[0] + '.eval.json'
        eval_data = {
            'episodes': episodes,
            'scenario': scenario,
            'progress_values': progress_values,
            'max_progress': max_progress,
            'metrics': metrics,
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, indent=2)
        convert_eval_json_to_md(json_path)


def _run_sequential(args, to_process, total_episodes):
    """Evaluate models one at a time."""
    scenario_def = TrainingScenarioDefinition.get(args.scenario)

    def make_env():
        model_def = ModelDefinition.get(args.model)
        render_mode = 'human' if args.render else None
        return make_zelda_env(scenario_def, model_def.action_space, render_mode=render_mode,
                              frame_stack=args.frame_stack)

    observation_space, action_space = None, None
    with tqdm(total=total_episodes) as progress:
        def update_progress():
            progress.update(1)

        for model_name, path in to_process:
            if observation_space is None:
                observation_space, action_space = Network.load_spaces(path)

            model_def = ModelDefinition.get(model_name)
            network = model_def.neural_net(observation_space, action_space)
            network.load(path)

            metrics, progress_values, max_progress = evaluate_one_model(
                make_env, network, args.episodes, update_progress)
            _save_results(path, metrics, progress_values, max_progress,
                          args.episodes, args.scenario, model_name)


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

    # Filter to specific step counts if requested
    if args.steps:
        step_set = set(args.steps)
        step_count_keys = [k for k in step_count_keys if k in step_set]
        other_keys = []  # skip non-step-count models when filtering by steps

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
    parser.add_argument("--render", action='store_true', help="Render the game while evaluating the models.")
    parser.add_argument("--limit", type=int, default=-1, help="Limit the number of models to evaluate.")

    parser.add_argument('model_path', nargs='?', default=None, help='The directory containing the models to evaluate')
    parser.add_argument('model', type=str, nargs='?', default=None, help='The model to evaluate.')
    parser.add_argument('scenario', type=str, nargs='?', default=None, help='The scenario to evaluate.')
    parser.add_argument('--reprocess', action='store_true', help='Reprocess the models')
    parser.add_argument('--steps', type=int, nargs='+', default=None,
                        help='Only evaluate models with these step counts')
    parser.add_argument('--compare', type=str, nargs=2, metavar='EVAL_JSON',
                        help='Compare two .eval.json files instead of running evaluation')
    parser.add_argument('--summary', action='store_true',
                        help='Print the learning curve from existing .eval.json files without re-running')

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
