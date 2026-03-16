#!/usr/bin/env python3
"""Train models to play The Legend of Zelda (NES)."""

# pylint: disable=duplicate-code

import argparse
import sys
import os
import time
import faulthandler
import traceback

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text
from rich.table import Table
from torch.utils.tensorboard import SummaryWriter

from triforce import (ActionSpaceDefinition, ModelKindDefinition, TrainingScenarioDefinition,
                      TrainingCallback, make_zelda_env)
from triforce.ml_ppo import PPO
from triforce.models import Network
from triforce.scenario_wrapper import TrainingCircuitDefinition, TrainingCircuitEntry

BAR_WIDTH = 50
BAR_FILL = "▄"
BAR_EMPTY = " "


class TrainingDisplay(TrainingCallback):
    """Rich-based TUI for training progress and metrics, with tensorboard logging."""

    def __init__(self, live, log_dir):
        self._live = live
        self._log_dir = log_dir
        self._tensorboard = None

        # Circuit-level state
        self._scenarios = []        # list of (name, iterations)
        self._name_width = 0
        self._active_index = -1
        self._scenario_steps = {}   # scenario_name -> steps completed
        self._scenario_total = {}   # scenario_name -> total steps
        self._completed = set()     # scenario names that are done
        self._total_budget = 0
        self._total_spent = 0

        # Current scenario state
        self._current_steps = 0
        self._current_total = 0

        # Metrics state
        self._game_metrics = {}
        self._optimize_stats = {}

    def on_circuit_start(self, scenarios):
        self._scenarios = scenarios
        self._name_width = max(len(name) for name, _ in scenarios)
        self._name_width = max(self._name_width, len("Total"))
        self._total_budget = sum(iters for _, iters in scenarios)
        self._total_spent = 0
        for name, iters in scenarios:
            self._scenario_steps[name] = 0
            self._scenario_total[name] = iters
        self._refresh()

    def on_scenario_start(self, scenario_name, iterations):
        self._active_index = next(
            i for i, (name, _) in enumerate(self._scenarios) if name == scenario_name)
        self._current_steps = 0
        self._current_total = iterations
        self._scenario_total[scenario_name] = iterations
        self._game_metrics = {}
        self._optimize_stats = {}

        # Set up tensorboard for this scenario
        if self._tensorboard:
            self._tensorboard.close()
        scenario_log_dir = os.path.join(self._log_dir, scenario_name)
        os.makedirs(scenario_log_dir, exist_ok=True)
        self._tensorboard = SummaryWriter(scenario_log_dir)
        self._refresh()

    def on_scenario_end(self, scenario_name):
        self._scenario_steps[scenario_name] = self._current_steps
        self._total_spent += self._current_steps
        self._completed.add(scenario_name)
        self._refresh()

    def on_progress(self, steps, total_steps):
        self._current_steps += steps
        self._current_total = total_steps
        if self._active_index >= 0:
            name = self._scenarios[self._active_index][0]
            self._scenario_steps[name] = self._current_steps
            self._scenario_total[name] = total_steps
        self._refresh()

    def on_metrics(self, metrics, iteration, total_iterations):
        self._game_metrics = dict(metrics)
        if self._tensorboard:
            timestamp = time.time()
            for name, value in metrics.items():
                if '/' not in name:
                    name = f"metrics/{name}"
                self._tensorboard.add_scalar(name, value, iteration, timestamp)
            self._tensorboard.flush()
        self._refresh()

    def on_optimize(self, stats, iteration, total_iterations):
        self._optimize_stats = dict(stats)
        if self._tensorboard:
            for name, value in stats.items():
                self._tensorboard.add_scalar(name, value, iteration)
            self._tensorboard.flush()
        self._refresh()

    def on_training_complete(self):
        if self._tensorboard:
            self._tensorboard.close()
            self._tensorboard = None
        self._refresh()

    def _refresh(self):
        self._live.update(self._render())

    def _render(self):
        parts = []
        parts.append(Text(""))  # blank line

        for i, (name, _) in enumerate(self._scenarios):
            if name in self._completed:
                line = Text("  ✔ ", style="green")
                line.append(name, style="green")
                parts.append(line)
            elif i == self._active_index:
                steps = self._scenario_steps.get(name, 0)
                total = self._scenario_total.get(name, 1)
                parts.append(self._render_bar(name, steps, total, active=True))
            else:
                total = self._scenario_total.get(name, 0)
                parts.append(self._render_bar(name, 0, total, active=False))

        # Total progress bar
        total_done = self._total_spent + (self._current_steps if self._active_index >= 0 else 0)
        parts.append(Text(""))
        parts.append(self._render_bar("Total", total_done, self._total_budget, active=True))

        # Metrics table
        if self._game_metrics or self._optimize_stats:
            parts.append(Text(""))
            parts.append(self._render_metrics())

        return Group(*parts)

    def _render_bar(self, name, steps, total, active):
        padded = name.ljust(self._name_width)
        pct = (steps / total * 100) if total > 0 else 0
        filled = int(BAR_WIDTH * min(steps, total) / total) if total > 0 else 0
        bar_str = BAR_FILL * filled + BAR_EMPTY * (BAR_WIDTH - filled)

        line = Text(f"  {padded}  [")
        line.append(bar_str[:filled], style="green")
        line.append(bar_str[filled:])
        line.append(f"] {pct:5.1f}%")
        if active and total > 0:
            line.append(f"  ({steps:,} / {total:,})")
        return line

    def _render_metrics(self):
        table = Table(show_header=True, show_edge=False, pad_edge=False, box=None)
        table.add_column("Metric", style="cyan", min_width=24)
        table.add_column("Value", justify="right", min_width=12)

        # Key game metrics first
        key_game = ["success-rate", "rewards", "room-progress", "progress/median"]
        for key in key_game:
            if key in self._game_metrics:
                table.add_row(key, f"{self._game_metrics[key]:.4f}")

        # Key optimization stats
        key_stats = [
            ("charts/SPS", "SPS", ",.0f"),
            ("losses/policy_loss", "policy loss", ".4f"),
            ("losses/value_loss", "value loss", ".4f"),
            ("losses/entropy", "entropy", ".4f"),
            ("losses/approx_kl", "approx KL", ".4f"),
            ("losses/explained_variance", "explained var", ".4f"),
        ]
        for stat_key, display_name, fmt in key_stats:
            if stat_key in self._optimize_stats:
                table.add_row(display_name, f"{self._optimize_stats[stat_key]:{fmt}}")

        return table

def _dump_trace_with_locals(exc_type, exc_value, exc_traceback):
    with open("crash_log.txt", "w", encoding="utf8") as f:
        f.write(f"Unhandled exception: {exc_type.__name__}: {exc_value}\n\n")

        for frame, lineno in traceback.walk_tb(exc_traceback):
            f.write(f"File: {frame.f_code.co_filename}, Line: {lineno}, Function: {frame.f_code.co_name}\n")
            f.write("Locals:\n")
            for var_name, var_value in frame.f_locals.items():
                typename = type(var_value).__name__
                f.write(f"  {typename} {var_name}: {var_value}\n")
            f.write("\n")


def _next_counter(base_dir):
    """Find the next counter directory (0, 1, 2, ...) under base_dir."""
    if not os.path.exists(base_dir):
        return 0
    existing = [int(d) for d in os.listdir(base_dir) if d.isdigit() and os.path.isdir(os.path.join(base_dir, d))]
    return max(existing, default=-1) + 1


def _model_stem(model_kind_name, action_space_name):
    """Returns the base filename stem: {model-kind}_{action-space}."""
    return f"{model_kind_name}_{action_space_name}"


def _get_kwargs_from_args(args, model_kind, action_space_def):
    kwargs = {}
    if args.load is not None:
        obs, act = Network.load_spaces(args.load)
        network = model_kind.network_class(obs, act,
                                           model_kind=model_kind.name,
                                           action_space_name=action_space_def.name)
        network.load(args.load)
        kwargs['model'] = network

        # Infer obs_kind and frame_stack from saved model when not explicitly set
        if args.obs_kind is None and args.frame_stack is None:
            from triforce.observation_wrapper import infer_obs_kind  # pylint: disable=import-outside-toplevel
            inferred_kind, inferred_stack = infer_obs_kind(obs)
            kwargs['obs_kind'] = inferred_kind
            kwargs['frame_stack'] = inferred_stack

    if args.frame_stack is not None:
        kwargs['frame_stack'] = args.frame_stack

    # Auto-select obs_kind for impala models (they require full-rgb)
    if 'obs_kind' not in kwargs:
        obs_kind = args.obs_kind
        if obs_kind is None and model_kind.name.startswith('impala'):
            obs_kind = 'full-rgb'
        if obs_kind is not None:
            kwargs['obs_kind'] = obs_kind

    # impala models use frame_stack=1 by default with full-rgb
    if kwargs.get('obs_kind') == 'full-rgb' and 'frame_stack' not in kwargs:
        kwargs['frame_stack'] = 1

    if args.render_mode:
        kwargs['render_mode'] = args.render_mode

    if args.ent_coef is not None:
        kwargs['ent_coeff'] = args.ent_coef

    if args.device is not None:
        kwargs['device'] = args.device

    if args.parallel > 1:
        kwargs['envs'] = args.parallel

    circuit_def = TrainingCircuitDefinition.get(args.scenario)
    if circuit_def is None:
        circuit = [TrainingCircuitEntry(scenario=args.scenario)]
    else:
        circuit = circuit_def.scenarios

    return kwargs, circuit

def train_once(ppo, scenario_def, model_kind, action_space_def, checkpoint_dir, iterations,
               callback=None, **kwargs):
    """Trains a model with the given scenario.  Returns (model, iterations_used)."""
    multihead = getattr(model_kind.network_class, 'is_multihead', False)
    kwargs['multihead'] = multihead

    def create_env():
        return make_zelda_env(scenario_def, action_space_def.actions, **kwargs)

    steps_before = kwargs.get('model', None)
    steps_before = steps_before.steps_trained if steps_before else 0

    # Pass env-creation info for multi-env subprocess spawning
    kwargs['scenario_def'] = scenario_def
    kwargs['action_space_name'] = action_space_def.actions

    # Pass metadata for checkpoint naming
    stem = _model_stem(model_kind.name, action_space_def.name)
    kwargs['model_name'] = stem
    kwargs['model_kind'] = model_kind.name
    kwargs['action_space_name_str'] = action_space_def.name

    model = ppo.train(model_kind.network_class, create_env, iterations, callback,
                      save_path=checkpoint_dir, **kwargs)

    # Save leg checkpoint with scenario name
    model.save(f"{checkpoint_dir}/{stem}_{scenario_def.name}_{model.steps_trained}.pt")
    return model, model.steps_trained - steps_before

def _run_circuit(ppo, circuit, model_kind, action_space_def, checkpoint_dir, kwargs, total_budget,
                 callback=None):
    """Run training circuit and return (final_model, final_scenario_def)."""
    iterations_spent = 0
    model = None
    scenario_def = None

    # Resolve iteration counts for all scenarios upfront so the display can show them
    scenario_plan = []
    for scenario_entry in circuit:
        sdef = TrainingScenarioDefinition.get(scenario_entry.scenario)
        if sdef is None:
            raise ValueError(f"Unknown scenario: {scenario_entry.scenario}")

        if scenario_entry.iterations is not None:
            iters = scenario_entry.iterations
        elif total_budget is not None:
            iters = total_budget
        else:
            iters = sdef.iterations
        scenario_plan.append((sdef.name, iters))

    if callback:
        callback.on_circuit_start(scenario_plan)

    for scenario_entry in circuit:
        scenario_def = TrainingScenarioDefinition.get(scenario_entry.scenario)

        if scenario_entry.iterations is not None:
            iterations = scenario_entry.iterations
        elif total_budget is not None:
            iterations = total_budget - iterations_spent
        else:
            iterations = scenario_def.iterations

        # Cap to remaining budget if a total budget was specified
        if total_budget is not None:
            iterations = min(iterations, total_budget - iterations_spent)

        if iterations <= 0:
            break

        if scenario_entry.exit_criteria:
            assert scenario_entry.threshold is not None, "Threshold must be set if exit criteria is set"
            kwargs['exit_criteria'] = scenario_entry.exit_criteria
            kwargs['exit_threshold'] = scenario_entry.threshold
        elif 'exit_criteria' in kwargs:
            del kwargs['exit_criteria']
            del kwargs['exit_threshold']

        if callback:
            callback.on_scenario_start(scenario_def.name, iterations)

        model, used = train_once(ppo, scenario_def, model_kind, action_space_def,
                                 checkpoint_dir, iterations, callback, **kwargs)

        if callback:
            callback.on_scenario_end(scenario_def.name)

        kwargs['model'] = model
        iterations_spent += used

    return model, scenario_def


def main():
    """Main entry point."""
    args = parse_args()

    if args.hook_exceptions:
        faulthandler.enable()
        sys.excepthook = _dump_trace_with_locals

    # Resolve model kind and action space (use defaults if not specified)
    action_space_def = ActionSpaceDefinition.get(args.action_space) if args.action_space \
        else ActionSpaceDefinition.get_default()
    model_kind = ModelKindDefinition.get(args.model_kind) if args.model_kind \
        else ModelKindDefinition.get_default()

    # Build directory structure: output/scenario/counter/
    output_base = args.output or 'training'
    scenario_dir = os.path.join(output_base, args.scenario)
    counter = _next_counter(scenario_dir)
    run_dir = os.path.join(scenario_dir, str(counter))
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    console = Console()
    console.print(f"Output: {run_dir}")
    console.print(f"Model kind: {model_kind.name}, Action space: {action_space_def.name}")

    kwargs, circuit = _get_kwargs_from_args(args, model_kind, action_space_def)
    ppo = PPO(**kwargs)

    with Live(console=console, refresh_per_second=4) as live:
        display = TrainingDisplay(live, log_dir)
        model, scenario_def = _run_circuit(ppo, circuit, model_kind, action_space_def,
                                           checkpoint_dir, kwargs, args.iterations,
                                           callback=display)
        display.on_training_complete()

    # Save final result in the run directory (not checkpoints)
    stem = _model_stem(model_kind.name, action_space_def.name)
    final_path = f"{run_dir}/{stem}.pt"
    model.save(final_path)
    console.print(f"\nFinal model: {final_path}")

    if args.evaluate:
        _run_post_training_eval(model, action_space_def, model_kind, scenario_def,
                                args.evaluate, console, **kwargs)


def _run_post_training_eval(model, action_space_def, model_kind, scenario_def, episodes,
                            console=None, **kwargs):
    """Runs evaluation episodes after training and prints a progress report."""
    # pylint: disable=import-outside-toplevel
    from rich.progress import Progress
    from evaluate import evaluate_one_model, print_progress_report

    if console is None:
        console = Console()
    console.print(f"\nRunning {episodes} evaluation episodes...")

    multihead = getattr(model_kind.network_class, 'is_multihead', False)
    kwargs['multihead'] = multihead

    def create_eval_env():
        return make_zelda_env(scenario_def, action_space_def.actions, **kwargs)

    with Progress(console=console) as progress:
        task = progress.add_task("Evaluating...", total=episodes)
        def update():
            progress.advance(task)
        _, progress_values, max_progress = evaluate_one_model(
            create_eval_env, model, episodes, update)

    if progress_values is not None:
        print_progress_report(progress_values, max_progress, episodes, scenario_def.name)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="train - Train Zelda ML models")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity.")
    parser.add_argument("--ent-coef", type=float, default=None, help="Entropy coefficient for the PPO algorithm.")
    parser.add_argument("--frame-stack", type=int, default=None, help="The number of frames to stack.")
    parser.add_argument("--device", choices=['cpu', 'cuda'], default=None, help="The device to use.")
    parser.add_argument("--render-mode", type=str, default=None, help="The render mode to use.")
    parser.add_argument("--obs-kind", type=str, default=None,
                        choices=['viewport', 'gameplay', 'full-rgb'],
                        help="Observation kind. Auto-selected for impala models if not set.")

    parser.add_argument('scenario', type=str, help='The scenario or circuit to train on.')
    parser.add_argument('action_space', type=str, nargs='?', default=None,
                        help='Action space name (default: from triforce.yaml).')
    parser.add_argument('model_kind', type=str, nargs='?', default=None,
                        help='Model kind name (default: from triforce.yaml).')
    parser.add_argument("--output", type=str, help="Location to write to.")
    parser.add_argument("--iterations", type=int, default=None, help="Override iteration count.")
    parser.add_argument("--parallel", type=int, default=6, help="Number of parallel environments to run.")
    parser.add_argument("--load", type=str, help="Load a model to continue training.")
    parser.add_argument("--evaluate", type=int, default=None, metavar="N",
                        help="Run N evaluation episodes after training and print a progress report.")
    parser.add_argument("--hook-exceptions", action='store_true', help="Dump tracebacks on unhandled exceptions.")

    try:
        args = parser.parse_args()
        return args

    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(e)
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
