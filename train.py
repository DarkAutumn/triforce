#!/usr/bin/env python3
"""Train models to play The Legend of Zelda (NES)."""

# pylint: disable=duplicate-code

import argparse
import sys
import os
import faulthandler
import traceback

from tqdm import tqdm
from triforce import ModelDefinition, TrainingScenarioDefinition, make_zelda_env
from triforce.ml_ppo import PPO
from triforce.models import Network
from triforce.scenario_wrapper import TrainingCircuitDefinition, TrainingCircuitEntry

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


def _get_kwargs_from_args(args, model_def):
    kwargs = {}
    if not args.high_lr:
        kwargs['dynamic_lr'] = True

    if args.load is not None:
        obs, act = Network.load_spaces(args.load)
        network = model_def.neural_net(obs, act)
        network.load(args.load)
        kwargs['model'] = network

    if args.frame_stack is not None:
        kwargs['frame_stack'] = args.frame_stack

    if args.render_mode:
        kwargs['render_mode'] = args.render_mode

    if args.ent_coef is not None:
        kwargs['ent_coeff'] = args.ent_coef

    if args.device is not None:
        kwargs['device'] = args.device

    circuit_def = TrainingCircuitDefinition.get(args.scenario)
    if circuit_def is None:
        circuit = [TrainingCircuitEntry(scenario=args.scenario)]
    else:
        circuit = circuit_def.scenarios

    return kwargs, circuit

def train_once(ppo : PPO, scenario_def, model_def, save_path, iterations, **kwargs):
    """Trains a model with the given scenario.  Returns (model, iterations_used)."""
    def create_env():
        return make_zelda_env(scenario_def, model_def.action_space, **kwargs)

    steps_before = kwargs.get('model', None)
    steps_before = steps_before.steps_trained if steps_before else 0
    model = ppo.train(model_def.neural_net, create_env, iterations, tqdm(ncols=100), save_path=save_path, **kwargs)
    model_name = model_def.name.replace(' ', '_')
    model.save(f"{save_path}/{model_name}-{scenario_def.name}.pt")
    return model, model.steps_trained - steps_before

def main():
    """Main entry point."""
    args = parse_args()

    if args.hook_exceptions:
        faulthandler.enable()
        sys.excepthook = _dump_trace_with_locals

    model_name = args.model
    model_def = ModelDefinition.get(model_name)
    if model_def is None:
        raise ValueError(f"Unknown model: {model_name}")

    model_directory = os.path.join(args.output if args.output else 'training/', model_name)
    log_dir = os.path.join(model_directory, "logs")

    os.makedirs(model_directory, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    kwargs, circuit = _get_kwargs_from_args(args, model_def)
    ppo = PPO(log_dir, **kwargs)

    total_budget = args.iterations
    iterations_spent = 0

    for scenario_entry in circuit:
        scenario_def = TrainingScenarioDefinition.get(scenario_entry.scenario)
        if scenario_def is None:
            raise ValueError(f"Unknown scenario: {scenario_entry.scenario}")

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
            print(f"Skipping {scenario_def.name}: no iteration budget remaining.")
            break

        if scenario_entry.exit_criteria:
            assert scenario_entry.threshold is not None, "Threshold must be set if exit criteria is set"
            kwargs['exit_criteria'] = scenario_entry.exit_criteria
            kwargs['exit_threshold'] = scenario_entry.threshold

        elif 'exit_criteria' in kwargs:
            del kwargs['exit_criteria']
            del kwargs['exit_threshold']

        kwargs['model_name'] = model_def.name + '-' + scenario_def.name
        if scenario_entry.exit_criteria:
            criteria = f" or {scenario_entry.exit_criteria} >= {scenario_entry.threshold}"
        else:
            criteria = ""

        print(f"Training {model_def.name} on {scenario_def.name} for up to {iterations:,} iterations{criteria}.")
        model, used = train_once(ppo, scenario_def, model_def, model_directory, iterations, **kwargs)
        kwargs['model'] = model
        iterations_spent += used

    model.save(f"{model_directory}/{model_name}.pt")

    if args.evaluate:
        _run_post_training_eval(model, model_def, scenario_def, args.evaluate, **kwargs)


def _run_post_training_eval(model, model_def, scenario_def, episodes, **kwargs):
    """Runs evaluation episodes after training and prints a progress report."""
    # pylint: disable=import-outside-toplevel
    from evaluate import evaluate_one_model, print_progress_report

    print(f"\nRunning {episodes} evaluation episodes...")

    def create_eval_env():
        return make_zelda_env(scenario_def, model_def.action_space, **kwargs)

    with tqdm(total=episodes) as progress:
        def update():
            progress.update(1)
        _, progress_values, max_progress = evaluate_one_model(
            create_eval_env, model, episodes, update)

    if progress_values is not None:
        print_progress_report(progress_values, max_progress, episodes, scenario_def.name)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="train - Train Zelda ML models")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity.")
    parser.add_argument("--ent-coef", type=float, default=None, help="Entropy coefficient for the PPO algorithm.")
    parser.add_argument("--high-lr", action='store_true', default=None, help="Use a fixed high learning rate.")
    parser.add_argument("--frame-stack", type=int, default=None, help="The number of frames to stack.")
    parser.add_argument("--device", choices=['cpu', 'cuda'], default=None, help="The device to use.")
    parser.add_argument("--render-mode", type=str, default=None, help="The render mode to use.")

    parser.add_argument('model', type=str, help='The model to train.')
    parser.add_argument('scenario', type=str, help='The scenario to train on.')
    parser.add_argument("--output", type=str, help="Location to write to.")
    parser.add_argument("--iterations", type=int, default=None, help="Override iteration count.")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel environments to run.")
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
