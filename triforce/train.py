"""Train one specialist with PPO (design §7.1).

    python -m triforce.train --specialist d1-melee [--load CKPT] [--parallel N] [--iterations N]
        [--steps-per-update N] [--device cpu|cuda] [--run-id ID] [--models-dir DIR]

Each iteration collects ``--steps-per-update`` env steps (split across ``--parallel`` subprocess workers, one
emulator each; ``--parallel 1`` runs the env in-process for debugging), then runs one PPO update.  Output goes
to ``<models>/<specialist>/<run-id>/``: ``ckpt.pt`` (see ``registry``), ``metrics.jsonl``, ``tb/``
(TensorBoard) and ``run.json`` (arguments and configs).

Weights come from ``--load`` (a checkpoint file or run dir) or else the specialist's ``warm_start`` (a
checkpoint path or another specialist's name, meaning its promoted checkpoint).  Loading a checkpoint of the
same specialist also restores the optimizer state and the iteration/step counters; a different specialist's
checkpoint only supplies weights.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from . import registry
from .demo import collect_demo_batch, parse_demo_actions
from .metrics import MetricsLogger, format_summary
from .model import ModelConfig, ZeldaPolicy
from .observation import ObservationBuilder
from .ppo import PPO, IterationResult, PPOConfig
from .ppo_worker import ModelFactory, ZeldaEnvFactory
from .specialist import load_specialist


def parse_args(argv=None):
    """Command line."""
    parser = argparse.ArgumentParser(prog='python -m triforce.train', description=__doc__.split('\n\n')[0])
    parser.add_argument('--specialist', required=True, help="specialists/<name>.yaml")
    parser.add_argument('--load', default=None, help="checkpoint file or run dir to start from")
    parser.add_argument('--parallel', type=int, default=1, help="env workers (1 = in-process)")
    parser.add_argument('--iterations', type=int, default=100, help="PPO iterations (rollout + update)")
    parser.add_argument('--steps-per-update', type=int, default=PPOConfig.target_steps)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--models-dir', default=None, help=f"default ${registry.MODELS_DIR_ENV} or ./models")
    parser.add_argument('--save-every', type=int, default=10, help="checkpoint every N iterations (and at the end)")
    parser.add_argument('--learning-rate', type=float, default=PPOConfig.learning_rate)
    parser.add_argument('--grid-encoder', default='conv', choices=['conv', 'linear'],
                        help="new models only; a loaded checkpoint keeps its architecture")
    parser.add_argument('--start-state', default=None, help="pin every episode to this savestate")
    parser.add_argument('--capture-states', action='store_true',
                        help="let envs write first-entry savestates into the integration dir (off by default: "
                             "parallel workers would race on the same files)")
    parser.add_argument('--demo', default=None, help="demo trace (Action.parse lines) for demo regularization")
    parser.add_argument('--demo-bc-coeff', type=float, default=0.1)
    parser.add_argument('--demo-allow-incomplete', action='store_true',
                        help="accept a demo that stops before the episode ends in success")
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args(argv)


def resolve_warm_start(value: str, root) -> Path:
    """A warm_start entry is a checkpoint path/run dir, or a specialist name (its promoted checkpoint)."""
    if os.path.exists(value):
        return Path(value)
    return registry.promoted_checkpoint(value, root)


def initial_model(args, specialist: str, root):
    """(network, optimizer_state, start_iteration, start_steps, source)."""
    config = load_specialist(specialist)
    source = args.load or (str(resolve_warm_start(config.warm_start, root)) if config.warm_start else None)
    if source is None:
        network = ZeldaPolicy(ObservationBuilder.observation_space(), ModelConfig(grid_encoder=args.grid_encoder))
        return network, None, 0, 0, None

    network, checkpoint = registry.load_model(source)
    if args.load and checkpoint['specialist'] == specialist:
        return network, checkpoint.get('optimizer'), checkpoint['iterations'], checkpoint['steps'], source
    return network, None, 0, 0, source


def main(argv=None, session=None) -> Path:
    """Runs training; returns the run directory.  ``session`` (an existing GameSession) is used by the
    in-process env when ``--parallel 1`` (tests share one emulator per process)."""
    # pylint: disable=too-many-locals, too-many-statements
    args = parse_args(argv)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if args.parallel < 1:
        raise SystemExit("--parallel must be >= 1")
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")

    specialist = load_specialist(args.specialist).name
    root = registry.models_root(args.models_dir)
    run_id = args.run_id or registry.new_run_id()
    out_dir = registry.run_dir(specialist, run_id, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    network, optimizer_state, start_iteration, start_steps, source = initial_model(args, specialist, root)
    env_factory = ZeldaEnvFactory(specialist, start_state=args.start_state,
                                  **({} if args.capture_states else {'capture_dir': None}))
    in_process_env = env_factory(session) if args.parallel == 1 else None

    demo_batch = None
    if args.demo:
        demo_env = in_process_env or env_factory()
        try:
            demo_batch = collect_demo_batch(demo_env, parse_demo_actions(args.demo),
                                            require_success=not args.demo_allow_incomplete)
        finally:
            if demo_env is not in_process_env:
                demo_env.close()

    ppo_config = PPOConfig(target_steps=args.steps_per_update, learning_rate=args.learning_rate,
                           demo_bc_coeff=args.demo_bc_coeff if demo_batch is not None else 0.0)
    ppo = PPO(ppo_config, device=args.device, optimizer_state=optimizer_state, demo_batch=demo_batch)

    with open(out_dir / 'run.json', 'w', encoding='utf-8') as file:
        json.dump({'args': vars(args), 'specialist': specialist, 'run_id': run_id, 'source': source,
                   'parameters': network.parameter_count(), 'model_config': network.config.to_dict(),
                   'ppo_config': ppo_config.to_dict(), 'start_iteration': start_iteration,
                   'start_steps': start_steps}, file, indent=2, default=str)

    print(f"[train] {specialist} run={run_id} dir={out_dir}")
    print(f"[train] parameters={network.parameter_count():,} grid={network.config.grid_encoder} "
          f"parallel={args.parallel} steps/update={args.steps_per_update} device={args.device}"
          + (f" from={source}" if source else ""))

    logger = MetricsLogger(str(out_dir))
    state = {'iteration': start_iteration, 'steps': start_steps, 'metrics': {}, 'saved': None}

    def save():
        registry.save_checkpoint(specialist, run_id, network, ppo.optimizer, iterations=state['iteration'],
                                 steps=state['steps'], metrics=state['metrics'], root=root)
        state['saved'] = state['iteration']

    def on_iteration(result: IterationResult):
        state['iteration'] = start_iteration + result.iteration
        state['steps'] = result.total_steps
        metrics = {**result.rollout.summary(), **result.stats}
        state['metrics'] = metrics
        logger.log(state['iteration'], state['steps'], metrics)
        print(f"[train] iter={state['iteration']} steps={state['steps']} " + format_summary(metrics), flush=True)
        if result.iteration % args.save_every == 0:
            save()

    started = time.time()
    try:
        ppo.train(network, (lambda: in_process_env) if in_process_env is not None else env_factory,
                  args.iterations, n_envs=args.parallel,
                  make_network=ModelFactory(ZeldaPolicy, None, network.config.to_dict()),
                  callback=on_iteration, start_steps=start_steps,
                  observation_space=ObservationBuilder.observation_space())
    except KeyboardInterrupt:
        print("[train] interrupted; saving checkpoint")
    finally:
        logger.close()
        if state['iteration'] > start_iteration and state['saved'] != state['iteration']:
            save()

    elapsed = time.time() - started
    trained = state['steps'] - start_steps
    print(f"[train] done: {state['iteration'] - start_iteration} iterations, {trained} steps in {elapsed:.1f}s "
          f"({trained / elapsed if elapsed else 0:.1f} steps/sec); checkpoint "
          f"{registry.checkpoint_path(specialist, run_id, root)}")
    return out_dir


if __name__ == '__main__':
    main()
