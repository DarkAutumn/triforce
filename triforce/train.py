"""Train one specialist with PPO (design §7.1).

    python -m triforce.train --specialist d1-melee [--load CKPT] [--parallel N] [--iterations N]
        [--steps-per-update N] [--device cpu|cuda] [--run-id ID] [--models-dir DIR]
        [--stop-success RATE [--stop-patience N]] [--curriculum TRAJECTORIES_DIR] [--events JSONL | --logs-dir DIR]

Each iteration collects ``--steps-per-update`` env steps (split across ``--parallel`` subprocess workers, one
emulator each; ``--parallel 1`` runs the env in-process for debugging), then runs one PPO update.  Output goes
to ``<models>/<specialist>/<run-id>/``: ``ckpt.pt`` (see ``registry``), ``metrics.jsonl``, ``tb/``
(TensorBoard), ``run.json`` (arguments and configs) and ``train_summary.json`` (iterations, steps, early
stop, wall time).  ``--stop-success`` ends training once the start-state-balanced training success rate
(``balanced_success_rate``, see ``EarlyStop``) stays at or above the threshold for ``--stop-patience``
consecutive iterations.  Only episodes that started at a room entry count; ``--curriculum`` episodes are reported
separately as ``curriculum/*``.
``--curriculum`` starts a share of episodes along trajectories found by ``triforce.explore`` (backward
curriculum, ``curriculum.py``); its shared progress file is ``<run dir>/curriculum.json``.

Live progress goes to an event stream (``events.py``): ``--events FILE`` appends to a caller-owned stream
(``scripts/train_round.py``); otherwise the run gets its own ``logs/<yymmdd>_<NNN>_<specialist>/`` folder with
``training.json`` and ``<specialist>.jsonl``, ending with an ``end`` event.

Weights come from ``--load`` (a checkpoint file or run dir) or else the specialist's ``warm_start`` (a
checkpoint path or another specialist's name, meaning its promoted checkpoint).  Loading a checkpoint of the
same specialist also restores the optimizer state and the iteration/step counters; a different specialist's
checkpoint only supplies weights.
"""

import argparse
import json
import os
import signal
import sys
import time
import traceback
from collections import Counter, deque
from pathlib import Path
from typing import Deque, Dict, Optional

import torch

from . import registry
from .curriculum import CurriculumSpec
from .demo import collect_demo_batch, parse_demo_actions
from .events import LOGS_DIR_ENV, PROGRESS_METRICS, EventLog, TrainingPlan, new_log_dir
from .metrics import MetricsLogger, format_summary
from .model import ModelConfig, ZeldaPolicy
from .observation import ObservationBuilder
from .ppo import PPO, IterationResult, PPOConfig
from .ppo_worker import ModelFactory, ZeldaEnvFactory
from .specialist import load_specialist


class EarlyStop:
    """Stop once the start-state-balanced training success rate >= threshold for `patience` consecutive iterations
    that finished room-entry episodes.

    The start-state pool oversamples states that fail, so the raw ``success_rate`` mostly measures the hardest (or
    an unwinnable) state and moves with the sampling as well as the policy.  Balanced = mean over start states of
    each state's success over its last ``window`` episodes; every state seen needs ``min_episodes`` first.
    """

    def __init__(self, threshold: Optional[float], patience: int, window: int = 20, min_episodes: int = 5):
        self.threshold = threshold
        self.patience = patience
        self.min_episodes = min_episodes
        self.window = window
        self.results: Dict[str, Deque[bool]] = {}
        self.balanced: Optional[float] = None
        self.streak = 0

    def update(self, start_results: Counter) -> bool:
        """Feeds one iteration's (start state, success) -> episodes counts; True = stop now."""
        if not start_results:
            return False
        for (state, success), count in start_results.items():
            self.results.setdefault(state, deque(maxlen=self.window)).extend([success] * count)
        if any(len(results) < self.min_episodes for results in self.results.values()):
            self.balanced = None
            return False
        self.balanced = sum(sum(r) / len(r) for r in self.results.values()) / len(self.results)
        if self.threshold is None:
            return False
        self.streak = self.streak + 1 if self.balanced >= self.threshold else 0
        return self.streak >= self.patience


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
    parser.add_argument('--stop-success', type=float, default=None,
                        help="stop early once training success_rate >= this for --stop-patience iterations")
    parser.add_argument('--stop-patience', type=int, default=5)
    parser.add_argument('--curriculum', default=None,
                        help="trajectories dir from triforce.explore: backward-curriculum starts (curriculum.py)")
    parser.add_argument('--curriculum-fraction', type=float, default=CurriculumSpec.fraction,
                        help="share of resets that start on a trajectory while any is incomplete")
    parser.add_argument('--curriculum-threshold', type=float, default=CurriculumSpec.threshold)
    parser.add_argument('--events', default=None,
                        help="append events to this JSONL (the caller owns the stream and writes its final 'end'); "
                             "default: a new folder under --logs-dir with training.json and <specialist>.jsonl")
    parser.add_argument('--logs-dir', default=None, help=f"default ${LOGS_DIR_ENV} or <repo>/logs")
    return parser.parse_args(argv)


def interrupt_on_sigterm():
    """Makes SIGTERM raise KeyboardInterrupt, so a killed run still saves its checkpoint and writes its final
    events.  (Background processes can inherit an ignored SIGINT; SIGTERM always arrives.)"""
    def handler(signum, frame):  # pylint: disable=unused-argument
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, handler)


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


def open_events(args, specialist: str, run_id: str, run_dir: Path):
    """(EventLog, TrainingPlan or None).  With --events the caller owns the stream; otherwise this training run
    gets its own log folder and writes the final 'end' event itself."""
    context = {'specialist': specialist, 'run_id': run_id}
    if args.events:
        return EventLog(args.events, **context), None
    log_dir = new_log_dir(specialist, args.logs_dir)
    plan = TrainingPlan(log_dir, specialist, 'train', [{
        'specialist': specialist, 'run_id': run_id, 'events': f'{specialist}.jsonl', 'run_dir': str(run_dir),
        'target_iterations': args.iterations, 'target_steps': args.iterations * args.steps_per_update}],
                        command=['python', '-m', 'triforce.train', *sys.argv[1:]])
    plan.update_run(0, status='running')
    print(f"[train] events: {log_dir / f'{specialist}.jsonl'}")
    return EventLog(log_dir / f'{specialist}.jsonl', **context), plan


def main(argv=None, session=None) -> Path:
    """Runs training; returns the run directory.  ``session`` (an existing GameSession) is used by the
    in-process env when ``--parallel 1`` (tests share one emulator per process)."""
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
    events, plan = open_events(args, specialist, run_id, out_dir)
    try:
        summary = _train(args, specialist, run_id, root, out_dir, session, events)
    except Exception as error:
        events.emit('error', message=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        events.emit('train_end', status='error')
        if plan is not None:
            events.emit('end', status='error', message=f"{type(error).__name__}: {error}")
            plan.update_run(0, status='error')
            plan.finish('failed')
        raise
    if plan is not None:
        events.emit('end', **summary)
        plan.update_run(0, **summary)
        plan.finish('interrupted' if summary['status'] == 'interrupted' else 'completed')
    return out_dir


def _train(args, specialist: str, run_id: str, root, out_dir: Path, session, events: EventLog) -> dict:
    """The training run proper; returns the train_end event fields."""
    # pylint: disable=too-many-locals, too-many-statements, too-many-arguments, too-many-positional-arguments
    network, optimizer_state, start_iteration, start_steps, source = initial_model(args, specialist, root)
    curriculum = None
    if args.curriculum:
        state_path = out_dir / 'curriculum.json'
        state_path.unlink(missing_ok=True)     # progress is per training invocation
        curriculum = CurriculumSpec(args.curriculum, str(state_path), fraction=args.curriculum_fraction,
                                    threshold=args.curriculum_threshold)
    env_factory = ZeldaEnvFactory(specialist, start_state=args.start_state, curriculum=curriculum,
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
    target_steps = start_steps + args.iterations * args.steps_per_update
    events.emit('start', target_iterations=start_iteration + args.iterations, target_steps=target_steps,
                start_iteration=start_iteration, start_steps=start_steps, steps_per_update=args.steps_per_update,
                parallel=args.parallel, device=args.device, curriculum=args.curriculum, source=source,
                stop_success=args.stop_success, stop_patience=args.stop_patience, run_dir=str(out_dir),
                pid=os.getpid())

    logger = MetricsLogger(str(out_dir))
    state = {'iteration': start_iteration, 'steps': start_steps, 'metrics': {}, 'saved': None,
             'stopped_early': False}
    early = EarlyStop(args.stop_success, args.stop_patience)
    ckpt = registry.checkpoint_path(specialist, run_id, root)

    def save():
        registry.save_checkpoint(specialist, run_id, network, ppo.optimizer, iterations=state['iteration'],
                                 steps=state['steps'], metrics=state['metrics'], root=root)
        state['saved'] = state['iteration']
        events.emit('checkpoint', iteration=state['iteration'], steps=state['steps'], path=str(ckpt))

    def on_iteration(result: IterationResult):
        state['iteration'] = start_iteration + result.iteration
        state['steps'] = result.total_steps
        metrics = {**result.rollout.summary(), **result.stats}
        stop = early.update(result.rollout.start_results)
        if early.balanced is not None:
            metrics['balanced_success_rate'] = early.balanced
        state['metrics'] = metrics
        logger.log(state['iteration'], state['steps'], metrics)
        print(f"[train] iter={state['iteration']} steps={state['steps']} " + format_summary(metrics), flush=True)
        elapsed = time.time() - started
        fraction = result.iteration / args.iterations
        events.emit('progress', iteration=state['iteration'], iterations=start_iteration + args.iterations,
                    steps=state['steps'], target_steps=target_steps, fraction=round(fraction, 4),
                    elapsed=round(elapsed, 1), sps=round((state['steps'] - start_steps) / elapsed, 1),
                    eta=round(elapsed * (1 - fraction) / fraction, 1),
                    metrics={k: metrics[k] for k in PROGRESS_METRICS if k in metrics})
        if result.iteration % args.save_every == 0:
            save()
        if stop:
            print(f"[train] early stop: balanced success_rate >= {args.stop_success} for {args.stop_patience} "
                  "iterations", flush=True)
            events.emit('early_stop', iteration=state['iteration'], steps=state['steps'],
                        balanced_success_rate=early.balanced)
            state['stopped_early'] = True
            return False
        return True

    started = time.time()
    status = 'completed'
    try:
        ppo.train(network, (lambda: in_process_env) if in_process_env is not None else env_factory,
                  args.iterations, n_envs=args.parallel,
                  make_network=ModelFactory(ZeldaPolicy, None, network.config.to_dict()),
                  callback=on_iteration, start_steps=start_steps,
                  observation_space=ObservationBuilder.observation_space())
    except KeyboardInterrupt:
        print("[train] interrupted; saving checkpoint")
        status = 'interrupted'
    finally:
        logger.close()
        if state['iteration'] > start_iteration and state['saved'] != state['iteration']:
            save()

    elapsed = time.time() - started
    trained = state['steps'] - start_steps
    sps = trained / elapsed if elapsed else 0.0
    with open(out_dir / 'train_summary.json', 'w', encoding='utf-8') as file:
        json.dump({'iterations': state['iteration'], 'steps': state['steps'],
                   'stopped_early': state['stopped_early'], 'wall_seconds': elapsed,
                   'steps_per_second': sps}, file, indent=2)
    print(f"[train] done: {state['iteration'] - start_iteration} iterations, {trained} steps in {elapsed:.1f}s "
          f"({sps:.1f} steps/sec); checkpoint {ckpt}")
    summary = {'status': 'early_stop' if state['stopped_early'] else status, 'iterations': state['iteration'],
               'steps': state['steps'], 'wall_seconds': round(elapsed, 1), 'sps': round(sps, 1),
               'checkpoint': str(ckpt) if ckpt.is_file() else None}
    events.emit('train_end', **summary)
    return summary


if __name__ == '__main__':
    interrupt_on_sigterm()
    main()
