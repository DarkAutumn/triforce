"""Train one specialist with PPO (design §7.1).

    python -m triforce.train --specialist stalfos-keese-gel [--load CKPT] [--parallel N] [--threads N]
        [--iterations N] [--steps-per-update N] [--device cpu|cuda] [--run-id ID] [--models-dir DIR]
        [--stop-success RATE [--stop-patience N]] [--curriculum TRAJECTORIES_DIR] [--events JSONL | --logs-dir DIR]
        [--ent-anneal SHARE] [--shaping-anneal N] [--best-of-n RUNS [--best-keep K] [--best-every N]
        [--best-coeff C] [--best-parallel P]] [--sil-coeff C [--sil-value-coeff B] [--sil-capacity ROWS]]

Each iteration collects ``--steps-per-update`` env steps, split across ``--parallel`` envs of one native
``EnvPool`` (``--threads`` worker threads, default ``min(parallel, cpu count)``; results do not depend on the
thread count): one ``EnvPool.rollout`` call in which the network's Rust copy acts inside the pool
(``ppo.Collector``, docs/rust-actor.md).  Then one PPO update on ``--device``.  Output goes
to ``<models>/<specialist>/<run-id>/``: ``ckpt.pt`` (see ``registry``), ``metrics.jsonl``, ``tb/``
(TensorBoard), ``run.json`` (arguments and configs) and ``train_summary.json`` (``status`` completed | early_stop |
interrupted, total and this run's iterations/steps, early stop, wall time).  An interrupted run (SIGINT, or SIGTERM
from the command line) still saves its checkpoint and summary but exits 130, so a driver never mistakes it for a
finished fixed-budget run.  ``--stop-success`` ends training once the start-state-balanced training success rate
(``balanced_success_rate``, see ``EarlyStop``) stays at or above the threshold for ``--stop-patience``
consecutive iterations.  Only episodes that started at a room entry count; ``--curriculum`` episodes are reported
separately as ``curriculum/*``.
``--shaping-anneal N`` (outcome fine-tuning, better-play §1a) lowers the pool's shaping scale (``reward.shaping_scale``:
every non-terminal reward and penalty is multiplied by it) linearly to 0 over the first N iterations, then keeps it
at 0, so the policy optimizes the health-scaled terminal outcome (plus ``reward.frame_penalty``, if set).
``--ent-anneal SHARE`` lowers the entropy bonus linearly to 0 over the last SHARE of the iterations (§6b).
``--best-of-n RUNS`` (best-of-N self-distillation, §3; ``best_of_n.py``): at the start and every ``--best-every``
iterations the current policy plays RUNS episodes from every seed start state on its own ``EvalTask`` pool
(``--best-parallel`` envs), and its best ``--best-keep`` successful runs per state (success, hearts lost, frames)
become the behavior-cloning batch, weighted ``--best-coeff`` (the demo-regularization term; ``best/*`` metrics).
``--sil-coeff C`` (self-imitation learning, §3; ``ppo.SilReplay``): each rollout's rows whose λ=1 return R beat the
critic's value V(s) go into a FIFO replay of ``--sil-capacity`` rows; every minibatch adds
``C · (−log π(a|s) · max(0, R − V) + ½ β · max(0, R − V)²)`` over a minibatch-sized uniform sample of it
(β = ``--sil-value-coeff``; ``losses/sil_*``, ``charts/sil_rows``, ``charts/sil_positive_share``).  0 = off.
``--curriculum`` starts a share of episodes along trajectories found by ``triforce.explore`` (backward
curriculum, native ``CurriculumSpec``); its shared progress file is ``<run dir>/curriculum.json``.

Live progress goes to an event stream (``events.py``): ``--events FILE`` appends to a caller-owned stream
(``scripts/train_round.py``); otherwise the run gets its own ``logs/<yymmdd>_<NNN>_<specialist>/`` folder with
``training.json`` and ``<specialist>.jsonl``, ending with an ``end`` event.

Weights come from ``--load`` or else the specialist's ``warm_start``: a checkpoint file, a run dir, or a specialist
name (its promoted checkpoint; ``--load <own name>`` continues the promoted model).  Loading a checkpoint of the
same specialist also restores the optimizer state and the iteration/step counters; a different specialist's
checkpoint only supplies weights.  With ``obs.target`` on, a checkpoint from before the ``target`` block (schema 1)
is widened to the current layout (``model.widen_policy``: the new inputs start at zero, so it acts exactly as
before) and its optimizer state is dropped; with it off the checkpoint keeps its own layout.
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
from ._native import CurriculumSpec, Env, EnvPool, TrainTask
from .best_of_n import collect_best
from .demo import collect_demo_batch, parse_demo_actions
from .events import LOGS_DIR_ENV, PROGRESS_METRICS, EventLog, TrainingPlan, new_log_dir
from .metrics import MetricsLogger, format_summary
from .model import ModelConfig, ZeldaPolicy, obs_spec_from_space
from .observation import observation_space
from .paths import STATES_DIR, TRAJECTORIES_DIR
from .ppo import PPO, IterationResult, PPOConfig
from .specialist import SpecialistConfig, load_specialist


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
    parser.add_argument('--load', default=None,
                        help="checkpoint file, run dir, or a specialist name (its promoted checkpoint) to start from")
    parser.add_argument('--parallel', type=int, default=1, help="number of envs in the pool")
    parser.add_argument('--threads', type=int, default=None,
                        help="env worker threads (default: min(parallel, cpu count))")
    parser.add_argument('--iterations', type=int, default=100, help="PPO iterations (rollout + update)")
    parser.add_argument('--steps-per-update', type=int, default=PPOConfig.target_steps)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--models-dir', default=None, help=f"default ${registry.MODELS_DIR_ENV} or ./models")
    parser.add_argument('--save-every', type=int, default=10, help="checkpoint every N iterations (and at the end)")
    parser.add_argument('--learning-rate', type=float, default=PPOConfig.learning_rate)
    parser.add_argument('--ent-anneal', type=float, default=PPOConfig.ent_anneal,
                        help="share of the iterations, at the end, over which the entropy bonus falls to 0")
    parser.add_argument('--shaping-anneal', type=int, default=PPOConfig.shaping_anneal,
                        help="iterations over which the shaping scale falls linearly to 0 (outcome fine-tuning)")
    parser.add_argument('--start-state', default=None, help="pin every episode to this savestate (name or path)")
    parser.add_argument('--states-dir', default=None,
                        help="where bare start-state names resolve (default: $TRIFORCE_STATES_DIR or triforce/states)")
    parser.add_argument('--capture-states', action='store_true',
                        help="let envs write first-entry savestates into the states dir (off by default: "
                             "parallel envs would race on the same files)")
    parser.add_argument('--demo', default=None, help="demo trace (Action.parse lines) for demo regularization")
    parser.add_argument('--demo-bc-coeff', type=float, default=0.1)
    parser.add_argument('--demo-allow-incomplete', action='store_true',
                        help="accept a demo that stops before the episode ends in success")
    parser.add_argument('--best-of-n', type=int, default=0, metavar='RUNS',
                        help="best-of-N self-distillation (better-play §3): every --best-every iterations, play RUNS "
                             "episodes from every seed start state and behavior-clone the best --best-keep "
                             "successful runs of each state (weight --best-coeff); 0 = off")
    parser.add_argument('--best-keep', type=int, default=2)
    parser.add_argument('--best-every', type=int, default=20)
    parser.add_argument('--best-coeff', type=float, default=0.2)
    parser.add_argument('--best-parallel', type=int, default=4, help="envs of the best-of-N collection pool")
    parser.add_argument('--sil-coeff', type=float, default=PPOConfig.sil_coeff,
                        help="self-imitation learning (better-play §3): weight of the SIL loss on replayed rows whose "
                             "Monte-Carlo return beat the critic; 0 = off")
    parser.add_argument('--sil-value-coeff', type=float, default=PPOConfig.sil_value_coeff,
                        help="SIL value-term weight (beta)")
    parser.add_argument('--sil-capacity', type=int, default=PPOConfig.sil_capacity, help="SIL replay rows (FIFO)")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--stop-success', type=float, default=None,
                        help="stop early once training success_rate >= this for --stop-patience iterations")
    parser.add_argument('--stop-patience', type=int, default=5)
    parser.add_argument('--curriculum', default=None,
                        help="trajectories dir from triforce.explore: backward-curriculum starts (curriculum.py); "
                             "default: the specialist's curriculum config")
    parser.add_argument('--no-curriculum', action='store_true', help="ignore the specialist's curriculum config")
    parser.add_argument('--curriculum-fraction', type=float, default=None,
                        help="share of resets that start on a trajectory while any is incomplete (default: config)")
    parser.add_argument('--curriculum-threshold', type=float, default=None, help="default: config")
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


EXIT_INTERRUPTED = 130


def run_status(run_dir) -> Optional[str]:
    """``train_summary.json``'s status (completed | early_stop | interrupted), or None without a summary (a run that
    crashed, is still running, or predates the field)."""
    path = Path(run_dir) / 'train_summary.json'
    if not path.is_file():
        return None
    with open(path, encoding='utf-8') as file:
        return json.load(file).get('status')


def cli(argv=None) -> int:
    """``python -m triforce.train`` / ``triforce train``: trains, then exits 0, or ``EXIT_INTERRUPTED`` if the run was
    interrupted (its checkpoint and summary are still saved)."""
    interrupt_on_sigterm()
    return EXIT_INTERRUPTED if run_status(main(argv)) == 'interrupted' else 0


def resolve_warm_start(value: str, root) -> Path:
    """A warm_start entry is a checkpoint path/run dir, or a specialist name (its promoted checkpoint)."""
    if os.path.exists(value):
        return Path(value)
    return registry.promoted_checkpoint(value, root)


def initial_model(args, config: SpecialistConfig, root):
    """(network, optimizer_state, start_iteration, start_steps, source)."""
    if args.load:
        source = str(resolve_warm_start(args.load, root))
    else:
        source = str(resolve_warm_start(config.warm_start, root)) if config.warm_start else None
    if source is None:
        network = ZeldaPolicy(observation_space(), ModelConfig())
        return network, None, 0, 0, None

    # obs.target wants the full layout: an older checkpoint without `target` is widened (zero-initialised inputs).
    wanted = obs_spec_from_space(observation_space()) if config.obs.target else None
    network, checkpoint = registry.load_model(source, obs_spec=wanted)
    added = sorted(set(network.config.obs_spec) - set(checkpoint['model_config']['obs_spec']))
    if added:
        print(f"[train] widened {source} to the current observation layout ({added} start at zero, fresh optimizer)")
    if args.load and checkpoint['specialist'] == config.name:
        optimizer = None if added else checkpoint.get('optimizer')
        return network, optimizer, checkpoint['iterations'], checkpoint['steps'], source
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


def curriculum_spec(args, config, out_dir: Path) -> tuple[Optional[CurriculumSpec], Optional[str]]:
    """(spec, trajectories dir) of the run's curriculum: --curriculum, else the specialist's ``curriculum`` config
    when enabled (and not --no-curriculum).  Progress (``<run dir>/curriculum.json``) restarts with every
    training invocation."""
    settings = config.curriculum
    if args.no_curriculum or not (args.curriculum or settings.enabled):
        return None, None
    trajectories = str(args.curriculum or settings.trajectories or TRAJECTORIES_DIR / config.name)
    state_path = out_dir / 'curriculum.json'
    state_path.unlink(missing_ok=True)
    fraction = settings.fraction if args.curriculum_fraction is None else args.curriculum_fraction
    threshold = settings.threshold if args.curriculum_threshold is None else args.curriculum_threshold
    return CurriculumSpec(trajectories, str(state_path), fraction=fraction, threshold=threshold), trajectories


def main(argv=None) -> Path:
    """Runs training; returns the run directory."""
    args = parse_args(argv)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if args.parallel < 1:
        raise SystemExit("--parallel must be >= 1")
    if args.threads is None:
        args.threads = min(args.parallel, os.cpu_count() or 1)
    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")

    specialist = load_specialist(args.specialist).name
    root = registry.models_root(args.models_dir)
    run_id = args.run_id or registry.new_run_id()
    out_dir = registry.run_dir(specialist, run_id, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    events, plan = open_events(args, specialist, run_id, out_dir)
    try:
        summary = _train(args, specialist, run_id, root, out_dir, events)
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


def _train(args, specialist: str, run_id: str, root, out_dir: Path, events: EventLog) -> dict:
    """The training run proper; returns the train_end event fields."""
    # pylint: disable=too-many-locals, too-many-statements
    config = load_specialist(args.specialist)
    network, optimizer_state, start_iteration, start_steps, source = initial_model(args, config, root)
    curriculum, trajectories = curriculum_spec(args, config, out_dir)
    capture_dir = str(args.states_dir or STATES_DIR) if args.capture_states else None
    task = TrainTask(config, str(out_dir), start_state=args.start_state, curriculum=curriculum,
                     capture_dir=capture_dir, states_dir=args.states_dir)

    demo_batch = None
    if args.demo:
        demo_env = Env(config, start_state=args.start_state, capture_dir=None, states_dir=args.states_dir,
                       seed=args.seed or 0)
        demo_batch = collect_demo_batch(demo_env, parse_demo_actions(args.demo),
                                        require_success=not args.demo_allow_incomplete)

    ppo_config = PPOConfig(target_steps=args.steps_per_update, learning_rate=args.learning_rate,
                           ent_anneal=args.ent_anneal, shaping_anneal=args.shaping_anneal,
                           demo_bc_coeff=args.demo_bc_coeff if demo_batch is not None else 0.0,
                           sil_coeff=args.sil_coeff, sil_value_coeff=args.sil_value_coeff,
                           sil_capacity=args.sil_capacity)
    ppo = PPO(ppo_config, device=args.device, optimizer_state=optimizer_state, demo_batch=demo_batch)

    # Built before the 'start' event so bad start states or trajectories fail the run up front.
    pool_seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(8), 'little') >> 1
    pool = EnvPool(task, num_envs=args.parallel, threads=args.threads, seed=pool_seed)

    with open(out_dir / 'run.json', 'w', encoding='utf-8') as file:
        json.dump({'args': vars(args), 'specialist': specialist, 'run_id': run_id, 'source': source,
                   'parameters': network.parameter_count(), 'model_config': network.config.to_dict(),
                   'ppo_config': ppo_config.to_dict(), 'start_iteration': start_iteration,
                   'start_steps': start_steps}, file, indent=2, default=str)

    print(f"[train] {specialist} run={run_id} dir={out_dir}")
    print(f"[train] parameters={network.parameter_count():,} "
          f"parallel={args.parallel} threads={args.threads} steps/update={args.steps_per_update} "
          f"device={args.device}"
          + (f" from={source}" if source else ""))
    target_steps = start_steps + args.iterations * args.steps_per_update
    events.emit('start', target_iterations=start_iteration + args.iterations, target_steps=target_steps,
                start_iteration=start_iteration, start_steps=start_steps, steps_per_update=args.steps_per_update,
                parallel=args.parallel, threads=args.threads, device=args.device,
                curriculum=trajectories, source=source,
                stop_success=args.stop_success, stop_patience=args.stop_patience, run_dir=str(out_dir),
                pid=os.getpid())

    logger = MetricsLogger(str(out_dir))
    state = {'iteration': start_iteration, 'steps': start_steps, 'metrics': {}, 'saved': None,
             'stopped_early': False}
    early = EarlyStop(args.stop_success, args.stop_patience)
    ckpt = registry.checkpoint_path(specialist, run_id, root)

    def refresh_best(collection: int) -> dict:
        """Best-of-N: replaces the BC batch with the current policy's best runs; returns its metrics."""
        best = collect_best(config, network.to_native(), args.best_of_n, args.best_keep,
                            seed=(pool_seed + collection * 1_000_003) % (1 << 62), parallel=args.best_parallel,
                            states_dir=args.states_dir)
        ppo.demo_batch = best.batch if best.steps else None
        ppo.config.demo_bc_coeff = args.best_coeff if best.steps else 0.0
        return {'best/episodes': best.episodes, 'best/success_rate': best.successes / max(1, best.episodes),
                'best/kept': best.kept, 'best/states': best.states, 'best/steps': best.steps}

    if args.best_of_n > 0 and args.demo:
        raise ValueError("--best-of-n and --demo both set the behavior-cloning batch; use one")
    best_metrics = refresh_best(0) if args.best_of_n > 0 else {}

    def save():
        registry.save_checkpoint(specialist, run_id, network, ppo.optimizer, iterations=state['iteration'],
                                 steps=state['steps'], metrics=state['metrics'], root=root)
        state['saved'] = state['iteration']
        events.emit('checkpoint', iteration=state['iteration'], steps=state['steps'], path=str(ckpt))

    def on_iteration(result: IterationResult):
        state['iteration'] = start_iteration + result.iteration
        state['steps'] = result.total_steps
        metrics = {**result.rollout.summary(), **result.stats, **best_metrics}
        best_metrics.clear()
        if args.best_of_n > 0 and result.iteration % args.best_every == 0 and result.iteration < args.iterations:
            best_metrics.update(refresh_best(result.iteration // args.best_every))
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
                    metrics={k: v for k, v in metrics.items() if k in PROGRESS_METRICS
                             or k.startswith('linger/room/')})
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
        ppo.train(network, pool, args.iterations, callback=on_iteration, start_steps=start_steps,
                  seed=args.seed)
    except KeyboardInterrupt:
        print("[train] interrupted; saving checkpoint")
        status = 'interrupted'
    finally:
        pool.close()
        logger.close()
        if state['iteration'] > start_iteration and state['saved'] != state['iteration']:
            save()

    elapsed = time.time() - started
    trained = state['steps'] - start_steps
    sps = trained / elapsed if elapsed else 0.0
    status = 'early_stop' if state['stopped_early'] else status
    with open(out_dir / 'train_summary.json', 'w', encoding='utf-8') as file:
        json.dump({'status': status, 'iterations': state['iteration'], 'steps': state['steps'],
                   'iterations_trained': state['iteration'] - start_iteration, 'steps_trained': trained,
                   'stopped_early': state['stopped_early'], 'wall_seconds': elapsed,
                   'steps_per_second': sps}, file, indent=2)
    print(f"[train] {'interrupted' if status == 'interrupted' else 'done'}: {state['iteration'] - start_iteration} "
          f"iterations, {trained} steps in {elapsed:.1f}s ({sps:.1f} steps/sec); checkpoint {ckpt}")
    summary = {'status': status, 'iterations': state['iteration'],
               'steps': state['steps'], 'wall_seconds': round(elapsed, 1), 'sps': round(sps, 1),
               'checkpoint': str(ckpt) if ckpt.is_file() else None}
    events.emit('train_end', **summary)
    return summary


if __name__ == '__main__':
    sys.exit(cli())
