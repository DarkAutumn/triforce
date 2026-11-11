"""Train one specialist with PPO (design §7.1).

    python -m triforce.train --specialist stalfos-keese-gel [--load CKPT] [--parallel N] [--threads N]
        [--iterations N] [--steps-per-update N] [--device cpu|cuda] [--run-id ID] [--models-dir DIR]
        [--stop-success RATE [--stop-patience N]] [--curriculum TRAJECTORIES_DIR] [--events JSONL | --logs-dir DIR]
        [--ent-anneal SHARE] [--shaping-anneal N] [--best-of-n RUNS [--best-keep K] [--best-every N]
        [--best-coeff C] [--best-parallel P]] [--sil-coeff C [--sil-value-coeff B] [--sil-capacity ROWS]]
        [--search JSON [--search-coeff C]] [--plr [--plr-beta B] [--plr-rho R] [--plr-uniform U]]
        [--ppg [--ppg-every N] [--ppg-aux-epochs E] [--ppg-clone-coeff C]]

Each iteration collects ``--steps-per-update`` env steps, split across ``--parallel`` envs of one native
``EnvPool`` (``--threads`` worker threads, default ``min(parallel, cpu count)``; results do not depend on the
thread count): one ``EnvPool.rollout`` call in which the network's Rust copy acts inside the pool
(``ppo.Collector``, docs/rust-actor.md).  Then one PPO update on ``--device``.  Output goes
to ``<models>/<specialist>/<run-id>/``: ``ckpt.pt`` (see ``registry``), ``metrics.jsonl``, ``tb/``
(TensorBoard), ``run.json`` (arguments and configs) and ``train_summary.json`` (``status`` completed | early_stop |
interrupted, total and this run's iterations/steps, early stop, wall time).  Training into an existing run dir first
moves the earlier training's files to ``<run>.prev-N`` (``set_aside_previous``); only a same-dir ``--load``
continuation keeps its checkpoint, curriculum and ``metrics.jsonl`` (appended to).  An interrupted run (SIGINT, or
SIGTERM from the command line) still saves its checkpoint and summary but exits 130, so a driver never mistakes it for
a finished fixed-budget run.  ``--stop-success`` proposes a stop once the start-state-balanced success rate of the
per-start cohorts (``balanced_success_rate``, see ``EarlyStop``) stays at or above the threshold for
``--stop-patience`` consecutive iterations (never before a budget failure could have been counted), and stops only
if an evaluation of the current weights (``CONFIRM_EPISODES`` sampled episodes per seed start state on an
``EvalTask`` pool, seed = pool seed + iteration) also reaches it; every check is an ``early_stop_check`` event.  The
evaluation ends early once its failures rule the threshold out (``aborted``, with the partial ``rate``, ``failures``
and ``rate_upper_bound``, the best rate it could still have reached): the decision is the full evaluation's.
Only episodes that started at a room entry count, and the balance is over the seed start states only: episodes of
harvested start states (chain handoffs) are reported as ``harvested/*`` and never move it; ``--curriculum`` episodes
are reported separately as ``curriculum/*``.  ``in_flight/*``, ``success_rate_pessimistic`` and ``started/*``
(``metrics``) show the episodes still running at each rollout's end and the per-start cohort rate.  ``run.json``
and the ``start`` event record the harvested source (``harvested``: dir, state count, fraction, whether resets
draw them; a warning is printed when the dir holds states) and the curriculum tracks loaded with their
trajectory.json times (``curriculum_tracks``).
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
``--search JSON`` (Gumbel expert iteration, §2; docs/rust-actor.md "Search"): the pool searches a share of the
training decisions (``EnvPool.rollout(search=JSON)``, e.g. ``'{"candidates":4,"simulations":8,"futures":2,"depth":3,
"fraction":0.25}'``; missing keys take the Rust defaults) and executes the search's choice.  Those rows train the
policy by distillation, ``C · CE(π', π_θ)`` toward the search's improved policy (C = ``--search-coeff``), in place of
the PPO ratio term; value loss, entropy bonus and GAE cover every row (``losses/search_ce``, ``charts/search_rows``,
``charts/search_agreement``).  Searched decisions cost about 100× a plain one.
``--plr`` (prioritized level replay, §5; ``plr.PrioritizedReplay``): after each rollout the seed start states are
scored by their episodes' positive value loss (mean max(GAE advantage, 0)) and the pool draws the next rollout's seed
states by rank-prioritized scores (temperature ``--plr-beta``) mixed with staleness (``--plr-rho``) and uniform
(``--plr-uniform``) sampling instead of its failure-weighted draws (``plr/entropy``, ``plr/top_state``,
``plr/weight/<state>``, ``plr/scored_episodes``).  Curriculum and harvested starts are unaffected.
``--ppg`` (phasic policy gradient, §6a; ``ppo.PPGStore``, ``PPO.auxiliary_phase``): the PPO updates stay as they
are (the policy phase); every ``--ppg-every`` iterations an auxiliary phase trains ``--ppg-aux-epochs`` epochs over
those iterations' rollouts with the value loss ``½(V − R)²`` (R: the GAE returns) plus ``C · KL(π_old ‖ π)`` toward
the policy from before the phase (C = ``--ppg-clone-coeff``; ``ppg/value_loss``, ``ppg/value_loss_start``,
``ppg/kl``, ``ppg/explained_variance``, ``ppg/explained_variance_start``, ``ppg/rows``, ``ppg/seconds`` on the
iterations that run one).
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
import math
import os
import shutil
import signal
import sys
import time
import traceback
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Iterable, Optional

import torch

from . import registry
from ._native import CurriculumSpec, EnvPool, TrainTask
from .best_of_n import collect_best, success_by_state
from .events import LOGS_DIR_ENV, PROGRESS_METRICS, EventLog, TrainingPlan, new_log_dir, write_json_atomic
from .metrics import MetricsLogger, format_summary
from .model import ModelConfig, ZeldaPolicy, obs_spec_from_space
from .observation import observation_space
from .paths import TRAJECTORIES_DIR
from .ppo import PPO, IterationResult, PPOConfig
from .specialist import SpecialistConfig, load_specialist


# Episodes per seed start state of the evaluation that confirms an early-stop proposal (``EarlyStop``).
CONFIRM_EPISODES = 50


class EarlyStop:
    """Proposes an early stop once the start-state-balanced success rate of the per-start cohorts >= threshold for
    `patience` consecutive iterations; ``train`` confirms each proposal with an evaluation (``confirm_stop``).

    Fed with the released cohorts (``metrics.EpisodeTracker``: the room-entry episodes that started in one rollout,
    released once all of them ended), so a slow failure counts with the rollout it started in instead of arriving
    late.  The start-state pool oversamples states that fail, so the raw ``success_rate`` mostly measures the hardest
    (or an unwinnable) state and moves with the sampling as well as the policy.  Balanced = mean over start states of
    each state's exact success rate over its most recent whole cohorts that together cover at least ``window``
    episodes (a cohort with 113 episodes of a state counts all 113); every state seen needs ``min_episodes`` first.
    ``seed_states`` (the pool's seed labels, ``EnvPool.start_states``; None = every label, e.g. one pinned
    ``--start-state``) are the states balanced over: other room-entry episodes (harvested chain handoffs, one label
    per file) never move the rate; ``harvested`` holds their (successes, episodes) of the last update, logged as
    ``harvested/success_rate``.

    Guards: no proposal before ``min_iterations`` updates (``ceil(max_steps / steps per env) + patience``: a budget
    failure could not have been counted earlier); an iteration that finished no room-entry episode, or whose share
    of envs with an episode older than half the step budget exceeds ``1 - threshold``, resets the streak.
    """

    def __init__(self, threshold: Optional[float], patience: int, window: int = 20, min_episodes: int = 5,
                 min_iterations: int = 0, seed_states: Optional[Iterable[str]] = None):
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.threshold = threshold
        self.patience = patience
        self.min_episodes = min_episodes
        self.window = window
        self.min_iterations = min_iterations
        self.seed_states = None if seed_states is None else frozenset(seed_states)
        self.results: Dict[str, Deque[tuple[int, int]]] = {}     # state -> per-cohort (successes, episodes)
        self.harvested = (0, 0)
        self.balanced: Optional[float] = None
        self.streak = 0
        self.iterations = 0

    def state_rate(self, state: str) -> tuple[int, int]:
        """(successes, episodes) of ``state`` over its covered cohorts."""
        history = self.results[state]
        return sum(s for s, _ in history), sum(e for _, e in history)

    def rates(self) -> dict[str, float]:
        """Each state's success rate over its covered cohorts."""
        return {state: s / e for state, (s, e) in ((state, self.state_rate(state)) for state in self.results) if e}

    def add(self, cohort: Counter):
        """One cohort's (start state, success) -> episodes counts; states outside ``seed_states`` add to
        ``harvested`` instead."""
        counts: Dict[str, list[int]] = {}
        for (state, success), count in cohort.items():
            if self.seed_states is not None and state not in self.seed_states:
                self.harvested = (self.harvested[0] + (count if success else 0), self.harvested[1] + count)
                continue
            entry = counts.setdefault(state, [0, 0])
            entry[0] += count if success else 0
            entry[1] += count
        for state, (successes, episodes) in counts.items():
            history = self.results.setdefault(state, deque())
            history.append((successes, episodes))
            total = sum(e for _, e in history)
            while total - history[0][1] >= self.window:     # the older cohorts aren't needed for the window
                total -= history.popleft()[1]

    def update(self, cohorts, finished: int, over_half_budget: Optional[float] = None) -> bool:
        """One training iteration: its released ``cohorts`` (Counters, in rollout order), the room-entry episodes
        it finished and its ``in_flight/over_half_budget``; True = propose a stop."""
        self.iterations += 1
        self.harvested = (0, 0)
        for cohort in cohorts:
            self.add(cohort)
        rates = [self.state_rate(state) for state in self.results]
        if not rates or any(episodes < self.min_episodes for _, episodes in rates):
            self.balanced = None
        else:
            self.balanced = sum(successes / episodes for successes, episodes in rates) / len(rates)
        if self.threshold is None:
            return False
        blind = finished == 0 or (over_half_budget is not None and over_half_budget > 1 - self.threshold)
        if blind or self.balanced is None or self.balanced < self.threshold:
            self.streak = 0
        else:
            self.streak += 1
        return self.streak >= self.patience and self.iterations >= self.min_iterations

    def reset(self):
        """A proposal the confirmation evaluation turned down: the streak starts over."""
        self.streak = 0


def _search_settings(text: str) -> dict:
    """``--search``: a JSON object of ``EnvPool.rollout`` search settings (the pool validates the keys)."""
    try:
        settings = json.loads(text)
    except json.JSONDecodeError as err:
        raise argparse.ArgumentTypeError(f"not JSON: {err}") from err
    if not isinstance(settings, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return settings


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
    parser.add_argument('--models-dir', default=None,
                        help=f"default ${registry.MODELS_DIR_ENV} or <repo>/models")
    parser.add_argument('--save-every', type=int, default=10, help="checkpoint every N iterations (and at the end)")
    parser.add_argument('--learning-rate', type=float, default=PPOConfig.learning_rate)
    parser.add_argument('--ent-anneal', type=float, default=PPOConfig.ent_anneal,
                        help="share of the iterations, at the end, over which the entropy bonus falls to 0")
    parser.add_argument('--shaping-anneal', type=int, default=PPOConfig.shaping_anneal,
                        help="iterations over which the shaping scale falls linearly to 0 (outcome fine-tuning)")
    parser.add_argument('--start-state', default=None, help="pin every episode to this savestate (name or path)")
    parser.add_argument('--states-dir', default=None,
                        help="where bare start-state names resolve (default: $TRIFORCE_STATES_DIR or triforce/states)")
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
    parser.add_argument('--search', type=_search_settings, default=None, metavar='JSON',
                        help="Gumbel expert iteration (better-play §2): EnvPool.rollout search settings as a JSON "
                             "object, e.g. '{\"candidates\":4,\"simulations\":8,\"futures\":2,\"depth\":3,"
                             "\"fraction\":0.25}'; searched rows train by distillation toward the search's policy")
    parser.add_argument('--search-coeff', type=float, default=PPOConfig.search_coeff,
                        help="weight of the search distillation loss CE(pi', pi)")
    parser.add_argument('--plr', action='store_true',
                        help="prioritized level replay (better-play §5): draw seed start states by their recent "
                             "positive value loss (rank-prioritized, plus staleness and uniform shares)")
    parser.add_argument('--plr-beta', type=float, default=PPOConfig.plr_beta, help="PLR rank temperature")
    parser.add_argument('--plr-rho', type=float, default=PPOConfig.plr_rho, help="PLR staleness share")
    parser.add_argument('--plr-uniform', type=float, default=PPOConfig.plr_uniform, help="PLR uniform share")
    parser.add_argument('--ppg', action='store_true',
                        help="phasic policy gradient (better-play §6a): every --ppg-every iterations, an auxiliary "
                             "phase of value training with a KL clone term toward the policy from before it")
    parser.add_argument('--ppg-every', type=int, default=PPOConfig.ppg_every,
                        help="policy-phase iterations per auxiliary phase")
    parser.add_argument('--ppg-aux-epochs', type=int, default=PPOConfig.ppg_aux_epochs,
                        help="epochs of each auxiliary phase")
    parser.add_argument('--ppg-clone-coeff', type=float, default=PPOConfig.ppg_clone_coeff,
                        help="weight of KL(pi_old || pi) in the auxiliary phase")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--stop-success', type=float, default=None,
                        help="stop early once the start-state-balanced training success (balanced_success_rate, "
                             "see EarlyStop) >= this for --stop-patience iterations")
    parser.add_argument('--stop-patience', type=int, default=5)
    parser.add_argument('--curriculum', default=None,
                        help="trajectories dir from triforce.explore: backward-curriculum starts "
                             "(crates/env/src/curriculum.rs); default: the specialist's curriculum config")
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


def steps_per_iteration(args) -> int:
    """Env steps one iteration trains: ``--steps-per-update`` rounded down to a multiple of ``--parallel`` (each env
    steps ``steps_per_update // parallel`` decisions)."""
    return (args.steps_per_update // args.parallel) * args.parallel


def validate_args(args):
    """Rejects values that would crash mid-run or train something other than what is reported (SystemExit)."""
    checks = [(args.parallel >= 1, "--parallel must be >= 1"),
              (args.iterations >= 1, "--iterations must be >= 1"),
              (args.save_every >= 1, "--save-every must be >= 1"),
              (args.stop_patience >= 1, "--stop-patience must be >= 1"),
              (args.best_of_n <= 0 or args.best_every >= 1, "--best-every must be >= 1"),
              (args.steps_per_update >= args.parallel, "--steps-per-update must be >= --parallel")]
    for ok, message in checks:
        if not ok:
            raise SystemExit(message)


def open_events(args, specialist: str, run_id: str, run_dir: Path, argv: list[str]):
    """(EventLog, TrainingPlan or None).  With --events the caller owns the stream; otherwise this training run
    gets its own log folder and writes the final 'end' event itself."""
    context = {'specialist': specialist, 'run_id': run_id}
    if args.events:
        return EventLog(args.events, **context), None
    log_dir = new_log_dir(specialist, args.logs_dir)
    plan = TrainingPlan(log_dir, specialist, 'train', [{
        'specialist': specialist, 'run_id': run_id, 'events': f'{specialist}.jsonl', 'run_dir': str(run_dir),
        'target_iterations': args.iterations, 'target_steps': args.iterations * steps_per_iteration(args)}],
                        command=['python', '-m', 'triforce.train', *argv])
    plan.update_run(0, status='running')
    print(f"[train] events: {log_dir / f'{specialist}.jsonl'}")
    return EventLog(log_dir / f'{specialist}.jsonl', **context), plan


def continued_run_dir(args, source: Optional[str], start_iteration: int) -> Optional[Path]:
    """The run dir a ``--load`` continuation (same specialist: counters and optimizer restored) comes from."""
    if not args.load or source is None or start_iteration == 0:
        return None
    path = Path(source)
    return (path if path.is_dir() else path.parent).resolve()


def _same_trajectories(run_dir: Path, trajectories: str) -> bool:
    """Whether ``run_dir`` trained on ``trajectories`` (its ``run.json`` ``curriculum``; runs from before that
    field count as the same: ``Curriculum::new`` drops or restarts any track that changed)."""
    try:
        recorded = json.loads((run_dir / 'run.json').read_text(encoding='utf-8')).get('curriculum')
    except (OSError, ValueError):
        recorded = None
    return recorded is None or Path(recorded).resolve() == Path(trajectories).resolve()


def curriculum_spec(args, config, out_dir: Path,
                    continued: Optional[Path] = None) -> tuple[Optional[CurriculumSpec], Optional[str]]:
    """(spec, trajectories dir) of the run's curriculum: --curriculum, else the specialist's ``curriculum`` config
    when enabled (and not --no-curriculum).  Progress (``<run dir>/curriculum.json``) starts fresh, except for a
    ``--load`` continuation (``continued``: its run dir) on the same trajectories, which resumes the loaded run's
    progress (copied; tracks whose length changed or that are new start fresh, ``Curriculum::new``)."""
    settings = config.curriculum
    if args.no_curriculum or not (args.curriculum or settings.enabled):
        return None, None
    trajectories = str(args.curriculum or settings.trajectories or TRAJECTORIES_DIR / config.name)
    state_path = out_dir / 'curriculum.json'
    resume = continued / 'curriculum.json' if continued is not None else None
    if resume is not None and resume.is_file() and _same_trajectories(continued, trajectories):
        if resume.resolve() != state_path.resolve():
            shutil.copyfile(resume, state_path)
        print(f"[train] curriculum resumes {resume}")
    else:
        state_path.unlink(missing_ok=True)
    fraction = settings.fraction if args.curriculum_fraction is None else args.curriculum_fraction
    threshold = settings.threshold if args.curriculum_threshold is None else args.curriculum_threshold
    return CurriculumSpec(trajectories, str(state_path), fraction=fraction, threshold=threshold), trajectories


def main(argv=None) -> Path:
    """Runs training; returns the run directory."""
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parse_args(argv)
    validate_args(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
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
    events, plan = open_events(args, specialist, run_id, out_dir, argv)
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


# A run dir's files from an earlier training (``set_aside_previous``); train.log belongs to the caller
# (scripts/train_round.py tees the current run's output into it).
PREVIOUS_RUN_FILES = ('ckpt.pt', 'metrics.jsonl', 'train_summary.json', 'act', 'curriculum.json', 'run.json', 'tb')


def set_aside_previous(out_dir: Path, continuing: bool) -> Optional[Path]:
    """Moves an earlier training's files out of ``out_dir`` into ``<run>.prev-N`` (first free N) so the dir never
    claims the old run's status or metrics; returns that dir, or None when nothing was there.  A same-dir
    ``--load`` continuation (``continuing``) keeps its checkpoint, metrics.jsonl (appended to), curriculum and tb;
    its train_summary.json, act and eval files still go (they describe the old weights)."""
    stale = sorted(path for path in out_dir.iterdir()
                   if path.name in ('train_summary.json', 'act')
                   or (path.name.startswith('eval') and path.suffix == '.json')
                   or (not continuing and path.name in PREVIOUS_RUN_FILES))
    if not stale:
        return None
    n = 1
    while (aside := out_dir.with_name(f"{out_dir.name}.prev-{n}")).exists():
        n += 1
    aside.mkdir()
    for path in stale:
        path.rename(aside / path.name)
    return aside


def harvested_source(args, config) -> dict:
    """Where the run's harvested start states would come from (F069): ``dir``, ``states`` (files there now),
    ``fraction`` (``harvested_fraction``) and ``drawn`` (whether and how resets draw them)."""
    states = len(config.resolve_harvested_states(args.states_dir))
    if args.start_state is not None:
        drawn = "not drawn: --start-state pins every reset"
    elif not config.resolve_start_states(args.states_dir):
        drawn = "drawn on every reset: the specialist has no seed start states"
    elif config.harvested_fraction > 0:
        drawn = f"drawn on {config.harvested_fraction:.0%} of the resets (harvested_fraction)"
    else:
        drawn = "not drawn: harvested_fraction is 0"
    return {'dir': config.harvested_dir(args.states_dir), 'states': states,
            'fraction': config.harvested_fraction, 'drawn': drawn}


def curriculum_tracks(trajectories: Optional[str]) -> Optional[Dict[str, str]]:
    """The curriculum's tracks (``<trajectories>/<track>/trajectory.json``, what the native loader reads) and each
    trajectory.json's modification time (ISO, UTC); None without a curriculum (F131: stale tracks show)."""
    if trajectories is None:
        return None
    return {path.parent.name: datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            .isoformat(timespec='seconds') for path in sorted(Path(trajectories).glob('*/trajectory.json'))}


def _train(args, specialist: str, run_id: str, root, out_dir: Path, events: EventLog) -> dict:
    """The training run proper; returns the train_end event fields."""
    # pylint: disable=too-many-locals, too-many-statements
    config = load_specialist(args.specialist)
    network, optimizer_state, start_iteration, start_steps, source = initial_model(args, config, root)
    continued = continued_run_dir(args, source, start_iteration)
    aside = set_aside_previous(out_dir, continuing=continued == out_dir.resolve())
    if aside is not None:
        print(f"[train] moved the previous files of {out_dir} to {aside}")
    curriculum, trajectories = curriculum_spec(args, config, out_dir, continued)
    task = TrainTask(config, str(out_dir), start_state=args.start_state, curriculum=curriculum,
                     states_dir=args.states_dir)

    ppo_config = PPOConfig(target_steps=args.steps_per_update, learning_rate=args.learning_rate,
                           ent_anneal=args.ent_anneal, shaping_anneal=args.shaping_anneal,
                           sil_coeff=args.sil_coeff, sil_value_coeff=args.sil_value_coeff,
                           sil_capacity=args.sil_capacity, search=args.search, search_coeff=args.search_coeff,
                           plr=args.plr, plr_beta=args.plr_beta, plr_rho=args.plr_rho, plr_uniform=args.plr_uniform,
                           ppg=args.ppg, ppg_every=args.ppg_every, ppg_aux_epochs=args.ppg_aux_epochs,
                           ppg_clone_coeff=args.ppg_clone_coeff)
    ppo = PPO(ppo_config, device=args.device, optimizer_state=optimizer_state)

    # Built before the 'start' event so bad start states or trajectories fail the run up front.
    pool_seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(8), 'little') >> 1
    pool = EnvPool(task, num_envs=args.parallel, threads=args.threads, seed=pool_seed)

    harvested = harvested_source(args, config)
    tracks = curriculum_tracks(trajectories)
    with open(out_dir / 'run.json', 'w', encoding='utf-8') as file:
        json.dump({'args': vars(args), 'specialist': specialist, 'run_id': run_id, 'source': source,
                   'parameters': network.parameter_count(), 'model_config': network.config.to_dict(),
                   'ppo_config': ppo_config.to_dict(), 'start_iteration': start_iteration,
                   'start_steps': start_steps, 'curriculum': trajectories, 'curriculum_tracks': tracks,
                   'harvested': harvested}, file, indent=2, default=str)

    print(f"[train] {specialist} run={run_id} dir={out_dir}")
    print(f"[train] parameters={network.parameter_count():,} "
          f"parallel={args.parallel} threads={args.threads} steps/update={steps_per_iteration(args)} "
          f"device={args.device}"
          + (f" from={source}" if source else ""))
    if steps_per_iteration(args) != args.steps_per_update:
        print(f"[train] note: --steps-per-update {args.steps_per_update} is not a multiple of --parallel "
              f"{args.parallel}; each iteration trains {steps_per_iteration(args)} steps")
    if tracks is not None:
        print(f"[train] curriculum {trajectories}: {len(tracks)} tracks "
              + (', '.join(f"{name} ({mtime})" for name, mtime in tracks.items()) or '(none)'))
    if harvested['states']:
        print(f"[train] warning: {harvested['states']} harvested start states in {harvested['dir']}: "
              f"{harvested['drawn']}", flush=True)
    target_steps = start_steps + args.iterations * steps_per_iteration(args)
    events.emit('start', target_iterations=start_iteration + args.iterations, target_steps=target_steps,
                start_iteration=start_iteration, start_steps=start_steps, steps_per_update=steps_per_iteration(args),
                parallel=args.parallel, threads=args.threads, device=args.device,
                curriculum=trajectories, curriculum_tracks=tracks, harvested=harvested, source=source,
                stop_success=args.stop_success, stop_patience=args.stop_patience, run_dir=str(out_dir),
                pid=os.getpid(), set_aside=str(aside) if aside is not None else None)

    logger = MetricsLogger(str(out_dir))
    state = {'iteration': start_iteration, 'steps': start_steps, 'metrics': {}, 'saved': None,
             'stopped_early': False}
    steps_per_env = args.steps_per_update // args.parallel
    early = EarlyStop(args.stop_success, args.stop_patience,
                      min_iterations=math.ceil(config.max_steps / steps_per_env) + args.stop_patience,
                      seed_states=pool.start_states or None)    # empty: one --start-state pins every reset
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

    best_metrics = refresh_best(0) if args.best_of_n > 0 else {}

    def save():
        registry.save_checkpoint(specialist, run_id, network, ppo.optimizer, iterations=state['iteration'],
                                 steps=state['steps'], metrics=state['metrics'], root=root)
        state['saved'] = state['iteration']
        events.emit('checkpoint', iteration=state['iteration'], steps=state['steps'], path=str(ckpt))

    def confirm_stop(iteration: int) -> bool:
        """An early-stop proposal: evaluates the current weights (``CONFIRM_EPISODES`` sampled episodes per seed
        start state); stops only if their success >= --stop-success, else the streak starts over.  The evaluation
        ends as soon as its failures rule the threshold out (``aborted``): the decision is the full one's."""
        seed = pool_seed + start_iteration + iteration
        states_dir = None if args.states_dir is None else str(args.states_dir)
        planned = CONFIRM_EPISODES * len(config.resolve_start_states(states_dir))
        failures, aborted = 0, False

        def hopeless(record: dict) -> bool:
            nonlocal failures, aborted
            failures += not record['success']
            aborted = (planned - failures) / planned < args.stop_success
            return aborted

        by_state = success_by_state(config, network.to_native(), CONFIRM_EPISODES, seed=seed,
                                    parallel=args.parallel, threads=args.threads, states_dir=args.states_dir,
                                    stop=hopeless)
        successes = sum(s for s, _ in by_state.values())
        episodes = sum(e for _, e in by_state.values())
        rate = successes / episodes if episodes else 0.0
        # An aborted evaluation's rate is partial and below its upper bound, which is below the threshold.
        decision = 'stop' if episodes and not aborted and rate >= args.stop_success else 'continue'
        upper = (planned - (episodes - successes)) / planned if planned else 0.0
        print(f"[train] early stop check: evaluation success {rate:.3f} over {episodes} of {planned} episodes"
              + (f" (aborted: at most {upper:.3f} reachable)" if aborted else '') + f" -> {decision}", flush=True)
        events.emit('early_stop_check', iteration=start_iteration + iteration, rate=rate, episodes=episodes,
                    planned=planned, failures=episodes - successes, aborted=aborted, rate_upper_bound=upper,
                    seed=seed, balanced_success_rate=early.balanced, train_rates=early.rates(),
                    states={state: s / e for state, (s, e) in sorted(by_state.items()) if e}, decision=decision)
        if decision != 'stop':
            early.reset()
        return decision == 'stop'

    def on_iteration(result: IterationResult):
        state['iteration'] = start_iteration + result.iteration
        state['steps'] = result.total_steps
        metrics = {**result.rollout.summary(), **result.stats, **best_metrics}
        best_metrics.clear()
        if args.best_of_n > 0 and result.iteration % args.best_every == 0 and result.iteration < args.iterations:
            best_metrics.update(refresh_best(result.iteration // args.best_every))
        stop = early.update([counts for _, counts in result.rollout.cohorts], result.rollout.episodes,
                            metrics.get('in_flight/over_half_budget'))
        if early.balanced is not None:
            metrics['balanced_success_rate'] = early.balanced
        if early.harvested[1]:
            metrics['harvested/episodes'] = early.harvested[1]
            metrics['harvested/success_rate'] = early.harvested[0] / early.harvested[1]
        if stop:
            stop = confirm_stop(result.iteration)
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
                  "iterations, confirmed by an evaluation", flush=True)
            events.emit('early_stop', iteration=state['iteration'], steps=state['steps'],
                        balanced_success_rate=early.balanced)
            state['stopped_early'] = True
            return False
        return True

    started = time.time()
    status = 'completed'
    try:
        ppo.train(network, pool, args.iterations, callback=on_iteration, start_steps=start_steps,
                  seed=args.seed, episode_budget=config.max_steps)
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
    write_json_atomic(out_dir / 'train_summary.json', {
        'status': status, 'iterations': state['iteration'], 'steps': state['steps'],
        'iterations_trained': state['iteration'] - start_iteration, 'steps_trained': trained,
        'stopped_early': state['stopped_early'], 'wall_seconds': elapsed, 'steps_per_second': sps})
    print(f"[train] {'interrupted' if status == 'interrupted' else 'done'}: {state['iteration'] - start_iteration} "
          f"iterations, {trained} steps in {elapsed:.1f}s ({sps:.1f} steps/sec); "
          + (f"checkpoint {ckpt}" if ckpt.is_file() else "no checkpoint"))
    summary = {'status': status, 'iterations': state['iteration'],
               'steps': state['steps'], 'wall_seconds': round(elapsed, 1), 'sps': round(sps, 1),
               'checkpoint': str(ckpt) if ckpt.is_file() else None}
    events.emit('train_end', **summary)
    return summary


if __name__ == '__main__':
    sys.exit(cli())
