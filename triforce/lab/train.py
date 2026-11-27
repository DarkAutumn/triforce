"""Lab training: one network over several families at once, with today's networks as teachers (docs/lab.md).

    python -m triforce.lab.train --name zol-skg --families zol@d3first stalfos-keese-gel@d3first \\
        --state-route d3first --parallel 8 --iterations 80 --teacher-coeff 1.0 --teacher-hold 20 --teacher-fade 60

Each ``--families`` item is a policy key, optionally with its task yaml (``zol@d3first=path/zol.yaml``); a key's
own yaml stays its task definition.  The run lives in ``<models>/<name>/<run-id>`` and every member family gets a
link ``<models>/<family>/<run-id>`` to it, so ``promote``, ``evaluate``, ``eval-chain``, ``select-checkpoint`` and
``triforce list`` treat it as each family's run (``runner.Policies`` loads it once).

Modes: ``ppo`` (default; the teacher term KL(teacher ‖ student) beside PPO with ``--teacher-coeff``, held
``--teacher-hold`` iterations, then fading linearly to 0 over ``--teacher-fade``) and ``imitate`` (no PPO: the
student minimises the KL alone; ``--actor student`` collects on the student's own states, DAgger-style, the
distribution it will play; ``--actor teacher`` on the teachers', a fixed distribution for capacity tests).
Metrics are logged per family (``family/<key>/...``) beside the merged ones.  The job's log folder (``--logs-dir``)
holds a plan of kind ``lab-train`` and a stream a family (``RunLog``), so the training GUI draws the run: one group,
a row a family.

Also (docs/lab.md): auxiliary targets (``--aux OUTPUT=TARGET:COEFF``) and look-ahead labels for candidates not taken
(``--try-others K:P``, training only); a recurrent network (``LabNet.initial_state``) trains on sequences of
``--bptt`` decisions from stored states; ``--sampling need`` shares the rows by each family's need (``--need-target-for``
a family's own target); ``--teacher-weight`` scales or removes a family's teacher term; ``--rehearse`` keeps tasks
outside the rollout in the loss by their stored teacher rows; demonstrations (``--demos``, and each family task's own
``demos:`` unless ``--no-task-demos``: one pool of rows, one weight), the early stop (``--stop-success``) and numbered
checkpoints (``--checkpoint-every``) as in ``triforce train``.

The generation's teachers (docs/lab.md "Groups of families", "The recipe on record", "Refusals", "Resume"):
``--group NAME`` trains a teacher of ``triforce/groups.yaml`` (its members, its route's states); ``run.json`` holds the
run's manifest (``lab.recipe.build``: ``triforce manifest diff`` reads it) and ``--same-recipe-as RUN --except KEY,...``
refuses a run that differs from RUN in more; the run refuses tasks that disagree on the observation, a network that
reads a block its tasks leave at zero (or does not read one they fill), a task drawing every route's states, a task's
``warm_start``, and demonstrations recorded without a key the network reads (``lab.recipe``); an interrupted run
(SIGTERM, the queue's stop, a crash) is continued with ``--resume NAME/RUN`` and the same command line.
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from .. import lineage, manifest, nets, paths, registry, store
from ..best_of_n import success_by_state
from ..demos import load_demos, recipe as demo_recipe, row_weights
from ..inputs import MissingInput, task_demos
from ..events import EventLog, logged_job, write_json_atomic
from ..jobs import Speed, announce, work
from ..jobspeed import lookahead_text, network_name
from ..jobs import progress as job_progress
from ..nets.base import LabNet
from ..nets.serve import learner_process, serve_mode
from ..metrics import MetricsLogger, format_summary
from ..ppo import PPO, IterationResult, PPOConfig, joint_kl, minibatch_forward
from ..train import (CONFIRM_EPISODES, EXIT_INTERRUPTED, EarlyStop, curriculum_spec, interrupt_on_sigterm,
                     interrupts_held, resolve_warm_start, run_status)
from . import groups
from .auxiliary import AuxLoss, aux_loss, held_out
from .collect import MultiCollector, NeedSampler, Teachers, parse_families, teacher_paths
from .recipe import (Refused, build as build_recipe, check_clips_off, check_demos, check_network_reads,
                     check_obs_agree, check_routes, check_same, check_warm_start, create_network, read_reference,
                     widen_loaded)
from .rehearse import Rehearsal

REFERENCE = Path(__file__).with_name('reference.json')
KIND = 'lab-train'          # the plan kind of a lab run's log folder
GPU_FRACTION_ENV = 'TRIFORCE_GPU_FRACTION'  # a measurement's cap on torch's share of the card (unset: none)
# The run's numbers (besides the family's own) every family's progress event carries: the teacher term and the
# auxiliary losses are the one network's.
SHARED_METRICS = ('losses/teacher_kl', 'charts/teacher_agreement', 'losses/approx_kl', 'losses/policy_loss',
                  'losses/value_loss', 'losses/entropy')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='python -m triforce.lab.train', description=__doc__.split('\n\n')[0])
    parser.add_argument('--name', required=True, help="the run's set name: <models>/<name>/<run-id>")
    parser.add_argument('--families', nargs='+', default=None, metavar='KEY[=YAML]',
                        help="the run's tasks (or --group)")
    parser.add_argument('--group', default=None, metavar='NAME',
                        help="a teacher of triforce/groups.yaml: its members are --families, its route --state-route")
    parser.add_argument('--all-routes', action='store_true',
                        help="let a task without state_routes draw every route's states (refused otherwise without "
                             "--state-route)")
    parser.add_argument('--allow-warm-start', action='store_true',
                        help="train although a task names a warm_start (the lab never loads it)")
    parser.add_argument('--allow-clips-off', action='store_true',
                        help="train although a task plays some episodes with CLIP masked (clips.off_share above 0; "
                             "the owner's decision before a retrain, clips on by default)")
    parser.add_argument('--resume', default=None, metavar='NAME/RUN',
                        help="continue an interrupted run (the same command line): weights, optimizer, counters, "
                             "each task's curriculum progress and the need sampler restored, to the same total")
    parser.add_argument('--same-recipe-as', default=None, metavar='RUN',
                        help="refuse to start unless this run's manifest (triforce.manifest) is RUN's (a run dir or "
                             "a store run NAME/RUN) but for the run's identity and seeds and --except's keys")
    parser.add_argument('--except', dest='except_', type=lambda text: [k for k in text.split(',') if k],
                        default=[], metavar='KEY,...',
                        help="with --same-recipe-as or --resume: the keys this run changes (a key names every "
                             "manifest path whose dotted parts it matches)")
    parser.add_argument('--shares', nargs='*', default=[], metavar='KEY=W', help="env shares (default equal)")
    parser.add_argument('--arch', default=None, choices=nets.names(),
                        help=f"the network's class (default: the group file's with --group, else {nets.ZELDA})")
    parser.add_argument('--arch-config', default=None,
                        help="JSON settings of the architecture (with --group: over the group file's, by key)")
    parser.add_argument('--load', default=None,
                        help="checkpoint or run dir to start from (its arch; a hitnet the run's tokens give more "
                             "groups is widened, their inputs zero: docs/lab.md \"Continuing with a new input\")")
    parser.add_argument('--mode', default='ppo', choices=['ppo', 'imitate'])
    parser.add_argument('--actor', default='student', choices=['student', 'teacher'],
                        help="who acts in the rollouts (imitate only; ppo always the student)")
    parser.add_argument('--teacher-coeff', type=float, default=0.0, help="KL(teacher || student) weight (0: off)")
    parser.add_argument('--teacher-value-coeff', type=float, default=0.0, help="½(V - V_teacher)² weight")
    parser.add_argument('--teacher-hold', type=int, default=0, help="iterations at the full weight")
    parser.add_argument('--teacher-fade', type=int, default=0,
                        help="iterations over which the weight then falls to 0 (0: stays)")
    parser.add_argument('--teacher', nargs='*', default=[], metavar='KEY=CKPT', help="teacher overrides")
    parser.add_argument('--parallel', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--steps-per-update', type=int, default=PPOConfig.target_steps)
    parser.add_argument('--learning-rate', type=float, default=PPOConfig.learning_rate)
    parser.add_argument('--ent-coeff', type=float, default=PPOConfig.ent_coeff)
    parser.add_argument('--epochs', type=int, default=PPOConfig.num_epochs)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--models-dir', default=None)
    parser.add_argument('--team', default=None,
                        help="the store's team teachers and --load names resolve in (NAME, current, "
                             "NAME+SLOT=RUN...); the run is allocated in the store and sealed at the end")
    parser.add_argument('--states-dir', default=None)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--checkpoint-every', type=int, default=0,
                        help="also keep a numbered checkpoint every N iterations (select-checkpoint reads them)")
    parser.add_argument('--checkpoint-keep', type=int, default=0, help="numbered checkpoints kept (0: all)")
    parser.add_argument('--no-curriculum', action='store_true')
    parser.add_argument('--state-route', default=None, metavar='ROUTE',
                        help="every family on this route's states alone (lab.collect.route_only: state_routes, the "
                             "start-state globs the route has, the curriculum tracks explored from its states)")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--aux', nargs='*', default=None, metavar='OUTPUT=TARGET:COEFF',
                        help="a loss on the network's auxiliary output OUTPUT against TARGET (hit, hit_or_blocked, "
                             "damaged, hearts_lost, frames; lab/auxiliary.py) weighted COEFF")
    parser.add_argument('--try-others', default=None, metavar='K:P',
                        help="on a share P of the decisions, try the policy's next K candidates in the env and keep "
                             "whether each lost health and whether a Bubble touched Link (training only; the 'hit' "
                             "and 'hit_or_blocked' targets' extra labels)")
    parser.add_argument('--bptt', type=int, default=16,
                        help="a recurrent network's sequence length (truncated back-propagation through time)")
    parser.add_argument('--sampling', default='fixed', choices=['fixed', 'need'],
                        help="rows per family: fixed (equal decisions an env) or by need (NeedSampler)")
    parser.add_argument('--need-target', default='teacher',
                        help="the success a family needs: 'teacher' (its key's success in lab/reference.json) or a "
                             "number for every family")
    parser.add_argument('--need-floor', type=float, default=0.05, help="a family's weight beside its need")
    parser.add_argument('--need-window', type=int, default=100, help="recent episodes a family's success reads")
    parser.add_argument('--need-target-for', nargs='*', default=[], metavar='KEY=X',
                        help="a family's own need target (overrides --need-target for it)")
    parser.add_argument('--teacher-weight', nargs='*', default=[], metavar='KEY=W',
                        help="weight of a family's rows in the teacher term (default 1; 0: no teacher for it)")
    parser.add_argument('--rehearse', default=None, metavar='CACHE',
                        help="a distillation cache (r36_merge_distill.py cache): KL to its teacher rows of the keys "
                             "not rolled out, every minibatch (lab.rehearse)")
    parser.add_argument('--rehearse-keys', nargs='*', default=None,
                        help="the keys rehearsed (default: every key of the cache not in --families)")
    parser.add_argument('--rehearse-rows', type=int, default=300_000, help="rows held, equal a key")
    parser.add_argument('--rehearse-coeff', type=float, default=1.0)
    parser.add_argument('--demos', nargs='+', default=None, metavar='FILE',
                        help="demonstration files (triforce.demos) behavior-cloned during PPO (--demo-coeff), with the "
                             "families' task demos")
    parser.add_argument('--demo-coeff', type=float, default=None,
                        help="the cloning weight (default: the tasks' demos: coeff, else 0.2)")
    parser.add_argument('--no-task-demos', action='store_true',
                        help="leave out the demonstration files the families' tasks declare (demos:)")
    parser.add_argument('--stop-success', type=float, default=None,
                        help="stop once every family's start-state-balanced training success >= this for "
                             "--stop-patience iterations and an evaluation of each confirms it")
    parser.add_argument('--stop-patience', type=int, default=5)
    parser.add_argument('--logs-dir', default=None,
                        help="the root of the run's log folder (default logs/, $TRIFORCE_LOGS_DIR)")
    args = parser.parse_args(argv)
    for flag, value in (('--name', args.name), ('--run-id', args.run_id)):
        if value and '+' in value:
            parser.error(f"{flag}: '+' separates the slots of a team spec, so no --team could name this run; use '-' "
                         f"or '_'")
    return args


def teacher_coeff(args, iteration: int) -> float:
    """The teacher term's weight at (1-based) ``iteration``: held, then fading linearly to 0."""
    if args.teacher_fade <= 0 or iteration <= args.teacher_hold:
        return args.teacher_coeff
    return args.teacher_coeff * max(0.0, 1.0 - (iteration - args.teacher_hold) / args.teacher_fade)


def initial_network(args, models):
    """(network, source, parent): ``parent`` the loaded checkpoint's identity (``lineage.identity``), None from
    scratch.  ``--load`` resolves as ``triforce train``'s (``train.resolve_warm_start``: a path, a store run
    ``FAMILY/RUN``, a team slot ``family@route``, a family). A loaded ``hitnet`` the run's ``--arch-config`` (or group
    file) gives more token groups is widened to them (``recipe.widen_loaded``: the new inputs start at zero)."""
    if args.load:
        try:
            path = resolve_warm_start(args.load, models)
        except MissingInput as error:
            raise SystemExit(f"--load: {error}") from None
        network, checkpoint = registry.load_model(path)
        network, added = widen_loaded(network, args.arch, json.loads(args.arch_config))
        if added:
            print(f"[lab] --load {args.load}: widened with the token group(s) {', '.join(added)} (their input "
                  "columns zero: it acts exactly as loaded)", flush=True)
        return network, str(path), lineage.identity(checkpoint, path)
    return create_network(args.arch, json.loads(args.arch_config)), None, None


def link_families(root: Path, name: str, run_id: str, keys: list[str]):
    """``<models>/<family>/<run-id>`` -> ``../<name>/<run-id>`` for every member family."""
    for family in sorted({registry.split_policy(k)[0] for k in keys}):
        link = registry.run_dir(family, run_id, root)
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists() and not link.is_symlink():
            link.symlink_to(Path('..') / name / run_id)


def imitate(network, optimizer, batch: dict, cfg: PPOConfig, value_coeff: float, aux_losses=()) -> dict:
    """One pass of the imitation-only update over ``batch`` (``BufferSet.flatten``): KL(teacher ‖ student)
    (+ ``value_coeff`` · ½(V − V_teacher)², + the auxiliary losses) in ``cfg.minibatches`` minibatches,
    ``cfg.num_epochs`` epochs; a recurrent network's in whole sequences (``ppo.minibatch_forward``)."""
    # pylint: disable=too-many-locals
    rows = batch['masks'].shape[0]
    device = batch['masks'].device
    sequence = batch.get('chunk_states')
    length = rows // sequence.shape[0] if sequence is not None else 1
    units = rows // length

    def forward(part):
        mb = (part[:, None] * length + torch.arange(length, device=device)).reshape(-1)
        obs = {k: v[mb] for k, v in batch['obs'].items()}
        out, joint_of = minibatch_forward(network, obs, batch['masks'][mb], batch['actions'][mb],
                                          None if sequence is None else (batch['starts'][mb], sequence[part]))
        return mb, out, joint_of

    network.train()
    kl = torch.tensor(0.0)
    stats: dict = held_out(list(aux_losses), network, batch) if aux_losses and sequence is None else {}
    for _ in range(cfg.num_epochs):
        for part in torch.tensor_split(torch.randperm(units, device=device), cfg.minibatches):
            mb, out, joint_of = forward(part)
            kl = joint_kl(batch['teacher_joint'][mb], joint_of())
            loss = kl
            if value_coeff > 0:
                loss = loss + value_coeff * 0.5 * (out.value - batch['teacher_value'][mb]).pow(2).mean()
            if aux_losses:
                term, aux_stats = aux_loss(list(aux_losses), out.aux, batch, mb)
                stats.update(aux_stats)
                loss = loss + term
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.max_grad_norm)
            optimizer.step()
    network.eval()
    with torch.no_grad():
        agree = []
        for part in torch.split(torch.arange(units, device=device), max(1, 4096 // length)):
            mb, _, joint_of = forward(part)
            agree.append(joint_of().argmax(-1) == batch['teacher_joint'][mb].argmax(-1))
    return {**stats, 'losses/teacher_kl': float(kl.detach()),
            'charts/teacher_agreement': float(torch.cat(agree).float().mean())}


def parse_try_others(text: Optional[str]) -> Optional[dict]:
    """``K:P`` -> ``EnvPool.rollout``'s ``tried`` settings."""
    if text is None:
        return None
    k, _, share = text.partition(':')
    try:
        tried = {'candidates': int(k), 'share': float(share or 1.0)}
    except ValueError as err:
        raise SystemExit(f"--try-others {text!r}: K:P (candidates, share of decisions)") from err
    if tried['candidates'] < 1 or not 0 < tried['share'] <= 1:
        raise SystemExit(f"--try-others {text!r}: K >= 1 and P in (0, 1]")
    return tried


class RunLog:
    """The run's log folder (``logs/<yymmdd>_<NNN>_lab-<name>``, docs/lab.md "What a lab job writes"): the plan
    ``training.json`` (kind ``KIND``, one run entry a family) and one stream a family, ``<family>.jsonl``, in
    ``triforce train``'s events (``start``, a ``progress`` an iteration, ``train_end``, ``end``): the iteration
    of N for every family alike, and in ``metrics`` the family's own numbers (``family/<key>/...`` without the
    prefix: success, balanced success, hearts, ``rows_share``, ``hits_taken_per_episode``, need) with the run's
    teacher coefficient (``teacher_coeff``) and losses (``SHARED_METRICS``, ``losses/aux_*``).  The training GUI
    draws the run as one group with a row a family."""

    def __init__(self, plan, keys: list[str], run_id: str, shape: tuple = (None, None, None)):
        self.plan = plan
        self.keys = keys
        self.streams = [EventLog(plan.directory / f'{key}.jsonl', specialist=key, run_id=run_id) for key in keys]
        self.began = time.time()
        self.start_steps = 0
        self.shape = shape          # (network, games, look-ahead): ``jobs.Speed``'s
        self.speed: Optional[Speed] = None

    @staticmethod
    def runs(keys: list[str], run_id: str, out_dir: Path, total: int, iterations: int) -> list[dict]:
        """The plan's run entries."""
        return [{'specialist': key, 'run_id': run_id, 'run_dir': str(out_dir), 'events': f'{key}.jsonl',
                 'target_steps': total, 'target_iterations': iterations} for key in keys]

    def start(self, **fields):
        """Every family's ``start`` (``start_steps``: where a resumed run continues; its rate counts from there)."""
        self.began = time.time()
        self.start_steps = int(fields.get('start_steps') or 0)
        total = fields.get('target_steps')
        self.speed = Speed(*self.shape, total=None if total is None else total - self.start_steps, began=self.began)
        for index, stream in enumerate(self.streams):
            stream.emit('start', **fields)
            self.plan.update_run(index, status='running')

    def progress(self, iteration: int, iterations: int, steps: int, target_steps: int, metrics: dict,
                 teacher: float):
        elapsed = time.time() - self.began
        done = steps - self.start_steps
        sps = done / elapsed if elapsed > 0 else 0.0
        eta = (target_steps - steps) / sps if sps else None
        shared = {key: value for key, value in metrics.items()
                  if key in SHARED_METRICS or key.startswith('losses/aux_')}
        job_progress(done, target_steps - self.start_steps, 'steps')
        if self.speed is not None:
            self.speed.update(done, metrics.get('charts/rollout_SPS'))
        for key, stream in zip(self.keys, self.streams):
            prefix = f'family/{key}/'
            own = {name[len(prefix):]: value for name, value in metrics.items() if name.startswith(prefix)}
            stream.emit('progress', iteration=iteration, iterations=iterations, steps=steps,
                        target_steps=target_steps, fraction=steps / target_steps if target_steps else None,
                        elapsed=round(elapsed, 1), sps=round(sps, 1), eta=None if eta is None else round(eta, 1),
                        metrics={**own, **shared, 'teacher_coeff': teacher})

    def end(self, status: str, **fields):
        """Every family's ``train_end`` and ``end`` (``status``: completed | early_stop | interrupted | error)."""
        for index, stream in enumerate(self.streams):
            if status != 'error':
                stream.emit('train_end', status=status, **fields)
            stream.emit('end', status=status, **fields)
            self.plan.update_run(index, status=status)


class FamilyStop:
    """``triforce train``'s early stop (``train.EarlyStop``) for a set of families: each family's own, fed with its
    cohorts; proposed once every family's holds (its balanced success >= the threshold for ``patience``
    iterations), confirmed by an evaluation of each family (``CONFIRM_EPISODES`` episodes a seed start state)
    whose success must reach the threshold too. Without the hearts and bombs plateau rules."""

    def __init__(self, args, collector: MultiCollector, steps_per_env: int):
        self.args = args
        self.collector = collector
        self.stops = {f.key: EarlyStop(args.stop_success, args.stop_patience,
                                       min_iterations=math.ceil(f.config.max_steps / steps_per_env)
                                       + args.stop_patience,
                                       seed_states=pool.start_states or None)
                      for f, pool in zip(collector.families, collector.pools)}

    def update(self, network, iteration: int, metrics: dict) -> bool:
        """One iteration; adds ``family/<key>/balanced_success_rate`` to ``metrics``; True: stop."""
        proposed = True
        for key, stop in self.stops.items():
            rollout = self.collector.by_family[key]
            proposed &= stop.update([counts for _, counts in rollout.cohorts], rollout.episodes,
                                    rollout.summary().get('in_flight/over_half_budget'))
            if stop.balanced is not None:
                metrics[f'family/{key}/balanced_success_rate'] = stop.balanced
        if not proposed:
            return False
        rates = {}
        for family in self.collector.families:
            by_state = success_by_state(family.config, network.to_native(), CONFIRM_EPISODES,
                                        seed=(self.args.seed or 0) + iteration, parallel=self.args.parallel,
                                        states_dir=self.args.states_dir, act=registry.DEFAULT_ACT)
            episodes = sum(e for _, e in by_state.values())
            rates[family.key] = sum(s for s, _ in by_state.values()) / episodes if episodes else 0.0
        confirmed = all(rate >= self.args.stop_success for rate in rates.values())
        print(f"[lab] early stop check at iteration {iteration}: "
              + ' '.join(f'{k}={r:.3f}' for k, r in rates.items()) + (' -> stop' if confirmed else ' -> continue'),
              flush=True)
        if not confirmed:
            for stop in self.stops.values():
                stop.reset()
        return confirmed


# A resumed run's pools are seeded ``seed + RESUME_SEED_STRIDE × the iteration it resumes at``: fresh episodes, not the
# first iterations' again (a run that never stopped: ``seed``, as before).
RESUME_SEED_STRIDE = 1_000_003


def expand_group(args) -> Optional[groups.Group]:
    """``--group NAME``: its members become ``--families`` and its route ``--state-route`` (refused beside
    ``--families`` or another ``--state-route``); without it ``--families`` names the tasks."""
    if args.group is None:
        if not args.families:
            raise SystemExit("name the run's tasks: --families KEY[=YAML]... or --group NAME (triforce/groups.yaml)")
        return None
    if args.families:
        raise SystemExit(f"--group {args.group} and --families: one or the other (a group's members are its tasks)")
    group = groups.get(args.group)
    if args.state_route not in (None, group.route):
        raise SystemExit(f"--group {group.name} plays route {group.route}'s states, not --state-route "
                         f"{args.state_route}")
    args.families, args.state_route = list(group.members), group.route
    return group


def apply_network(args, group: Optional[groups.Group]) -> Optional[dict]:
    """The run's ``--arch``, ``--arch-config``, ``--aux``, ``--try-others``: what the command line gives, else the
    group file's (``group.network``), else the lab's defaults.  A command-line ``--arch-config`` overrides the
    group's by key (all of it for another ``--arch``).  With a group: prints one line naming every key the command
    line changes from the file's, and returns each key's source (``group file``, ``command line``, ``default``) for the
    manifest; without one: None, and the arguments as given."""
    given = {'arch': args.arch, 'aux': args.aux, 'try_others': args.try_others,
             'arch_config': None if args.arch_config is None else json.loads(args.arch_config)}
    if group is None:
        args.arch = args.arch or nets.ZELDA
        args.arch_config = '{}' if args.arch_config is None else args.arch_config
        args.aux = [] if args.aux is None else args.aux
        return None
    pinned = group.network
    sources: dict = {}
    changed: list[str] = []

    def pick(key, default):
        if given[key] is None:
            sources[key] = 'group file' if key in pinned else 'default'
            return pinned.get(key, default)
        sources[key] = 'command line'
        if key in pinned and given[key] != pinned[key]:
            changed.append(f"{key} {json.dumps(pinned[key])} -> {json.dumps(given[key])}")
        return given[key]

    args.arch = pick('arch', nets.ZELDA)
    same_arch = pinned.get('arch', args.arch) == args.arch
    base = dict(pinned.get('arch_config') or {}) if same_arch else {}
    if given['arch_config'] is None:
        sources['arch_config'] = 'group file' if base else 'default'
        config = base
    else:
        sources['arch_config'] = 'command line'
        config = {**base, **given['arch_config']}
        changed += [f"arch_config.{key} {json.dumps(base[key]) if key in base else '(unset)'} -> "
                    f"{json.dumps(value)}" for key, value in given['arch_config'].items() if base.get(key) != value]
    if not same_arch and pinned.get('arch_config'):
        changed.append(f"arch_config: the file's is {pinned['arch']}'s, not used")
    args.arch_config = json.dumps(config, sort_keys=True)
    args.aux = list(pick('aux', []))
    args.try_others = pick('try_others', None)
    if changed:
        print(f"[lab] --group {group.name}: the command line overrides the group file's " + '; '.join(changed),
              flush=True)
    return sources


@dataclass
class Resume:
    """The run ``--resume`` continues: its folder, its store run (None in a models folder), its ``run.json`` and its
    checkpoint (None: interrupted before its first save; it starts over)."""
    out_dir: Path
    stored: Optional[store.Run]
    record: dict
    checkpoint: Optional[Path]


def find_resume(args, team, models) -> Resume:
    """``--resume NAME/RUN``: an open store run of this ``--name`` (a store team plays), else the models folder's
    ``<models>/<name>/<run>``, with the recipe it recorded and not ended (a completed or early-stopped run is
    continued into a new run with ``--load``)."""
    name, _, run_id = args.resume.strip('/').rpartition('/')
    if not name or not run_id:
        raise SystemExit(f"--resume {args.resume}: name the run NAME/RUN")
    if name != args.name:
        raise SystemExit(f"--resume {args.resume}: a run of {name}, not of --name {args.name}")
    if args.run_id not in (None, run_id):
        raise SystemExit(f"--resume {args.resume} and --run-id {args.run_id} name two runs")
    stored = None
    if team is not None:
        stored = store.Run(name, run_id)
        row = store.run_row(stored)
        if row is None:
            raise SystemExit(f"--resume {args.resume}: no such run in the store")
        if row['status'] != store.OPEN:
            raise SystemExit(f"--resume {args.resume}: the run is sealed (it ended); continue it into a new run with "
                             f"--load {args.resume}")
        out_dir = stored.path
    else:
        out_dir = registry.run_dir(name, run_id, models)
    try:
        record = json.loads((out_dir / 'run.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise SystemExit(f"--resume {args.resume}: no run.json in {out_dir} ({error}): nothing recorded to resume; "
                         f"start the run again") from None
    if not record.get('manifest'):
        raise SystemExit(f"--resume {args.resume}: {out_dir / 'run.json'} records no recipe (a lab run from before "
                         f"manifests): start the run again")
    status = run_status(out_dir)
    if status in ('completed', 'early_stop'):
        raise SystemExit(f"--resume {args.resume}: the run ended ({status}); continue it into a new run with "
                         f"--load {out_dir}")
    ckpt = out_dir / registry.CHECKPOINT_NAME
    return Resume(out_dir, stored, record, ckpt if ckpt.is_file() else None)


def keep_metrics(out_dir: Path, iteration: int):
    """``metrics.jsonl`` cut back to the iterations a resumed checkpoint holds (none: removed)."""
    path = out_dir / 'metrics.jsonl'
    if not path.is_file():
        return
    kept = [line for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip() and json.loads(line).get('iteration', 0) <= iteration]
    if kept:
        path.write_text('\n'.join(kept) + '\n', encoding='utf-8')
    else:
        path.unlink()


def training_settings(args, cfg: PPOConfig, tried, steps_per_env: int, recurrent: bool, teacher_weights: dict) -> dict:
    """The manifest's ``training``: what the run trains with besides its tasks, data and networks."""
    teacher = None
    if args.teacher_coeff > 0 or args.mode == 'imitate':
        teacher = {'coeff': args.teacher_coeff, 'hold': args.teacher_hold, 'fade': args.teacher_fade,
                   'value_coeff': args.teacher_value_coeff, 'weights': teacher_weights}
    sampling = {'kind': args.sampling}
    if args.sampling == 'need':
        sampling.update(target=args.need_target, floor=args.need_floor, window=args.need_window,
                        target_for=sorted(args.need_target_for))
    rehearse = None
    if args.rehearse:
        cache = Path(args.rehearse) / 'cache.json'
        rehearse = {'cache': manifest.label(args.rehearse), 'cache_json_sha256': manifest.file_sha256(cache),
                    'keys': args.rehearse_keys, 'rows': args.rehearse_rows, 'coeff': args.rehearse_coeff}
    return {'mode': args.mode, 'actor': args.actor if args.mode == 'imitate' else 'student',
            'iterations': args.iterations, 'parallel': args.parallel, 'steps_per_env': steps_per_env,
            'ppo': cfg.to_dict(), 'aux': list(args.aux), 'try_others': tried,
            'bptt': args.bptt if recurrent else None, 'teacher': teacher, 'sampling': sampling, 'rehearse': rehearse,
            'stop': None if args.stop_success is None else {'success': args.stop_success,
                                                            'patience': args.stop_patience},
            'curriculum': not args.no_curriculum,
            # How the rollouts' network is served (``nets.serve.serve_mode``: CUDA graphs since Nov 25, round40.md
            # "Where a teacher's time goes"): the same actions within float tolerance; a manifest without it is an
            # older run, served eagerly (a resume only notes it: unknown there).
            'serve': serve_mode()}


def serving_text(network, served: dict) -> str:
    """How the run's rollouts get their actions, for its log's start (``[lab] serving: ...``)."""
    if not isinstance(network, LabNet):
        return "the Rust actor plays the network (nothing served)"
    window = f"{served['gather'] * 1000:g} ms batch window"
    if served['forward'] == 'graphs':
        extra = ''.join(part for part, on in ((', one staged copy a batch', served.get('staged')),
                                              (', no hit logits answered', served.get('hits') is False)) if on)
        return (f"CUDA graphs on {served['device']}, each batch size captured at the first rollout{extra} "
                f"({window}; TRIFORCE_SERVE_GRAPHS=0 serves eagerly)")
    return f"the eager forward on {served['device']} ({window})"


def cli(argv=None) -> int:
    """``python -m triforce.lab.train`` (a job: ``triforce job run -- ...``): trains, then exits 0, ``EXIT_INTERRUPTED``
    when the run was interrupted (SIGTERM, the queue's stop: its checkpoint and summary saved, ``--resume`` continues
    it) or stopped before training started, 2 when refused (``recipe.Refused``: the message says what to change)."""
    interrupt_on_sigterm()
    try:
        out_dir = main(argv)
    except KeyboardInterrupt:
        print("[lab] stopped before training started", flush=True)
        return EXIT_INTERRUPTED
    except Refused as error:
        print(f"python -m triforce.lab.train: refused: {error}", file=sys.stderr, flush=True)
        return 2
    return EXIT_INTERRUPTED if run_status(out_dir) == 'interrupted' else 0


def main(argv=None) -> Path:
    # pylint: disable=too-many-locals, too-many-statements, too-many-branches
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parse_args(argv)
    # Its learner uses the GPU beside the rollouts' server: the server captures its CUDA graphs only during the first
    # rollout, while the learner waits for it (``nets.serve.learner_process``).
    learner_process(graphs=True)
    if os.environ.get(GPU_FRACTION_ENV) and torch.cuda.is_available():
        # A measurement's cap (round40.md "Where a teacher's time goes"): torch's allocator stops at this share of
        # the card, so the suites beside it keep their memory; an allocation past it fails as out of memory.
        torch.cuda.set_per_process_memory_fraction(float(os.environ[GPU_FRACTION_ENV]))
    if args.mode == 'imitate' and args.teacher_coeff <= 0:
        args.teacher_coeff = 1.0
    if args.actor == 'teacher' and args.mode != 'imitate':
        raise SystemExit("--actor teacher is for --mode imitate (PPO trains on the student's own samples)")
    group = expand_group(args)
    sources = apply_network(args, group)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    registry.use_team(args.team, args.models_dir)
    models = registry.models_root(args.models_dir)
    team = registry.team_in_use(args.models_dir)
    resume = find_resume(args, team, models) if args.resume else None
    shares = {**(group.shares if group is not None else {}),
              **{k: float(w) for k, _, w in (s.partition('=') for s in args.shares)}}
    families = parse_families(args.families, args.parallel, shares, args.state_route,
                              curriculum=not args.no_curriculum, states_dir=args.states_dir)
    keys = [f.key for f in families]
    check_routes(families, args.state_route, args.all_routes)
    check_warm_start(families, args.allow_warm_start)
    check_clips_off(families, args.allow_clips_off)
    check_obs_agree(families)
    try:
        declared = [] if args.no_task_demos else [task_demos(family.config) for family in families]
    except MissingInput as error:
        raise SystemExit(str(error)) from None
    missing = [path for path in args.demos or () if not Path(path).is_file()]
    if missing:
        raise SystemExit(f"--demos {', '.join(missing)}: no such file")
    args.demos, args.demo_coeff, demo_groups = demo_recipe(declared, args.demos, args.demo_coeff)
    steps_per_env = max(1, args.steps_per_update // args.parallel)
    total = args.iterations * steps_per_env * args.parallel

    # The network: a resumed run's own checkpoint (weights, optimizer, counters, the need sampler), else --load's,
    # else a fresh one of --arch.
    start_iteration = start_steps = 0
    optimizer_state, lab_state = None, {}
    if resume is not None and resume.checkpoint is not None:
        network, checkpoint = registry.load_model(resume.checkpoint)
        source, parent = resume.record.get('source'), resume.record.get('parent')
        start_iteration, start_steps = int(checkpoint['iterations']), int(checkpoint['steps'])
        optimizer_state, lab_state = checkpoint.get('optimizer'), checkpoint.get('lab') or {}
        if start_iteration >= args.iterations:
            raise SystemExit(f"--resume {args.resume}: its checkpoint is at iteration {start_iteration} of "
                             f"{args.iterations}: nothing left to train")
    else:
        network, source, parent = initial_network(args, models)
    check_network_reads(network, families)
    check_demos(network, families, args.demos)
    tried = parse_try_others(args.try_others)
    recurrent = getattr(network, 'recurrent', False)
    if recurrent and steps_per_env % args.bptt:
        raise SystemExit(f"steps per env ({steps_per_env}) must be a multiple of --bptt ({args.bptt})")
    if args.demos and (recurrent or args.mode != 'ppo'):
        raise SystemExit("--demos is for --mode ppo with a network without memory (demonstration rows are no "
                         "sequences)")
    recorded_seed = (resume.record['manifest'].get('seeds') or {}).get('seed') if resume is not None else None
    if resume is not None and args.seed is not None and args.seed != recorded_seed:
        raise SystemExit(f"--resume {args.resume}: the run's seed is {recorded_seed}, not --seed {args.seed}")
    seed = (recorded_seed if recorded_seed is not None else args.seed if args.seed is not None
            else int.from_bytes(os.urandom(4), 'little'))
    pool_seed = seed + RESUME_SEED_STRIDE * start_iteration
    cfg = PPOConfig(target_steps=steps_per_env * args.parallel, learning_rate=args.learning_rate,
                    ent_coeff=args.ent_coeff, num_epochs=args.epochs,
                    demo_bc_coeff=args.demo_coeff if args.demos else 0.0)
    wants_teachers = args.teacher_coeff > 0 or args.mode == 'imitate'
    overrides = dict(item.split('=', 1) for item in args.teacher)
    weights = {k: float(w) for k, _, w in (item.partition('=') for item in args.teacher_weight)}
    unknown = sorted(set(weights) - set(keys))
    if wants_teachers and unknown:
        raise SystemExit(f"--teacher-weight {unknown}: not in --families")
    teacher_files = ({key: str(path) for key, path in teacher_paths(keys, models, overrides, args.state_route).items()}
                     if wants_teachers else None)
    recipe = build_recipe(args, families, network, group=group, envs={f.key: f.envs for f in families},
                          source=source, teachers=teacher_files, demos=args.demos or (), demo_groups=demo_groups,
                          training=training_settings(args, cfg, tried, steps_per_env, recurrent, weights),
                          seeds={'seed': seed, 'pools': pool_seed}, sources=sources)
    if args.same_recipe_as:
        check_same(recipe, read_reference(args.same_recipe_as), args.except_,
                   f"--same-recipe-as {args.same_recipe_as}")
    if resume is not None:
        check_same(recipe, resume.record['manifest'], args.except_, f"--resume {args.resume} (its recorded recipe)")
    announce('train', work(train_steps=total - start_iteration * steps_per_env * args.parallel), f'lab-{args.name}')

    stored = None
    if resume is not None:
        run_id, root = resume.out_dir.name, resume.out_dir.parent.parent
        if resume.stored is not None:
            stored = store.resume_run(resume.stored)
        print(f"[lab] resuming {args.resume} at iteration {start_iteration} of {args.iterations} ({start_steps} "
              f"steps)" if resume.checkpoint is not None else
              f"[lab] resuming {args.resume}: no checkpoint was saved; it starts over", flush=True)
    elif team is not None:      # a store team plays: the run is the store's, sealed when the training ends
        stored = store.allocate_run(args.name, args.run_id, origin='lab', team=team)
        run_id, root = stored.name, paths.runs_dir()
        print(f"[lab] store run {stored.identity} (open until the training ends)", flush=True)
    else:
        run_id, root = args.run_id or registry.new_run_id(), models
    out_dir = registry.run_dir(args.name, run_id, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    identity = f'{args.name}/{run_id}'
    # run.json before anything trains, so a run stopped at any point is one --resume accepts.
    if resume is None:
        run = {'args': {key: value for key, value in vars(args).items() if key != 'team' or value},
               'command': ['python', '-m', 'triforce.lab.train', *argv], 'name': args.name, 'run_id': run_id,
               'group': None if group is None else group.name, 'families': keys,
               'envs': {f.key: f.envs for f in families}, 'arch': nets.arch_of(network.config.to_dict()),
               'parameters': network.parameter_count(), 'model_config': network.config.to_dict(),
               'ppo_config': cfg.to_dict(), 'source': source, 'parent': parent, 'teachers': teacher_files,
               'teacher_ids': None, 'demos': manifest.demo_entries(args.demos or (), demo_groups),
               'manifest': recipe}
    else:
        run = resume.record
        entry = {'iteration': start_iteration, 'steps': start_steps, 'job': recipe['job'],
                 'commit': recipe['code']['commit'], 'seeds': recipe['seeds'], 'time': round(time.time(), 1)}
        summary_path = out_dir / 'train_summary.json'
        if summary_path.is_file():
            entry['previous'] = json.loads(summary_path.read_text(encoding='utf-8'))
            summary_path.unlink()
        run.setdefault('resumes', []).append(entry)
        keep_metrics(out_dir, start_iteration)
    write_json_atomic(out_dir / 'run.json', run, default=str)
    if stored is None and resume is None:      # a store run's families find it through a team's slots, not links
        link_families(root, args.name, run_id, keys)
    with logged_job(f'lab-{args.name}', KIND, RunLog.runs(keys, run_id, out_dir, total, args.iterations),
                    args.logs_dir, work=work(train_steps=total), families=keys, run_dir=str(out_dir),
                    mode=args.mode) as plan:
        interrupt_on_sigterm()  # SIGTERM waits out a TensorBoard write (interrupts_held); logged_job restores it
        served = recipe['training']['serve']['forward'] if isinstance(network, LabNet) else None
        log = RunLog(plan, keys, run_id, (network_name(network.config.to_dict(), served),
                                          sum(f.envs for f in families), lookahead_text(tried)))
        aux_losses = [AuxLoss.parse(text) for text in args.aux]
        sampler = None
        if args.sampling == 'need':
            targets = (NeedSampler.teacher_targets(keys, REFERENCE) if args.need_target == 'teacher'
                       else {key: float(args.need_target) for key in keys})
            for key, _, value in (item.partition('=') for item in args.need_target_for):
                if key not in targets:
                    raise SystemExit(f"--need-target-for {key}: not in --families")
                targets[key] = float(value)
            sampler = NeedSampler(targets, args.need_floor, args.need_window)
            if lab_state.get('sampler'):
                sampler.restore(lab_state['sampler'])
        teachers = None
        if wants_teachers:
            teachers = Teachers(keys, args.device, models, overrides, weights, args.state_route)
            run['teacher_ids'] = teachers.identities
            write_json_atomic(out_dir / 'run.json', run, default=str)
        rehearsal = None
        if args.rehearse:
            names = json.loads((Path(args.rehearse) / 'cache.json').read_text(encoding='utf-8'))['keys']
            rehearsed = args.rehearse_keys if args.rehearse_keys is not None else [k for k in names
                                                                                   if k not in keys]
            rehearsal = Rehearsal(Path(args.rehearse), rehearsed, args.rehearse_rows, seed=args.seed or 0)
            print(f"[lab] rehearsal: {rehearsal.rows:,} rows of {len(rehearsed)} keys from {args.rehearse} "
                  f"(weight {args.rehearse_coeff})", flush=True)
        curricula = {}
        continuing = resume is not None and resume.checkpoint is not None
        for family in families:
            task_dir = out_dir / 'tasks' / family.key
            # A resumed run keeps each task's curriculum progress (<run>/tasks/<key>/curriculum.json, written on
            # every change); a run started (or started over) begins it fresh.
            spec, _ = curriculum_spec(argparse.Namespace(no_curriculum=args.no_curriculum, curriculum=None,
                                                         curriculum_fraction=None, curriculum_threshold=None,
                                                         curriculum_unrecorded='resume' if continuing else None),
                                      family.config, task_dir, continued=task_dir if continuing else None)
            curricula[family.key] = spec
        (out_dir / 'tasks').mkdir(exist_ok=True)
        for family in families:
            (out_dir / 'tasks' / family.key).mkdir(parents=True, exist_ok=True)
        ppo = PPO(cfg, device=args.device, optimizer_state=optimizer_state)
        ppo.aux_losses = aux_losses
        ppo.rehearsal = rehearsal
        ppo.rehearsal_coeff = args.rehearse_coeff
        if args.demos:
            ppo.demo_batch = load_demos(args.demos)
            ppo.demo_weights = row_weights(args.demos, demo_groups)
            drawn = 'every row alike' if demo_groups is None else f'{len(set(demo_groups))} groups, equal shares'
            print(f"[lab] demonstrations: {ppo.demo_batch[2].shape[0]} rows from {', '.join(args.demos)} "
                  f"(weight {args.demo_coeff}; {drawn})", flush=True)
        collector = MultiCollector(families, out_dir, steps_per_env, cfg.gamma, cfg.lam, seed=pool_seed,
                                   states_dir=args.states_dir, curricula=curricula, teachers=teachers,
                                   actor=args.actor if args.mode == 'imitate' else 'student', tried=tried,
                                   chunk_length=args.bptt, sampler=sampler)
        stopper = FamilyStop(args, collector, steps_per_env) if args.stop_success is not None else None
        arch = nets.arch_of(network.config.to_dict())
        print(f"[lab] {args.name} run={run_id} dir={out_dir} arch={arch} "
              f"parameters={network.parameter_count():,} envs={collector.shares} mode={args.mode}", flush=True)
        print(f"[lab] serving: {serving_text(network, recipe['training']['serve'])}", flush=True)
        logger = MetricsLogger(str(out_dir))
        state = {'iteration': start_iteration, 'steps': start_steps, 'metrics': {}, 'stopped_early': False,
                 'rollout_steps': 0.0, 'rollout_seconds': 0.0}

        def save():
            registry.save_checkpoint(args.name, run_id, network, ppo.optimizer, iterations=state['iteration'],
                                     steps=state['steps'], metrics=state['metrics'], root=root,
                                     extra={'families': keys, 'lab': {
                                         'mode': args.mode, 'source': source,
                                         'sampler': None if sampler is None else sampler.state()}})

        def on_iteration(result: IterationResult):
            iteration = start_iteration + result.iteration
            state['iteration'], state['steps'] = iteration, result.total_steps
            metrics = {**result.rollout.summary(), **result.stats, **collector.serve_stats,
                       **collector.family_metrics()}
            if torch.cuda.is_available():       # the learner's and the served copy's memory (torch's cache, graphs)
                metrics['charts/gpu_reserved_gb'] = torch.cuda.memory_reserved() / 2 ** 30
                metrics['charts/gpu_peak_reserved_gb'] = torch.cuda.max_memory_reserved() / 2 ** 30
            stop = stopper is not None and stopper.update(network, iteration, metrics)
            state['metrics'] = metrics
            state['rollout_steps'] += metrics.get('rollout/steps', 0.0)
            state['rollout_seconds'] += metrics.get('charts/rollout_seconds', 0.0)
            with interrupts_held():             # TensorBoard's queue: a SIGTERM waits for the scalars
                logger.log(iteration, result.total_steps, metrics)
            log.progress(iteration, args.iterations, result.total_steps, total, metrics, ppo.teacher_coeff)
            per_family = ' '.join(f"{k}={metrics.get(f'family/{k}/success_rate', float('nan')):.2f}" for k in keys)
            print(f"[lab] iter={iteration} steps={result.total_steps} {per_family} "
                  + format_summary(metrics), flush=True)
            ppo.teacher_coeff = teacher_coeff(args, iteration + 1)
            if iteration % args.save_every == 0:
                save()
            if args.checkpoint_every and iteration % args.checkpoint_every == 0:
                save()
                registry.snapshot_checkpoint(registry.checkpoint_path(args.name, run_id, root), iteration,
                                             args.checkpoint_keep)
            if stop:
                state['stopped_early'] = True
                return False
            return True

        began = time.time()
        status = 'completed'
        remaining = args.iterations - start_iteration
        train_seed = seed + start_iteration if args.seed is not None or resume is not None else None
        log.start(target_steps=total, target_iterations=args.iterations, steps_per_update=cfg.target_steps,
                  start_iteration=start_iteration, start_steps=start_steps, parallel=args.parallel,
                  device=args.device, run_dir=str(out_dir), stop_success=args.stop_success, families=keys,
                  mode=args.mode, arch=arch, parameters=network.parameter_count(), envs=collector.shares,
                  group=None if group is None else group.name)
        try:
            ppo.teacher_coeff = teacher_coeff(args, start_iteration + 1)
            ppo.teacher_value_coeff = args.teacher_value_coeff
            if args.mode == 'ppo':
                ppo.train(network, collector, remaining, callback=on_iteration, start_steps=start_steps,
                          seed=train_seed, collector=collector)
            else:
                optimizer = ppo.setup_optimizer(network)
                ppo.total_steps = start_steps
                for iteration in range(1, remaining + 1):
                    collector.start(network)
                    collector.take()
                    rollout = collector.ready(network, ppo.device)
                    stats = imitate(network, optimizer, collector.buffer.flatten(ppo.device), cfg,
                                    args.teacher_value_coeff, aux_losses)
                    ppo.total_steps += collector.buffer.rows
                    if on_iteration(IterationResult(iteration, ppo.total_steps, rollout, stats)) is False:
                        break
        except KeyboardInterrupt:
            status = 'interrupted'
        except BaseException as error:
            log.end('error', message=f"{type(error).__name__}: {error}", iterations=state['iteration'],
                    steps=state['steps'])
            print(f"[lab] {identity} failed at iteration {state['iteration']}; continue it with --resume {identity} "
                  f"(the same command line)", flush=True)
            raise
        finally:
            collector.close()
            logger.close()
            if state['iteration'] > start_iteration:
                save()
        status = 'early_stop' if state['stopped_early'] else status
        summary = {'status': status, 'iterations': state['iteration'], 'steps': state['steps'],
                   'start_iteration': start_iteration, 'wall_seconds': round(time.time() - began, 1)}
        # The run's steps a second (round40.md "Steps a second on record"): its mean over this process's wall time,
        # and the rollouts' alone (their steps over their seconds).
        summary['steps_per_second'] = round((state['steps'] - start_steps) / summary['wall_seconds'], 1) \
            if summary['wall_seconds'] > 0 else None
        summary['rollout_steps_per_second'] = round(state['rollout_steps'] / state['rollout_seconds'], 1) \
            if state['rollout_seconds'] > 0 else None
        log.end(status, iterations=state['iteration'], steps=state['steps'], wall_seconds=summary['wall_seconds'])
        write_json_atomic(out_dir / 'train_summary.json', summary)
        print(f"[lab] {status}: {summary}", flush=True)
    if status == 'interrupted' or not (out_dir / registry.CHECKPOINT_NAME).is_file():
        print(f"[lab] {identity} stays open: continue it with --resume {identity} (the same command line)",
              flush=True)
    elif stored is not None:
        sealed = store.seal_run(stored)
        store.record_summary('train', stored.identity, run=stored.identity, team=team['name'],
                             result={'status': status, 'iterations': summary['iterations']})
        print(f"[lab] sealed store run {stored.identity} (sha1 {sealed['sha1']})", flush=True)
    return out_dir


if __name__ == '__main__':
    sys.exit(cli())
