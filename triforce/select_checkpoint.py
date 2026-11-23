"""Choose a run's checkpoint by a chain segment (round 25).

    triforce select-checkpoint FAMILY RUN (--segment NAME | --from STATE [--stop-at ROOM|step:N] [--target L:0xRR])
        [--runs 1000] [--first-seed 0] [--clips on|off] [--act model|MODE]
        [--confirm 1000] [--confirm-top 1] [--confirm-first-seed S] [--only LABEL,...] [--no-promoted]
        [--apply] [--output PATH] [--chains-dir DIR] [--name BATCH] [--report-only] [--redo]
        [--parallel CORES] [--logs-dir DIR] [--models-dir DIR]

The candidates are the run's numbered checkpoints (``train --checkpoint-every N``: ``<run>/checkpoints/it<N>/``,
labelled ``it<N>``) and its ``ckpt.pt`` (``last``; a numbered checkpoint with the same weights is left out), and,
as the reference, the family's promoted run (``promoted``; left out with ``--no-promoted`` or when RUN is the
promoted run).  ``promoted`` is the run promoted when the selection starts, pinned for the whole batch: a promotion
while the batch waits or plays changes nothing in it.  ``--only`` keeps the listed labels.  Each candidate plays the
same chain seeds (``--runs`` from ``--first-seed``) of the segment in place of the family's promoted model, the rest
of the team as promoted (``eval-chain --checkpoint FAMILY=DIR``), all in one ``eval-batch``
(``triforce.eval_batch.run_batch``: one job of the machine's queue, ``triforce job run -- triforce select-checkpoint
...``, one group in the training GUI).

A segment (``--segment``, ``eval_batch.SEGMENTS``: ``d5`` = ``--from 5_76s_0``) is scored by its arrivals: the runs
that entered the next segment's start room (``eval_batch.TARGETS``), for the last segment (and a ``--stop-at``
without ``--target``) the runs that reached the end; with them Link's hearts on arrival (``eval_batch.arrival``,
``arrival_hearts``).  The screen ranks the run's candidates by arrivals, then hearts on arrival, each with its
standard error and the paired difference (same seeds) against ``last`` and against ``promoted``.

The winner of N noisy measurements looks better than it is, so the screen's winner is measured again on fresh
seeds (``--confirm`` runs from ``--confirm-first-seed``, default right after the screen's seeds), beside ``last``,
``promoted`` and the next best (``--confirm-top K``: the K best candidates other than ``last``; their entries pick
their checkpoint when they start, from the screen's results).  The confirmation's numbers are the ones to report.
The checkpoint chosen is the screen's winner when its confirmed paired difference in arrivals against ``last`` is
above zero, else ``last`` (``chosen``, ``reason``); the other confirmed candidates are information only (choosing
among them by the confirmation would bring the bias back).

Writes ``<run>/selection.json`` (``--output``): every candidate's numbers (training metrics of its iteration,
screen and confirmation scores, paired differences), the seeds, the winner, the confirmation and the choice.  The
chains go to ``<repo>/target/select/<family>-<run>-<segment>-clips<on|off>/`` (``--chains-dir``) as
``screen-<label>.jsonl`` and ``confirm-<label>.jsonl``; running the same selection again reuses every chain played
by the same weights, act mode and seeds (``--redo`` measures again), ``--report-only`` only reads them.

``--apply`` makes the chosen checkpoint the run's ``ckpt.pt`` (nothing happens when that is ``last``): the previous
``ckpt.pt`` is moved, with the evaluations of those weights (``eval*.json``), into its own numbered checkpoint dir
(``it<its iteration>``), the chosen weights are copied in (a new file: a hard-linked ``ckpt.pt`` is never written
through), the run's ``version`` follows them and its ``act`` keeps the run's mode, bound to the new sha1.  The run
then has no evaluation of its weights until it is evaluated again (``selection.json`` ``applied`` says what moved).
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .eval_batch import (BatchError, Entry, Options, arrival, arrival_hearts, bare_state, batch_work,
                         chain_entry, parse_target, read_runs, route_segments, run_batch, segment_starts)
from .jobs import announce
from .paths import scratch

LAST = 'last'
PROMOTED = 'promoted'
SELECTION_NAME = 'selection.json'
# Training metrics of a candidate's iteration, as its checkpoint stored them (``registry.save_checkpoint``).
TRAINING_KEYS = ('success_rate', 'balanced_success_rate', 'health_change', 'hearts_at_exit', 'losses/value_loss',
                 'losses/entropy', 'losses/explained_variance', 'losses/approx_kl')


@dataclass
class Candidate:
    """A checkpoint the selection plays: ``path`` is passed as ``--checkpoint FAMILY=PATH`` (None: no override, the
    family's promoted model as the chain finds it)."""
    label: str
    path: Optional[Path]
    checkpoint: str                     # provenance label (``provenance.checkpoint_label``)
    sha1: str                           # of its ``ckpt.pt`` (provenance ``models[].checkpoint_sha1``)
    act: str
    iteration: Optional[int] = None
    steps: Optional[int] = None
    training: dict = field(default_factory=dict)

    def record(self) -> dict:
        """The candidate's fields in ``selection.json``."""
        return {'label': self.label, 'checkpoint': self.checkpoint, 'sha1': self.sha1, 'act': self.act,
                'iteration': self.iteration, 'steps': self.steps, 'training': self.training}


def load_candidate(label: str, path: Path, override: bool) -> Candidate:
    """A candidate from a checkpoint file or dir (``override``: played through ``--checkpoint``)."""
    # pylint: disable=import-outside-toplevel
    from . import registry
    from .provenance import checkpoint_label
    checkpoint = registry.load_checkpoint(path)
    metrics = checkpoint.get('metrics') or {}
    return Candidate(label, Path(path) if override else None, checkpoint_label(path), registry.checkpoint_sha1(path),
                     registry.act_mode(path), checkpoint.get('iterations'), checkpoint.get('steps'),
                     {key: metrics[key] for key in TRAINING_KEYS if key in metrics})


def candidates(family: str, run_dir: Path, models_dir=None, only: Optional[list] = None,
               promoted: bool = True, route: Optional[str] = None, key: Optional[str] = None) -> list[Candidate]:
    """The run's numbered checkpoints (oldest first), ``last`` and ``promoted`` (module doc): the run that plays
    policy ``key`` (default the family's) on ``route`` (``registry.policy_run``; a route part without a run of its
    own plays the family's)."""
    # pylint: disable=import-outside-toplevel
    from . import registry
    if not (run_dir / registry.CHECKPOINT_NAME).is_file():
        raise BatchError(f"{run_dir}: no {registry.CHECKPOINT_NAME}")
    last_sha1 = registry.checkpoint_sha1(run_dir)
    found = [(f'it{iteration:06d}', path) for iteration, path in registry.list_snapshots(run_dir)
             if registry.checkpoint_sha1(path) != last_sha1]
    found.append((LAST, run_dir))
    if only is not None:
        unknown = sorted(set(only) - {label for label, _ in found} - {PROMOTED})
        if unknown:
            raise BatchError(f"--only {','.join(unknown)}: not a checkpoint of {run_dir} "
                             f"(it has {', '.join(label for label, _ in found)})")
        found = [(label, path) for label, path in found if label in only]
    result = [load_candidate(label, path, override=True) for label, path in found]
    if promoted and (only is None or PROMOTED in only):
        try:
            reference = registry.policy_checkpoint(key or family, models_dir, route)
        except FileNotFoundError:
            try:
                reference = registry.policy_checkpoint(family, models_dir, route)
            except FileNotFoundError:
                reference = None
        if reference is not None and reference.parent.resolve() != run_dir.resolve():
            result.append(load_candidate(PROMOTED, reference, override=True))
    return result


# ------------------------------------------------------------------------------------------------------- scoring

def score(runs: list[dict], target: Optional[str]) -> dict:
    """Arrivals and hearts on arrival of a chain's runs, with standard errors; ``by_seed``: seed -> (arrived, hearts
    on arrival or None)."""
    by_seed = {}
    for run in runs:
        record = arrival(run, target)
        by_seed[int(run['seed'])] = (record is not None, None if record is None else arrival_hearts(run, record))
    n = len(by_seed)
    arrived = sum(a for a, _ in by_seed.values())
    rate = arrived / n if n else 0.0
    hearts = [h for a, h in by_seed.values() if a]
    mean, se = mean_se(hearts)
    return {'runs': n, 'arrived': arrived, 'rate': rate, 'rate_se': math.sqrt(rate * (1 - rate) / n) if n else 0.0,
            'hearts': mean if hearts else None, 'hearts_se': se if hearts else None, 'by_seed': by_seed}


def mean_se(values: list[float]) -> tuple[float, float]:
    """Mean and its standard error (0 for fewer than two values)."""
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1) / len(values))


def paired(a: dict, b: dict) -> Optional[dict]:
    """``a`` minus ``b`` on their common seeds (``score`` results): arrivals (difference of the rates, standard error
    of the per-seed differences) and hearts on arrival over the seeds both arrived at."""
    seeds = sorted(set(a['by_seed']) & set(b['by_seed']))
    if not seeds:
        return None
    rate, rate_se = mean_se([float(a['by_seed'][s][0]) - float(b['by_seed'][s][0]) for s in seeds])
    both = [a['by_seed'][s][1] - b['by_seed'][s][1] for s in seeds if a['by_seed'][s][0] and b['by_seed'][s][0]]
    hearts, hearts_se = mean_se(both)
    return {'seeds': len(seeds), 'rate': rate, 'rate_se': rate_se, 'hearts': hearts if both else None,
            'hearts_se': hearts_se if both else None, 'hearts_seeds': len(both)}


def rank_key(result: dict) -> tuple:
    """Arrivals, then hearts on arrival (higher first: sort with reverse=True)."""
    return result['arrived'], result['hearts'] if result['hearts'] is not None else -math.inf


def public(result: Optional[dict]) -> Optional[dict]:
    """A score without its per-seed table (what ``selection.json`` keeps)."""
    return None if result is None else {k: v for k, v in result.items() if k != 'by_seed'}


# ------------------------------------------------------------------------------------------------------- the plan

@dataclass
class Plan:
    """What a selection measures (from the command line)."""
    family: str
    run_dir: Path
    start: str                  # --from
    clips: bool                 # --clips on
    route: Optional[str]        # --route (None: eval-chain's default)
    stop_at: Optional[str]      # --stop-at
    chain_args: list            # --from / --stop-at / --clips / --act / --route for every chain
    target: Optional[str]
    runs: int
    first_seed: int
    confirm: int
    confirm_first_seed: int
    confirm_top: int
    chains_dir: Path
    options: Options
    # The policy the candidates replace (round 32): ``family@route`` when the start is a step before the route's
    # ``own_models_until`` (a selection of the route's own run), else the family (``_native.step_policy``).
    key: Optional[str] = None

    @property
    def policy(self) -> str:
        """``key``, or the family."""
        return self.key or self.family

    def seeds(self, phase: str) -> tuple[int, int]:
        """(runs, first seed) of the screen or the confirmation."""
        return (self.runs, self.first_seed) if phase == 'screen' else (self.confirm, self.confirm_first_seed)


def chain_output(plan: Plan, phase: str, label: str) -> Path:
    """``<chains dir>/<phase>-<label>.jsonl``."""
    return plan.chains_dir / f'{phase}-{label}.jsonl'


def candidate_args(plan: Plan, candidate: Candidate, phase: str) -> list[str]:
    """``eval-chain`` arguments of one candidate's chain in ``phase``."""
    runs, first_seed = plan.seeds(phase)
    override = [] if candidate.path is None else ['--checkpoint', f'{plan.policy}={candidate.path}']
    return [*plan.chain_args, '--runs', str(runs), '--first-seed', str(first_seed), *override]


def played_by(plan: Plan, candidate: Candidate, phase: str) -> bool:
    """Whether the phase's chain output of ``candidate`` exists and holds the phase's seeds, finished, from the
    plan's start with its clips switch, played by ``candidate``'s weights (sha1) and act mode."""
    from .cli import meta_path  # pylint: disable=import-outside-toplevel
    output = chain_output(plan, phase, candidate.label)
    try:
        meta = json.loads(meta_path(output).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    if not output.is_file():
        return False
    runs, first_seed = plan.seeds(phase)
    provenance = meta.get('provenance') or {}
    batches = [(b.get('first_seed'), b.get('runs'), b.get('finished')) for b in meta.get('batches') or []]
    model = {m.get('name'): m for m in provenance.get('models') or []}.get(plan.policy) or {}
    return (batches == [(first_seed, runs, True)] and provenance.get('clips') == plan.clips
            and str(provenance.get('start_from') or 'start') == plan.start
            and model.get('checkpoint_sha1') == candidate.sha1
            and model.get('act') == candidate.act)


def measured_entry(plan: Plan, name: str, phase: str, pick: Callable[[], Candidate]) -> Entry:
    """A chain entry of the batch whose candidate ``pick()`` names when it starts (or is resumed): the screen's
    entries pick a fixed candidate, the confirmation's the screen's K-th best.  ``pick`` raises LookupError while it
    can't tell yet."""
    runs, first_seed = plan.seeds(phase)
    entry = Entry(name, 'chain', [], parallel=plan.options.cores, total=runs,
                  fields={'phase': phase, 'runs': runs, 'first_seed': first_seed, 'route': plan.route,
                          'start_from': plan.start, 'stop_at': plan.stop_at, 'target': plan.target,
                          'specialist': plan.family})

    def inner() -> tuple[Entry, Candidate]:
        candidate = pick()
        built = chain_entry(name, candidate_args(plan, candidate, phase), plan.options,
                            output=chain_output(plan, phase, candidate.label), target=plan.target)
        entry.output = built.output
        entry.fields['candidate'] = candidate.label
        return built, candidate

    def prepare():
        built, candidate = inner()
        entry.commands = built.commands
        entry.complete = lambda events: {**built.complete(events), 'candidate': candidate.label}
        built.prepare()

    def resume():
        if plan.options.redo:
            return None
        try:
            built, candidate = inner()
        except LookupError:
            return None
        done = built.resume() if played_by(plan, candidate, phase) else None
        return None if done is None else ({**done[0], 'candidate': candidate.label}, done[1])

    entry.prepare, entry.resume = prepare, resume
    return entry


class Selection:
    """The screen's and the confirmation's entries and their results, read back from the chain outputs."""

    def __init__(self, plan: Plan, pool: list[Candidate]):
        self.plan = plan
        self.pool = pool
        self.by_label = {candidate.label: candidate for candidate in pool}
        self._screen: Optional[dict] = None

    def entries(self) -> list[Entry]:
        """Screen (one per candidate), then confirmation (the K best but ``last``, ``last``, ``promoted``)."""
        plan = self.plan
        entries = [measured_entry(plan, f'screen-{c.label}', 'screen', lambda c=c: c) for c in self.pool]
        if plan.confirm <= 0:
            return entries
        others = [c for c in self.pool if c.label not in (LAST, PROMOTED)]
        for k in range(min(plan.confirm_top, len(others))):
            entries.append(measured_entry(plan, f'confirm-top{k + 1}', 'confirm',
                                          lambda k=k: self.ranked_others()[k]))
        for label in (LAST, PROMOTED):
            if label in self.by_label:
                entries.append(measured_entry(plan, f'confirm-{label}', 'confirm',
                                              lambda c=self.by_label[label]: c))
        return entries

    def results(self, phase: str) -> dict:
        """label -> score of the phase's chains that exist (played by the candidate, the phase's seeds)."""
        return {candidate.label: score(read_runs(chain_output(self.plan, phase, candidate.label)), self.plan.target)
                for candidate in self.pool if played_by(self.plan, candidate, phase)}

    def screen(self) -> dict:
        """The screen's scores; LookupError until the screen chain of every candidate of the run exists (the ranking
        needs them; ``promoted`` is a reference, scored when its chain exists)."""
        if self._screen is None:
            found = self.results('screen')
            missing = [c.label for c in self.pool if c.label != PROMOTED and c.label not in found]
            if missing:
                raise LookupError(f"the screen has no chain of {', '.join(missing)} yet")
            self._screen = found
        return self._screen

    def ranked(self) -> list[Candidate]:
        """The run's candidates (``promoted`` aside), best screen first."""
        screen = self.screen()
        run_candidates = [c for c in self.pool if c.label != PROMOTED]
        return sorted(run_candidates, key=lambda c: rank_key(screen[c.label]), reverse=True)

    def ranked_others(self) -> list[Candidate]:
        """``ranked`` without ``last`` (the confirmation measures ``last`` anyway)."""
        return [c for c in self.ranked() if c.label != LAST]

    def report(self) -> dict:
        """``selection.json``'s content (module doc)."""
        plan = self.plan
        screen = self.screen()
        confirm = self.results('confirm') if plan.confirm > 0 else {}
        ranked = self.ranked()
        winner = ranked[0]
        rows = []
        for candidate in self.pool:
            label = candidate.label
            row = {**candidate.record(), 'rank': next((i + 1 for i, c in enumerate(ranked) if c.label == label), None),
                   'screen': public(screen.get(label)), 'confirm': public(confirm.get(label))}
            for phase, results in (('screen', screen), ('confirm', confirm)):
                for reference in (LAST, PROMOTED):
                    if label != reference and label in results and reference in results:
                        row[f'{phase}_vs_{reference}'] = paired(results[label], results[reference])
            rows.append(row)
        chosen, reason = LAST, "the screen's winner is the last checkpoint"
        if winner.label != LAST:
            versus = (paired(confirm[winner.label], confirm[LAST])
                      if winner.label in confirm and LAST in confirm else None)
            if versus is None:
                reason = "not confirmed: the winner and last were not both measured on fresh seeds"
            elif versus['rate'] > 0:
                chosen, reason = winner.label, (f"confirmed: {versus['rate']:+.1%} ± {versus['rate_se']:.1%} arrivals "
                                                f"against last on {versus['seeds']} fresh seeds")
            else:
                reason = (f"not confirmed: {versus['rate']:+.1%} ± {versus['rate_se']:.1%} arrivals against last on "
                          f"{versus['seeds']} fresh seeds")
        return {'family': plan.family, 'policy': plan.policy, 'run': plan.run_dir.name, 'chain_args': plan.chain_args,
                'target': plan.target,
                'screen': {'first_seed': plan.first_seed, 'runs': plan.runs},
                'confirm': {'first_seed': plan.confirm_first_seed, 'runs': plan.confirm, 'top': plan.confirm_top},
                'chains_dir': str(plan.chains_dir), 'candidates': rows, 'ranking': [c.label for c in ranked],
                'winner': winner.label, 'chosen': chosen, 'reason': reason, 'applied': None,
                'time': time.strftime('%Y-%m-%dT%H:%M:%S%z')}


# ------------------------------------------------------------------------------------------------------- applying

def apply_choice(run_dir: Path, chosen: Path) -> dict:
    """Makes the checkpoint in ``chosen`` (a numbered checkpoint dir of ``run_dir``) the run's ``ckpt.pt`` (module
    doc); returns what moved."""
    # pylint: disable=import-outside-toplevel
    from . import registry
    ckpt = run_dir / registry.CHECKPOINT_NAME
    mode = registry.act_mode(ckpt)
    version = registry.model_version(chosen) or registry.model_version(ckpt)
    old_sha1 = registry.checkpoint_sha1(ckpt)
    iteration = registry.load_checkpoint(ckpt, check_schema=False).get('iterations') or 0
    kept = registry.snapshot_dir(run_dir, iteration)
    if kept.exists() and (not (kept / registry.CHECKPOINT_NAME).is_file()
                          or registry.checkpoint_sha1(kept) != old_sha1):
        kept = kept.with_name(f'{kept.name}-replaced-{old_sha1[:8]}')
    kept.mkdir(parents=True, exist_ok=True)
    os.replace(ckpt, kept / registry.CHECKPOINT_NAME)
    registry.write_model_version(kept, registry.model_version(run_dir) or version)
    registry.set_act_mode(kept, mode)
    moved = []
    for evaluation in sorted(run_dir.glob('eval*.json')):
        os.replace(evaluation, kept / evaluation.name)
        moved.append(evaluation.name)
    tmp = run_dir / f'.{registry.CHECKPOINT_NAME}.{os.getpid()}.tmp'
    shutil.copyfile(chosen / registry.CHECKPOINT_NAME, tmp)
    os.replace(tmp, ckpt)
    registry.write_model_version(run_dir, version)
    registry.set_act_mode(run_dir, mode)
    return {'from_sha1': old_sha1, 'to_sha1': registry.checkpoint_sha1(ckpt), 'source': chosen.name,
            'previous_kept': kept.name, 'stale_evaluations': moved, 'act': mode, 'version': version}


# ------------------------------------------------------------------------------------------------------- printing

def _pct(value, se=None) -> str:
    if value is None:
        return '-'
    return f"{value * 100:.1f}" + (f" ± {se * 100:.1f}" if se is not None else '')


def _hearts(value, se=None) -> str:
    if value is None:
        return '-'
    return f"{value:.2f}" + (f" ± {se:.2f}" if se is not None else '')


def _diff(row: dict, key: str) -> str:
    value = row.get(key)
    return '-' if value is None else f"{value['rate'] * 100:+.1f} ± {value['rate_se'] * 100:.1f}"


def table(report: dict) -> list[str]:
    """The selection as markdown: per candidate its iteration, screen, confirmation and paired differences."""
    lines = ['| candidate | iteration | screen arrivals % | hearts | vs last (pts) | vs promoted (pts) | confirm % '
             '| confirm hearts | confirm vs last | confirm vs promoted |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in sorted(report['candidates'], key=lambda r: (r['rank'] is None, r['rank'] or 0)):
        screen, confirm = row['screen'] or {}, row['confirm'] or {}
        mark = ' (winner)' if row['label'] == report['winner'] else ''
        lines.append(f"| {row['label']}{mark} | {row['iteration']} | {_pct(screen.get('rate'), screen.get('rate_se'))} "
                     f"| {_hearts(screen.get('hearts'), screen.get('hearts_se'))} | {_diff(row, 'screen_vs_last')} "
                     f"| {_diff(row, 'screen_vs_promoted')} | {_pct(confirm.get('rate'), confirm.get('rate_se'))} "
                     f"| {_hearts(confirm.get('hearts'), confirm.get('hearts_se'))} | {_diff(row, 'confirm_vs_last')} "
                     f"| {_diff(row, 'confirm_vs_promoted')} |")
    lines.append(f"winner {report['winner']}; chosen {report['chosen']}: {report['reason']}")
    return lines


# ------------------------------------------------------------------------------------------------------------ CLI

def build_parser() -> argparse.ArgumentParser:
    """The command line."""
    from . import registry  # pylint: disable=import-outside-toplevel
    parser = argparse.ArgumentParser(prog='triforce select-checkpoint', description=__doc__.split('\n\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split('\n\n', 1)[1])
    parser.add_argument('family', help="the specialist whose model is replaced in the chain")
    parser.add_argument('run', help="the run id (or run dir) whose checkpoints are measured")
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument('--segment', help="a chain segment of --route's table (eval_batch.ROUTE_SEGMENTS; the main "
                                         "route's: " + ', '.join(s.split(':')[0]
                                                                 for s in route_segments().segments.split()) + ")")
    where.add_argument('--from', dest='start_from', help="a start state (as eval-chain --from)")
    parser.add_argument('--stop-at', default=None, help="as eval-chain --stop-at")
    parser.add_argument('--target', default=None, metavar='L:0xRR[,...]',
                        help="arrival room (default: the segment's, else the route's end or --stop-at)")
    parser.add_argument('--route', default=None, help="eval-chain --route (default its own)")
    parser.add_argument('--runs', type=int, default=1000, help="screen runs per candidate (default 1000)")
    parser.add_argument('--first-seed', type=int, default=0)
    parser.add_argument('--clips', choices=('on', 'off'), default='on')
    parser.add_argument('--act', type=registry.act_argument(), default=registry.MODEL_ACT,
                        help="model (each model's act file, the default) or an act mode for every model")
    parser.add_argument('--confirm', type=int, default=1000, help="confirmation runs on fresh seeds (0: none)")
    parser.add_argument('--confirm-top', type=int, default=1,
                        help="confirm the K best candidates other than last (beside last and promoted)")
    parser.add_argument('--confirm-first-seed', type=int, default=None,
                        help="default --first-seed + --runs (the seeds right after the screen's)")
    parser.add_argument('--only', default=None, help="comma-separated candidate labels (it<N>, last, promoted)")
    parser.add_argument('--no-promoted', action='store_true', help="leave the promoted run out")
    parser.add_argument('--apply', action='store_true', help="make the chosen checkpoint the run's ckpt.pt")
    parser.add_argument('--output', type=Path, default=None, help=f"default <run>/{SELECTION_NAME}")
    parser.add_argument('--chains-dir', type=Path, default=None,
                        help="default <repo>/target/select/<family>-<run>-<segment>-clips<on|off>")
    parser.add_argument('--name', default=None, help="the batch's name (default select-<family>-<run>-<segment>)")
    parser.add_argument('--report-only', action='store_true', help="only read the chains that exist")
    parser.add_argument('--redo', action='store_true', help="measure again what was measured")
    parser.add_argument('--parallel', type=int, default=os.cpu_count() or 1, help="envs of each chain")
    parser.add_argument('--logs-dir', default=None)
    parser.add_argument('--models-dir', default=None)
    return parser


def build_plan(args) -> tuple[Plan, list[Candidate]]:
    """The plan and the candidates; raises BatchError when the command line can't run."""
    # pylint: disable=import-outside-toplevel
    from . import registry
    from ._native import route_state, route_steps, step_policy
    from .runner import _start_step
    run_dir = Path(args.run) if os.sep in args.run else registry.run_dir(args.family, args.run, args.models_dir)
    table = route_segments(args.route)
    if args.segment is not None:
        segments = dict(item.split(':', 1) for item in table.segments.split())
        if args.segment not in segments or segments[args.segment] == 'start':
            raise BatchError(f"--segment {args.segment}: one of "
                             + ', '.join(n for n, s in segments.items() if s != 'start') + " (or --from start)")
        start, where = segments[args.segment], args.segment
    else:
        start, where = args.start_from, args.start_from
    target = args.target
    if target is None and args.stop_at is None:
        target = table.targets.get(args.segment or segment_starts(args.route).get(bare_state(start or ''), ''))
    if target is not None:
        parse_target(target)
    if start != 'start':        # the name eval-chain plays and records (--route d3first --from 3_7cs_0: d3first_3_7cs_0)
        try:
            start = route_state(args.route or registry.MAIN_ROUTE, start)
        except ValueError as error:
            raise BatchError(f"--from {start}: {error}") from error
    if args.runs < 1 or args.confirm < 0 or args.confirm_top < 0 or args.parallel < 1:
        raise BatchError("--runs and --parallel must be >= 1, --confirm and --confirm-top >= 0")
    chain_args = ['--from', start, '--clips', args.clips, '--act', args.act]
    chain_args += ['--stop-at', args.stop_at] if args.stop_at else []
    chain_args += ['--route', args.route] if args.route else []
    confirm_first = args.confirm_first_seed if args.confirm_first_seed is not None else args.first_seed + args.runs
    fresh = confirm_first >= args.first_seed + args.runs or confirm_first + args.confirm <= args.first_seed
    if args.confirm > 0 and not fresh:
        raise BatchError(f"--confirm-first-seed {confirm_first}: the confirmation's seeds must be fresh "
                         f"(the screen plays {args.first_seed}..{args.first_seed + args.runs - 1})")
    where = Path(str(where)).name.removesuffix('.state')
    tag = '-'.join([f"{args.family}-{run_dir.name}-{where}", *([f"to-{args.stop_at}"] if args.stop_at else []),
                    *([args.route] if args.route else []),
                    *([f"act-{args.act}"] if args.act != registry.MODEL_ACT else []), f"clips{args.clips}"])
    chains_dir = args.chains_dir or scratch('select', ''.join(
        c if c.isalnum() or c in '-_.' else '-' for c in tag))
    options = Options(args.name or f'select-{args.family}-{run_dir.name}-{where}', args.parallel, 1, args.models_dir,
                      args.redo)
    # The part the candidates play (round 32): a start before the route's own_models_until selects its own run.
    route = args.route or registry.MAIN_ROUTE
    key = step_policy(route, _start_step(route_steps(route), start), args.family)
    plan = Plan(args.family, run_dir, start, args.clips == 'on', args.route, args.stop_at, chain_args, target,
                args.runs, args.first_seed, args.confirm, confirm_first, args.confirm_top, Path(chains_dir).resolve(),
                options, key=key)
    only = None if args.only is None else [label for label in args.only.split(',') if label]
    pool = candidates(args.family, run_dir, args.models_dir, only, promoted=not args.no_promoted, route=args.route,
                      key=key)
    if not [c for c in pool if c.label != PROMOTED]:
        raise BatchError(f"{run_dir}: no candidate of the run left")
    return plan, pool


def main(argv=None) -> int:
    """``triforce select-checkpoint``: 0 done, 1 a chain failed, 2 a selection that can't run, 130 stopped."""
    # pylint: disable=import-outside-toplevel
    from .events import write_json_atomic
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    try:
        plan, pool = build_plan(args)
    except BatchError as error:
        print(f"triforce select-checkpoint: error: {error}", file=sys.stderr)
        return 2
    selection = Selection(plan, pool)
    code = 0
    if not args.report_only:
        entries = selection.entries()
        announce('batch', batch_work(entries), plan.options.batch)
        plan.chains_dir.mkdir(parents=True, exist_ok=True)
        code = run_batch(plan.options.batch, entries, logs_dir=args.logs_dir, cores=args.parallel,
                         nice=os.nice(0), command=[sys.executable, '-m', 'triforce.select_checkpoint', *argv],
                         extra={'models_dir': args.models_dir, 'selection': str(plan.run_dir)})
        if code == 130:
            return code
    try:
        report = selection.report()
    except LookupError as error:
        print(f"triforce select-checkpoint: error: {error}", file=sys.stderr)
        return 1
    output = args.output or plan.run_dir / SELECTION_NAME
    if args.apply and report['chosen'] != LAST:
        chosen = next(c for c in pool if c.label == report['chosen'])
        report['applied'] = apply_choice(plan.run_dir, chosen.path)
    write_json_atomic(output, report)
    for line in table(report):
        print(line)
    if report['applied']:
        print(f"applied: {plan.run_dir.name}/ckpt.pt is now {report['chosen']} (the previous one is "
              f"checkpoints/{report['applied']['previous_kept']}); evaluate the run again")
    print(f"selection in {output}")
    return code


if __name__ == '__main__':
    sys.exit(main())
