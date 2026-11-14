"""Evaluations in one batch: one log folder, one plan, one progress bar per evaluation in the training GUI.

    triforce eval-batch NAME [--chains TAG [--act sample] [--runs 120] [--segments SPEC] [--chains-dir DIR]
                                           [--chain-args ARGS] [--no-report | --report-only]]
                             [--families all|a,b,c [--acts model[,sample,argmax]] [--run-id ID]
                                           [--episodes-per-state N] [--seed S] [--clips on|off|both]]
                             [--chain '[ENTRY:] EVAL-CHAIN ARGS' ...] [--evaluate '[ENTRY:] EVALUATE ARGS' ...]
                             [--parallel CORES] [--eval-parallel 4] [--jobs 1] [--nice N] [--redo]
                             [--logs-dir DIR] [--models-dir DIR] [--allow-incompatible-models]

A batch is a named list of evaluations, in command-line order:

- ``--chain '[ENTRY:] ARGS'``: one ``triforce eval-chain`` with ARGS (everything it takes but ``--append``,
  ``--events``, ``--no-log``, ``--logs-dir``).  Its output is ARGS' ``--output``, else
  ``runs/<batch>/<entry>.jsonl``; the chain writes ``<output>.tmp`` and the batch moves it (and its ``.meta.json``)
  into place only when the chain succeeded.  A chain from a segment start (``SEGMENTS``) without ``--stop-at`` is
  also scored by its arrivals: the runs that entered the next segment's start room.
- ``--evaluate '[ENTRY:] ARGS'``: one ``triforce evaluate`` with ARGS (specialist name or YAML, run id or checkpoint,
  act mode, episodes...; not ``--events``, ``--no-log``, ``--logs-dir``); its result file is ``evaluate``'s own.
- ``--chains TAG``: the chain segments of ``scripts/experiments/r56_chains.sh`` (``--segments``, default
  ``SEGMENTS``), ``--runs`` runs each in ``--act`` mode, written as ``<chains-dir>/<TAG>-<segment>.jsonl`` with the
  console output in ``<TAG>-<segment>.log``, then the report (``scripts/yardstick.py report`` into
  ``<TAG>-yardstick.json`` and ``triforce room-stats TAG``) as the batch's last entry.
- ``--families all|a,b,c``: the evaluation of each family's promoted run (``--run-id``: that run) in each of
  ``--acts`` (default ``model``: the run's own act mode, the canonical ``eval.json``); ``all`` = every specialist
  with a promoted run (or that run), in route order.  ``--clips`` goes to every evaluation (default: each family's
  own, ``clips.default_mode``: both switch states for a clip family; the headline then adds
  ``eval_clips_on_success_rate`` / ``eval_clips_off_success_rate``).

Each evaluation runs as its own process (``python -m triforce.cli eval-chain`` / ``python -m triforce.evaluate``
with ``--events <batch folder>/<entry>.jsonl``), its console output in ``<entry>.log`` (or the chain segment's own
log).  One at a time by default: a chain takes ``--parallel`` envs (default: every core) unless its ARGS set
``--parallel``, an evaluation ``--eval-parallel`` (default 4).  ``--jobs N`` runs up to N family evaluations side by
side as long as their envs fit in ``--parallel``; chains and the report always run alone.  The batch lowers its
priority first (``--nice``, default ``$TRIFORCE_NICE`` else 15, ``triforce.jobs``) and its evaluations inherit it.

Log folder ``logs/<yymmdd>_<NNN>_<NAME>/`` (``triforce/events.py``): ``training.json`` (kind ``eval-batch``, every
entry listed up front as pending, then ``evaluating`` and its final status ``ok`` / ``failed`` / ``interrupted``
with its headline: chains ``chain_runs``, ``chain_successes``, ``chain_success_rate``, ``chain_target``,
``chain_arrived``, ``chain_arrival_rate``; evaluations ``eval_success_rate``, ``eval_episodes``,
``eval_health_change``...; both ``wall_seconds`` and ``output``), ``round.jsonl`` / ``round.log`` (the batch's own
events: ``start``, ``entry_start``, ``entry_end``, ``end``) and one ``<entry>.jsonl`` stream each (the command's
``chain_start`` / ``chain_progress`` / ``chain`` or ``eval_start`` / ``eval_progress`` / ``eval`` events, then the
batch's ``error`` and final ``end``).

A failed evaluation is marked and the batch goes on (exit 1 at the end).  One SIGTERM (or Ctrl-C) stops the running
evaluations and marks them and everything still pending ``interrupted`` (exit 130).  Running the same batch again
resumes it: a chain whose output exists is done (as ``r56_chains.sh`` skipped finished segment files); an evaluation
whose result file holds the same measurement (same weights by sha1, act mode, episodes per state, seed, code and
data versions) is done.  Done entries show as finished with their headline (``resumed``); ``--redo`` measures them
again (a chain's existing output must then be moved away first: eval-chain never overwrites one).
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .events import LOGS_DIR_ENV, EventLog, TrainingPlan, new_log_dir
from .jobs import add_nice_argument, child_env, lower_priority

ROOT = Path(__file__).resolve().parents[1]
KIND = 'eval-batch'
EXIT_INTERRUPTED = 130
POLL_SECONDS = 0.2
TERMINATE_SECONDS = 60

# The chain segments (r56_chains.sh): name:start state ('start' = the game start).  Every segment plays on to the
# route's end; its arrival is the first entry into the next segment's start room (TARGETS).
SEGMENTS = ('full:start d2:2_7ds_0 post-d2:0_3cc_0 d1:1_73s_0 post-d1:0_37c_0 d3:3_7cs_0 post-d3:0_74c_0 '
            'd4:4_71s_0')
TARGETS = {'d2': '0:0x3c', 'post-d2': '1:0x73', 'd1': '0:0x37', 'post-d1': '3:0x7c', 'd3': '0:0x74',
           'post-d3': '4:0x71'}
# The full run is also reported at each milestone room, in order (yardstick ``@L:0xRR,...``).
MILESTONES = (('d2', '2:0x7d'), ('post-d2', '0:0x3c'), ('d1', '1:0x73'), ('post-d1', '0:0x37'), ('d3', '3:0x7c'),
              ('post-d3', '0:0x74'), ('d4', '4:0x71'))
CHAIN_RESERVED = ('--append', '--events', '--no-log', '--logs-dir')
EVAL_RESERVED = ('--events', '--no-log', '--logs-dir')


def segment_starts() -> dict:
    """Segment start state -> segment name (``SEGMENTS``, the game start left out)."""
    return {start: name for name, start in (item.split(':', 1) for item in SEGMENTS.split()) if start != 'start'}


# ------------------------------------------------------------------------------------------------------ arrivals

def parse_target(target: str) -> list[tuple[int, int]]:
    """'2:0x7d,0:0x3C' → [(2, 0x7d), (0, 0x3c)]; anything else raises ValueError."""
    rooms = []
    for room in target.split(','):
        level, sep, number = room.strip().partition(':')
        try:
            if not sep or not number.lower().startswith('0x'):
                raise ValueError(room)
            rooms.append((int(level), int(number, 16)))
        except ValueError:
            raise ValueError(f"arrival target {room!r} is not L:0xRR (in {target!r})") from None
    return rooms


def arrival(run: dict, target: Optional[str]):
    """The room record where ``run`` (an eval-chain output line) arrived: the first entry into ``target``
    ('L:0xRR', outside caves) after its start room, or with no target the last room of a successful run; None if it
    never got there.  ``target`` may list rooms in order ('2:0x7d,0:0x3c'): the arrival is the entry into the last
    one after entering the others in turn."""
    if target is None:
        return run['rooms'][-1] if run['result'] == 'success' and run['rooms'] else None
    rooms = parse_target(target)
    at = 0
    for record in run['rooms'][1:]:
        if (int(record['level']), int(str(record['room']), 16)) == rooms[at] and not record['in_cave']:
            at += 1
            if at == len(rooms):
                return record
    return None


def read_runs(path: Path) -> list[dict]:
    """The runs of an eval-chain output (JSON lines)."""
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


# ------------------------------------------------------------------------------------------------------ headlines

def chain_headline(summary: dict, runs: Optional[list[dict]] = None, target: Optional[str] = None) -> dict:
    """A chain entry's plan/end fields from its ``chain`` summary, plus its arrivals at ``target`` (from the output's
    ``runs``) when it has one."""
    fields = {'chain_runs': summary.get('runs'), 'chain_successes': summary.get('successes'),
              'chain_success_rate': summary.get('success_rate'), 'chain_target': target}
    if target is not None and runs is not None:
        arrived = sum(arrival(run, target) is not None for run in runs)
        fields.update(chain_arrived=arrived, chain_arrival_rate=arrived / len(runs) if runs else None)
    return fields


def eval_headline(event: dict) -> dict:
    """An evaluation entry's plan/end fields from its ``eval`` event (``evaluate.eval_event`` fields)."""
    return {'eval_success_rate': event.get('success_rate'), 'eval_episodes': event.get('episodes'),
            'eval_act': ('argmax' if event['deterministic'] else 'sample') if 'deterministic' in event else None,
            'eval_health': event.get('health'), 'eval_health_change': event.get('health_change'),
            'eval_health_change_total': event.get('health_change_total'),
            'eval_hearts_dropped': event.get('hearts_dropped'), 'eval_hearts_taken': event.get('hearts_taken'),
            'eval_hearts_left': event.get('hearts_left'),
            **{f'eval_clips_{switch}_success_rate': stats.get('success_rate')
               for switch, stats in (event.get('by_clips') or {}).items()}}


def _pct(value) -> str:
    return '-' if value is None else f"{value:.1%}"


def fmt_duration(seconds) -> str:
    """Seconds -> 1h 02m / 3m 05s / 42s."""
    if seconds is None:
        return '-'
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"


def headline_text(fields: dict) -> str:
    """One line of an entry's result: 'arrivals 412/1000 = 41.2% at 0:0x3c · success 22.0% · 4m 19s'."""
    parts = []
    if fields.get('chain_runs') is not None:
        runs = fields['chain_runs']
        if fields.get('chain_arrived') is not None:
            parts.append(f"arrivals {fields['chain_arrived']}/{runs} = {_pct(fields.get('chain_arrival_rate'))} "
                         f"at {fields.get('chain_target')}")
        parts.append(f"success {fields.get('chain_successes')}/{runs} = {_pct(fields.get('chain_success_rate'))}")
    if fields.get('eval_success_rate') is not None:
        parts.append(f"success {_pct(fields['eval_success_rate'])} of {fields.get('eval_episodes')} episodes")
        if fields.get('eval_health_change') is not None:
            parts.append(f"health {fields['eval_health_change']:+.2f} hearts")
    if fields.get('wall_seconds') is not None:
        parts.append(fmt_duration(fields['wall_seconds']))
    return ' · '.join(parts)


# -------------------------------------------------------------------------------------------------------- entries

@dataclass
class Entry:
    """One evaluation of a batch.  ``commands`` run in turn (the first gets ``--events <stream>`` when ``events``);
    ``complete(stream events)`` turns a successful run into its headline fields and ``resume()`` returns (headline,
    events to write into its stream) when an earlier run already measured it, else None."""
    name: str
    kind: str                                   # chain | eval | report
    commands: list
    parallel: int = 1
    total: Optional[int] = None                 # runs or episodes
    output: Optional[Path] = None
    log: Optional[Path] = None                  # console output (default <batch>/<entry>.log)
    cwd: Optional[Path] = None
    events: bool = True                         # pass --events <stream> to the first command
    fields: dict = field(default_factory=dict)  # more plan fields (specialist, act, start_from, ...)
    prepare: Optional[Callable[[], None]] = None
    complete: Optional[Callable[[list], dict]] = None
    resume: Optional[Callable[[], Optional[tuple]]] = None
    builder: Optional[Callable[[], list]] = None    # computes ``commands`` at launch (the report)

    def plan_entry(self, stream: str, log: Path, log_dir: Path) -> dict:
        """The entry's ``training.json`` run (status pending)."""
        log_text = os.path.relpath(log, log_dir) if log.parent == log_dir else str(log)
        return {'name': self.name, 'kind': self.kind, 'events': stream, 'log': log_text,
                'output': None if self.output is None else str(self.output), 'total': self.total,
                'parallel': self.parallel, 'command': self.commands, **self.fields}


def _safe(name: str) -> str:
    return ''.join(c if c.isalnum() or c in '-_.' else '-' for c in name)


def _has_option(args: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(option + '=') for arg in args)


def _drop_option(args: list[str], option: str) -> list[str]:
    """``args`` without ``option VALUE`` / ``option=VALUE``."""
    kept, skip = [], False
    for arg in args:
        if skip:
            skip = False
        elif arg == option:
            skip = True
        elif not arg.startswith(option + '='):
            kept.append(arg)
    return kept


def _stream_events(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return []
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


def _last(events: list[dict], kind: str) -> Optional[dict]:
    return next((event for event in reversed(events) if event.get('event') == kind), None)


class BatchError(ValueError):
    """A batch that can't run as given (nothing runs; exit 2)."""


@dataclass
class Options:
    """What every entry shares."""
    batch: str
    cores: int
    eval_parallel: int
    models_dir: Optional[str] = None
    redo: bool = False


def chain_entry(name: Optional[str], args: list[str], options: Options, *, output: Optional[Path] = None,
                log: Optional[Path] = None, target: Optional[str] = None) -> Entry:
    """A ``triforce eval-chain ARGS`` entry (``--output``/``output`` the final output, written through
    ``<output>.tmp``); ``target``: its arrival room (default: the next segment's start room for a segment start
    without ``--stop-at``)."""
    # pylint: disable=import-outside-toplevel
    from .cli import add_eval_chain_arguments, meta_path
    from .runner import RunResult, chain_summary
    parser = argparse.ArgumentParser(prog=f'eval-batch --chain {name or ""}'.strip())
    add_eval_chain_arguments(parser)
    parsed = parser.parse_args(args)
    reserved = [option for option in CHAIN_RESERVED if _has_option(args, option)]
    if reserved:
        raise BatchError(f"chain {name or ' '.join(args)}: {', '.join(reserved)} can't be used in a batch")
    if name is None:
        name = 'chain' + (f"-from-{parsed.start_from}" if parsed.start_from else '') + (
            f"-to-{parsed.stop_at}" if parsed.stop_at else '') + ('-power-on' if parsed.power_on else '')
    if target is None and parsed.stop_at is None and parsed.start_from in segment_starts():
        target = TARGETS.get(segment_starts()[parsed.start_from])
    if target is not None:
        parse_target(target)
    output = Path(parsed.output) if parsed.output else (output or Path('runs') / _safe(options.batch)
                                                        / f"{_safe(name)}.jsonl")
    tmp = Path(f"{output}.tmp")
    args = _drop_option(args, '--output')
    parallel = parsed.parallel if _has_option(args, '--parallel') else options.cores
    if not _has_option(args, '--parallel'):
        args += ['--parallel', str(parallel)]
    prefix = [sys.executable, '-m', 'triforce.cli'] + (
        ['--models-dir', options.models_dir] if options.models_dir else [])
    command = [*prefix, 'eval-chain', *args, '--output', str(tmp)]

    def prepare():
        for stale in (tmp, meta_path(tmp)):         # eval-chain refuses an existing --output
            stale.unlink(missing_ok=True)

    def complete(events: list[dict]) -> dict:
        summary = _last(events, 'chain')
        if summary is None:
            raise RuntimeError("the chain ended without its summary ('chain' event)")
        if tmp.exists():
            os.replace(meta_path(tmp), meta_path(output))
            os.replace(tmp, output)
        return chain_headline(summary, read_runs(output) if target is not None else None, target)

    def resume():
        if options.redo or not output.is_file() or output.stat().st_size == 0:
            return None
        runs = read_runs(output)
        summary = chain_summary([RunResult.from_dict(run) for run in runs])
        return chain_headline(summary, runs, target), [('chain', {**summary, 'resumed': True})]

    fields = {'route': parsed.route, 'runs': parsed.runs, 'act': parsed.act, 'start_from': parsed.start_from,
              'stop_at': parsed.stop_at, 'target': target}
    return Entry(name, 'chain', [command], parallel=parallel, total=parsed.runs, output=output, log=log,
                 fields=fields, prepare=prepare, complete=complete, resume=resume)


# What makes an evaluation result the same measurement as the one an entry would make (seeded episodes play the
# same, so measuring it again would give the same numbers).
MEASUREMENT_FIELDS = ('specialist', 'act', 'episodes_per_state', 'seed', 'clips', 'checkpoint_sha1', 'code', 'data')


def same_measurement(result: dict, expected: dict) -> bool:
    """Whether an evaluation result (``evaluate_specialist``'s) holds ``expected``'s measurement: every
    ``MEASUREMENT_FIELDS`` equal (``act`` from ``registry.eval_mode``)."""
    from .registry import eval_mode  # pylint: disable=import-outside-toplevel
    found = {**{key: result.get(key) for key in MEASUREMENT_FIELDS}, 'act': eval_mode(result)}
    return all(found[key] == expected.get(key) for key in MEASUREMENT_FIELDS)


def eval_entry(name: Optional[str], args: list[str], options: Options) -> Entry:
    """A ``triforce evaluate ARGS`` entry; its result file is evaluate's (``--output`` or ``default_output``)."""
    # pylint: disable=import-outside-toplevel
    from . import registry
    from .clips import default_mode, episode_jobs
    from .evaluate import build_parser, default_output, eval_event, resolve_checkpoint
    from .provenance import code_version, data_hash
    from .specialist import load_specialist
    parser = build_parser()
    parser.prog = f'eval-batch --evaluate {name or ""}'.strip()
    parsed = parser.parse_args(args)
    reserved = [option for option in EVAL_RESERVED if _has_option(args, option)]
    if reserved:
        raise BatchError(f"evaluation {name or ' '.join(args)}: {', '.join(reserved)} can't be used in a batch")
    config = load_specialist(parsed.specialist)
    models_dir = parsed.models_dir or options.models_dir
    try:
        checkpoint = resolve_checkpoint(config.name, parsed.checkpoint, parsed.run_id, models_dir)
    except FileNotFoundError as error:
        raise BatchError(f"evaluation of {config.name}: no checkpoint ({error})") from error
    if name is None:
        name = config.name + (f"-{parsed.run_id}" if parsed.run_id else '') + (
            f"-{parsed.act}" if parsed.act != 'model' else '')
    clips = parsed.clips or default_mode(config)
    output = Path(parsed.output) if parsed.output else default_output(checkpoint, config.name, parsed.act,
                                                                      parsed.episodes_per_state, parsed.seed,
                                                                      parsed.clips)
    parallel = parsed.parallel if _has_option(args, '--parallel') else options.eval_parallel
    if not _has_option(args, '--parallel'):
        args = [*args, '--parallel', str(parallel)]
    if options.models_dir and not parsed.models_dir:
        args = [*args, '--models-dir', options.models_dir]
    command = [sys.executable, '-m', 'triforce.evaluate', *args]
    act = registry.act_mode(checkpoint) if parsed.act == 'model' else parsed.act

    def complete(events: list[dict]) -> dict:
        event = _last(events, 'eval')
        if event is None:
            raise RuntimeError("the evaluation ended without its result ('eval' event)")
        return eval_headline(event)

    def resume():
        if options.redo or not output.is_file():
            return None
        try:
            result = json.loads(output.read_text(encoding='utf-8'))
        except ValueError:
            return None
        expected = {'specialist': config.name, 'act': act, 'episodes_per_state': parsed.episodes_per_state,
                    'seed': parsed.seed, 'clips': clips, 'checkpoint_sha1': registry.checkpoint_sha1(checkpoint),
                    'code': code_version(), 'data': data_hash()}
        if not same_measurement(result, expected):
            return None
        event = eval_event(result)
        return eval_headline(event), [('eval', {**event, 'path': str(output), 'resumed': True})]

    episodes = len(episode_jobs(config, parsed.episodes_per_state, parsed.seed, clips))
    fields = {'specialist': config.name, 'run_id': checkpoint.parent.name, 'run_dir': str(checkpoint.parent),
              'act': act, 'episodes_per_state': parsed.episodes_per_state, 'seed': parsed.seed, 'clips': clips}
    return Entry(name, 'eval', [command], parallel=parallel, total=episodes,
                 output=output, fields=fields, complete=complete, resume=resume)


def chains_entries(tag: str, options: Options, *, act: str, runs: int, segments: str, chains_dir: Path,
                   chain_args: list[str], report: bool, only_report: bool) -> list[Entry]:
    """The chain segments of a tag (``r56_chains.sh``), then its report."""
    parsed = []
    for item in segments.split():
        name, sep, start = item.partition(':')
        if not sep or not name or not start:
            raise BatchError(f"--segments: {item!r} is not NAME:START (START a state or 'start')")
        parsed.append((name, start))
    entries = []
    if not only_report:
        for name, start in parsed:
            args = ['--runs', str(runs), '--act', act, *(['--from', start] if start != 'start' else []), *chain_args]
            entries.append(chain_entry(name, args, options, output=chains_dir / f"{tag}-{name}.jsonl",
                                       log=chains_dir / f"{tag}-{name}.log", target=TARGETS.get(name)))
    if report or only_report:
        entries.append(report_entry(tag, [name for name, _ in parsed], chains_dir, options))
    return entries


def report_commands(tag: str, segments: list[str], chains_dir: Path, models_dir: Optional[str]) -> list[list[str]]:
    """``scripts/yardstick.py report`` over the segment outputs that exist (the full run also at every milestone),
    then ``triforce room-stats TAG``."""
    chains = []
    for name in segments:
        out = chains_dir / f"{tag}-{name}.jsonl"
        if not out.is_file() or out.stat().st_size == 0:
            continue
        target = TARGETS.get(name)
        chains += ['--chain', f"{name}={out}" + (f"@{target}" if target else '')]
        if name == 'full':
            sequence = ''
            for milestone, room in MILESTONES:
                sequence = f"{sequence},{room}" if sequence else room
                chains += ['--chain', f"full→{milestone}={out}@{sequence}"]
    models = ['--models-dir', models_dir] if models_dir else []
    return [[sys.executable, str(ROOT / 'scripts' / 'yardstick.py'), 'report', '--run', 'promoted', *models, *chains,
             '--output', str(chains_dir / f"{tag}-yardstick.json")],
            [sys.executable, '-m', 'triforce.cli', *models, 'room-stats', tag, '--chains-dir', str(chains_dir)]]


def report_entry(tag: str, segments: list[str], chains_dir: Path, options: Options) -> Entry:
    """The yardstick report and room statistics of a tag's chains (always run; commands built at launch)."""
    def build():
        return report_commands(tag, segments, chains_dir, options.models_dir)
    return Entry('report', 'report', build(), parallel=1, cwd=ROOT, events=False,
                 output=chains_dir / f"{tag}-yardstick.json", fields={'tag': tag}, complete=lambda _events: {},
                 builder=build)


def family_names(families: str, run_id: Optional[str], models_dir: Optional[str]) -> list[str]:
    """``all``: every specialist with a promoted run (or ``run_id``), in route order; else the listed names."""
    # pylint: disable=import-outside-toplevel
    from . import _native, registry
    from .specialist import list_specialists
    if families != 'all':
        return [name for name in families.split(',') if name]
    order = list(dict.fromkeys(step['specialist'] for step in _native.route_steps(_native.DEFAULT_ROUTE)))
    names = []
    for name in list_specialists():
        try:
            run = run_id or registry.promoted_run(name, models_dir)
        except FileNotFoundError:
            continue
        if registry.checkpoint_path(name, run, models_dir).is_file():
            names.append(name)
    return sorted(names, key=lambda n: (order.index(n) if n in order else len(order), n))


def families_entries(families: str, options: Options, *, acts: list[str], run_id: Optional[str],
                     episodes_per_state: Optional[int], seed: Optional[int],
                     clips: Optional[str] = None) -> list[Entry]:
    """One evaluation per family and act mode (``clips``: every evaluation's ``--clips``, None: each family's
    default)."""
    names = family_names(families, run_id, options.models_dir)
    if not names:
        raise BatchError(f"--families {families}: no specialist with a {'run ' + run_id if run_id else 'promoted run'}")
    entries = []
    for name in names:
        for act in acts:
            args = ['--specialist', name, '--act', act, *(['--run-id', run_id] if run_id else []),
                    *(['--episodes-per-state', str(episodes_per_state)] if episodes_per_state else []),
                    *(['--seed', str(seed)] if seed is not None else []),
                    *(['--clips', clips] if clips else [])]
            entry = name if acts == ['model'] else f"{name}-{act}"
            entries.append(eval_entry(entry + (f"-clips{clips}" if clips else ''), args, options))
    return entries


def split_entry(spec: str) -> tuple[Optional[str], list[str]]:
    """'d3: --from 3_7cs_0 --runs 100' -> ('d3', ['--from', '3_7cs_0', '--runs', '100']); no 'NAME:' -> None."""
    tokens = shlex.split(spec)
    if tokens and tokens[0].endswith(':') and not tokens[0].startswith('-') and len(tokens[0]) > 1:
        return tokens[0][:-1], tokens[1:]
    return None, tokens


# ---------------------------------------------------------------------------------------------------------- runner

class _Running:
    """A started entry: its process, the commands still to run, its console log."""

    def __init__(self, entry: Entry, stream: Path, log: Path, nice: int):
        self.entry = entry
        self.stream = stream
        self.started = time.monotonic()
        self.nice = nice
        self.failures = []
        if entry.builder is not None:
            entry.commands = entry.builder()
        self.commands = [list(command) for command in entry.commands]
        if entry.events and self.commands:
            self.commands[0] += ['--events', str(stream)]
        log.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log, 'w', encoding='utf-8')  # pylint: disable=consider-using-with
        self.process = None
        self._next()

    def _next(self):
        command = self.commands.pop(0)
        self.log.write(f"$ {shlex.join(command)}\n")
        self.log.flush()
        self.process = subprocess.Popen(  # pylint: disable=consider-using-with
            command, stdout=self.log, stderr=subprocess.STDOUT, cwd=self.entry.cwd,
            env={**child_env(self.nice), 'PYTHONUNBUFFERED': '1'})

    def poll(self) -> Optional[int]:
        """None while running; else the exit code (the first non-zero one of a command sequence; every command of
        the sequence runs)."""
        code = self.process.poll()
        if code is None:
            return None
        if code != 0:
            self.failures.append(code)
        if self.commands:
            self._next()
            return None
        self.log.close()
        return self.failures[0] if self.failures else 0

    def stop(self):
        """SIGTERM, then SIGKILL after ``TERMINATE_SECONDS``."""
        self.commands = []
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(TERMINATE_SECONDS)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()


def _tail(path: Path, lines: int = 20) -> str:
    try:
        with open(path, encoding='utf-8', errors='replace') as file:
            return ''.join(file.readlines()[-lines:])
    except OSError:
        return ''


def _fits(entry: Entry, running: list[Entry], cores: int, jobs: int) -> bool:
    """Whether ``entry`` may start next to ``running``: only family evaluations share, up to ``jobs`` of them and
    ``cores`` envs."""
    if not running:
        return True
    if entry.kind != 'eval' or any(other.kind != 'eval' for other in running):
        return False
    return len(running) < jobs and sum(other.parallel for other in running) + entry.parallel <= cores


def run_batch(name: str, entries: list[Entry], *, logs_dir=None, cores: int = 1, jobs: int = 1, nice: int = 0,
              command: Optional[list] = None, say: Callable[[str], None] = print, extra: Optional[dict] = None) -> int:
    """Runs ``entries`` under one log folder (module doc); returns 0 (all ok), 1 (any failed) or 130 (stopped)."""
    # pylint: disable=too-many-arguments, too-many-locals, too-many-statements, too-many-branches
    names = [entry.name for entry in entries]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise BatchError(f"entry names appear twice: {', '.join(duplicates)} (name them 'NAME: ARGS')")
    log_dir = new_log_dir(name, logs_dir)
    console = open(log_dir / 'round.log', 'a', encoding='utf-8')  # pylint: disable=consider-using-with

    def tell(text: str):
        say(text)
        console.write(text + '\n')
        console.flush()

    streams = [log_dir / f"{_safe(entry.name)}.jsonl" for entry in entries]
    logs = [entry.log or log_dir / f"{_safe(entry.name)}.log" for entry in entries]
    plan = TrainingPlan(log_dir, name, KIND, [entry.plan_entry(stream.name, log, log_dir)
                                              for entry, stream, log in zip(entries, streams, logs)],
                        command=command, nice=nice, cores=cores, jobs=jobs, events='round.jsonl', log='round.log',
                        **(extra or {}))
    events = EventLog(log_dir / 'round.jsonl', batch=name)
    events.emit('start', entries=names, nice=nice, cores=cores, jobs=jobs, log_dir=str(log_dir), pid=os.getpid())
    tell(f"[batch] {name}: {len(entries)} evaluations, nice {nice} (logs: {log_dir})")
    statuses: dict[int, str] = {}

    def finish(index: int, status: str, fields: dict, message: Optional[str] = None, log_tail: str = ''):
        entry = entries[index]
        stream = EventLog(streams[index], entry=entry.name)
        if message and status == 'failed':
            stream.emit('error', message=message, log_tail=log_tail)
        stream.emit('end', status=status, **fields, **({'message': message} if message else {}))
        plan.update_run(index, status=status, **fields, **({'error': message} if message else {}))
        events.emit('entry_end', index=index, name=entry.name, status=status, **fields)
        statuses[index] = status
        text = headline_text(fields) or message or ''
        label = 'done earlier' if fields.get('resumed') else status
        tell(f"[batch] {entry.name}: {label}" + (f" · {text}" if text else ''))

    pending = []
    for index, entry in enumerate(entries):
        done = None
        if entry.resume is not None:
            try:
                done = entry.resume()
            except Exception as error:  # pylint: disable=broad-exception-caught
                tell(f"[batch] {entry.name}: can't read the earlier result ({type(error).__name__}: {error}); "
                     f"measuring again")
        if done is None:
            pending.append(index)
            continue
        fields, stream_events = done
        stream = EventLog(streams[index], entry=entry.name)
        for kind, event in stream_events:
            stream.emit(kind, **event)
        finish(index, 'ok', {**fields, 'resumed': True, 'output': str(entry.output)})

    running: dict[int, _Running] = {}
    previous = signal.getsignal(signal.SIGTERM)
    installed = False
    try:
        try:
            signal.signal(signal.SIGTERM, _interrupt)
            installed = True
        except ValueError:          # not the main thread
            pass
        while pending or running:
            while pending and _fits(entries[pending[0]], [running[i].entry for i in running], cores, jobs):
                index = pending.pop(0)
                entry = entries[index]
                plan.update_run(index, status='evaluating')
                events.emit('entry_start', index=index, name=entry.name, kind=entry.kind)
                try:
                    if entry.prepare is not None:
                        entry.prepare()
                    running[index] = _Running(entry, streams[index], logs[index], nice)
                    tell(f"[batch] {entry.name}: started ({shlex.join(running[index].process.args)})")
                except Exception as error:  # pylint: disable=broad-exception-caught
                    finish(index, 'failed', {}, f"{type(error).__name__}: {error}", traceback.format_exc())
            time.sleep(POLL_SECONDS)
            for index in list(running):
                job = running[index]
                code = job.poll()
                if code is None:
                    continue
                del running[index]
                fields = {'wall_seconds': round(time.monotonic() - job.started, 1),
                          'output': None if job.entry.output is None else str(job.entry.output)}
                if code != 0:
                    finish(index, 'failed', fields, f"exit code {code} (log {logs[index]})", _tail(logs[index]))
                    continue
                try:
                    fields.update(job.entry.complete(_stream_events(streams[index])) if job.entry.complete else {})
                except Exception as error:  # pylint: disable=broad-exception-caught
                    finish(index, 'failed', fields, f"{type(error).__name__}: {error}", _tail(logs[index]))
                    continue
                finish(index, 'ok', fields)
                if job.entry.kind == 'report':
                    _print_report(logs[index], tell)
    except KeyboardInterrupt as stop:
        message = f"stopped ({stop or 'interrupt'})"
        for index, job in list(running.items()):
            job.stop()
            finish(index, 'interrupted', {'wall_seconds': round(time.monotonic() - job.started, 1)}, message)
        for index in pending:
            finish(index, 'interrupted', {}, message)
        events.emit('end', status='interrupted', message=message)
        plan.finish('interrupted')
        console.close()
        return EXIT_INTERRUPTED
    finally:
        if installed:
            signal.signal(signal.SIGTERM, previous)
    failed = [entries[i].name for i, status in sorted(statuses.items()) if status != 'ok']
    status = 'failed' if failed else 'completed'
    events.emit('end', status=status, **({'failed': failed} if failed else {}))
    plan.finish(status, **({'error': f"failed: {', '.join(failed)}"} if failed else {}))
    tell(f"[batch] {name}: {status}" + (f" ({', '.join(failed)} failed)" if failed else '') + f"  (logs: {log_dir})")
    console.close()
    return 1 if failed else 0


def _interrupt(signum, _frame):
    raise KeyboardInterrupt(f"signal {signum}")


def _print_report(log: Path, tell: Callable[[str], None]):
    """The report's chain table onward (what ``r56_chains.sh report`` printed)."""
    lines = _tail(log, 10_000).splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith('| segment')), None)
    for line in lines[start:] if start is not None else []:
        if not line.startswith('$ '):
            tell(line)


# ------------------------------------------------------------------------------------------------------------- CLI

class _Ordered(argparse.Action):
    """Appends (option, value) to ``args.entries`` so explicit entries keep their command-line order."""

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, 'entries', None) or []
        namespace.entries = [*items, (self.dest, values)]


def build_parser() -> argparse.ArgumentParser:
    """The command line."""
    parser = argparse.ArgumentParser(prog='triforce eval-batch', description=__doc__.split('\n\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split('\n\n', 1)[1])
    parser.add_argument('name', help="the batch's name: its log folder and the training GUI's group")
    parser.add_argument('--chain', action=_Ordered, metavar="'[ENTRY:] ARGS'", help="a triforce eval-chain entry")
    parser.add_argument('--evaluate', action=_Ordered, metavar="'[ENTRY:] ARGS'", help="a triforce evaluate entry")
    parser.add_argument('--chains', action=_Ordered, metavar='TAG', help="the chain segments of TAG, then the report")
    parser.add_argument('--families', action=_Ordered, metavar='all|a,b,c',
                        help="the evaluation of each family's promoted run (or --run-id)")
    chains = parser.add_argument_group('--chains')
    chains.add_argument('--act', choices=('model', 'sample', 'argmax'), default='sample',
                        help="act mode of the segments (default sample, as r56_chains.sh)")
    chains.add_argument('--runs', type=int, default=120, help="runs per segment (seeds 0..N-1)")
    chains.add_argument('--segments', default=SEGMENTS, help="NAME:START ... (START a state or 'start')")
    chains.add_argument('--chains-dir', type=Path, default=None, help="default <repo>/target/r56-chains")
    chains.add_argument('--chain-args', default='', help="more eval-chain arguments for every segment")
    chains.add_argument('--no-report', action='store_true', help="the segments only")
    chains.add_argument('--report-only', action='store_true', help="only the report (of the outputs that exist)")
    families = parser.add_argument_group('--families')
    families.add_argument('--acts', default='model', help="comma-separated act modes: model, sample, argmax")
    families.add_argument('--run-id', default=None, help="evaluate this run of every family instead of the promoted")
    families.add_argument('--episodes-per-state', type=int, default=None, help="default evaluate's (100)")
    families.add_argument('--seed', type=int, default=None, help="default evaluate's (0)")
    families.add_argument('--clips', choices=('on', 'off', 'both'), default=None,
                          help="every evaluation's clips switch states (default each family's: both for clips on)")
    parser.add_argument('--parallel', type=int, default=os.cpu_count() or 1,
                        help="core budget; a chain's envs unless its ARGS set --parallel (default every core)")
    parser.add_argument('--eval-parallel', type=int, default=4,
                        help="envs of each family evaluation unless its ARGS set --parallel")
    parser.add_argument('--jobs', type=int, default=1,
                        help="family evaluations side by side (within --parallel envs); chains always run alone")
    parser.add_argument('--redo', action='store_true', help="measure entries again that an earlier run measured")
    add_nice_argument(parser)
    parser.add_argument('--logs-dir', default=None, help=f"event logs root (default ${LOGS_DIR_ENV} or <repo>/logs)")
    parser.add_argument('--models-dir', default=None)
    parser.add_argument('--allow-incompatible-models', action='store_true',
                        help="load models whose version differs from this code's in its first three numbers")
    return parser


def build_entries(args) -> list[Entry]:
    """The batch's entries in command-line order; raises BatchError (or argparse's SystemExit) when one is wrong."""
    # pylint: disable=import-outside-toplevel
    from .registry import ACT_CHOICES
    from .room_stats import CHAINS_DIR
    options = Options(args.name, args.parallel, args.eval_parallel, args.models_dir, args.redo)
    entries = []
    for kind, value in getattr(args, 'entries', None) or []:
        if kind == 'chain':
            entries.append(chain_entry(*split_entry(value), options))
        elif kind == 'evaluate':
            entries.append(eval_entry(*split_entry(value), options))
        elif kind == 'chains':
            entries += chains_entries(value, options, act=args.act, runs=args.runs, segments=args.segments,
                                      chains_dir=Path(args.chains_dir or CHAINS_DIR).resolve(),
                                      chain_args=shlex.split(args.chain_args), report=not args.no_report,
                                      only_report=args.report_only)
        else:
            acts = [act for act in args.acts.split(',') if act]
            wrong = [act for act in acts if act not in ACT_CHOICES]
            if wrong or not acts:
                raise BatchError(f"--acts {args.acts}: each must be one of {', '.join(ACT_CHOICES)}")
            entries += families_entries(value, options, acts=acts, run_id=args.run_id,
                                        episodes_per_state=args.episodes_per_state, seed=args.seed, clips=args.clips)
    if not entries:
        raise BatchError("nothing to run: give --chains, --families, --chain or --evaluate")
    return entries


def main(argv=None) -> int:
    """``triforce eval-batch``: 0 every evaluation ok, 1 any failed, 2 a batch that can't run, 130 stopped."""
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    from .registry import set_allow_incompatible  # pylint: disable=import-outside-toplevel
    set_allow_incompatible(args.allow_incompatible_models)
    if args.parallel < 1 or args.eval_parallel < 1 or args.jobs < 1:
        print("triforce eval-batch: error: --parallel, --eval-parallel and --jobs must be >= 1", file=sys.stderr)
        return 2
    nice = lower_priority(args.nice)
    try:
        entries = build_entries(args)
        return run_batch(args.name, entries, logs_dir=args.logs_dir, cores=args.parallel, jobs=args.jobs, nice=nice,
                         command=[sys.executable, '-m', 'triforce.eval_batch', *argv],
                         extra={'models_dir': args.models_dir})
    except BatchError as error:
        print(f"triforce eval-batch: error: {error}", file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
