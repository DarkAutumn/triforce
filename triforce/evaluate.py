"""Evaluate one specialist on its own rooms: N episodes from every seed start state.

    python -m triforce.evaluate --specialist NAME [--checkpoint PATH | --run-id ID] [--models-dir DIR]
        [--episodes-per-state 100] [--parallel 1] [--threads N] [--seed 0] [--act model|sample|argmax]
        [--output PATH] [--replays DIR [--streams LIST] [--keyframe-frames N] [--state embed|path]]

Training metrics mix start states and ``eval-chain`` plays the whole route; this measures one model in
isolation.  Every seed savestate of the specialist (``SpecialistConfig.resolve_start_states``) is played
``--episodes-per-state`` times in a native ``EvalTask`` pool of ``--parallel`` envs (same rewards, end conditions
and RAM randomization as training), episode ``i`` seeded with ``seed + i``.  The network acts inside the pool
(``EnvPool.play`` with its Rust copy, docs/rust-actor.md) the way the run acts (``--act model``, the default: its
registry act mode, ``registry.act_mode``) or as told (``sample`` / ``argmax``); sampling uses Gumbel noise keyed to
(episode seed, decision), so an episode plays the same whichever env or thread runs it.
``--replays`` records every episode as a replay file (``triforce.replay``).  The checkpoint defaults to the
specialist's promoted model; records are sorted by seed.  The result goes to ``<checkpoint dir>/eval.json`` only
for the canonical evaluation (the run's own act mode, ``--episodes-per-state 100``, ``--seed 0``, and the
specialist the checkpoint was trained for); otherwise to ``eval[-<specialist>][-<act>][-e<N>][-s<seed>].json``,
each part only where it differs (``default_output``).  Writes are atomic (``write_result``).  Every result names
what produced it: ``act`` (``sample``/``argmax``) and ``deterministic``, ``checkpoint_sha1`` (of ``ckpt.pt``),
``model_hash`` (the Rust network's), ``config`` (the specialist config as JSON), ``code`` and ``data``
(``replay.code_version``/``data_hash``).  ``read_evaluation`` finds a run's evaluation in a given act mode.
``mean_health_change`` is hearts at the episode's end minus hearts at its start (after hp randomization);
``mean_start_health_fraction`` divides the start hearts by the max hearts at the start (``start_max_health``; a
heart container picked up later doesn't count), ``mean_health_fraction`` the end hearts by the max at the end.
Dropped hearts are kept per episode (``records[i]['hearts']``); ``mean_health_change_total``
adds the hearts wasted by picking them up at (or near) full health, and ``hearts_dropped`` / ``hearts_taken`` /
``hearts_left`` are means per episode (also per room and per start state).  ``mean_bombs_used`` are bombs placed per
episode (``records[i]['bombs']``), ``mean_bomb_<stat>`` its ``_native.BOMB_STATS`` (placed, hits, misses, damage,
tenth_kills, tenth_tries, armed, picked_up: crates/zelda/src/bombs.rs) and ``mean_cheated_bombs`` / ``mean_cheated_keys`` / ``mean_cheated_rupees``
cheated items used per episode (``records[i]['cheated']``: a room that needs an item Link lacks grants it, a bomb in
``requires_bombs`` rooms, a key at a locked route door, rupees in ``requires_rupees`` shops:
crates/zelda/src/cheats.rs), also per room and per start state.  ``mean_boss_stuns`` / ``mean_boss_eats`` are
Dodongo stuns (a bomb's smoke; the sword can kill him then) and bombs he swallowed per episode
(``records[i]['boss']``: ``stuns``, ``eats``;
crates/env/src/critic.rs ``BossTransitions``), and ``stun_kill_rate`` the share of successful episodes won by
stun-and-sword alone (a stun, no swallow), also per room and per start state.
Lingering records retain longest quiet streak, terminal tail, visit time and boss/pickup milestones. Overall,
room and start-state aggregates report mean longest/tail steps and frames plus the worst measured streak.

``format: 2`` evaluations also keep every room visit (``zelda::visits``): ``records[i]['visits']`` lists the
episode's visits in order (``level``, ``room``, ``in_cave``, ``steps``, ``frames``, ``health_in``/``health_out``,
``max_health``, ``health_lost``/``health_gained`` summed over the visit, ``hits`` = steps that lost health,
``hits_dealt``, ``drops``/``pickups`` per item kind (bombs, rupee, blue_rupee, heart, fairy, clock; zeros
omitted), ``keys_gained`` (natural keys: a cheated key is no gain), ``bombs_used`` (bombs placed),
``bomb_<stat>`` (``_native.BOMB_STATS``), ``cheated_bombs``/``cheated_keys``/``cheated_rupees`` (cheated items used),
``boss_stuns``/``boss_eats`` (Dodongo stunned, bombs swallowed) and ``exit``:
``n``/``s``/``e``/``w`` or ``cave`` when Link left the room, else how the episode ended there,
``death``/``success``/``timeout``/``other``).  ``visited_rooms`` aggregates them per
room entered (keyed like ``by_room``, which only knows start rooms): ``visits``, ``episodes``, ``deaths``,
``death_rate``, ``exits`` and per-visit means (``steps_per_visit``, ``health_lost/gained/change_per_visit``,
``hits_per_visit``, ``no_hit_rate``, ``hits_dealt_per_visit``, ``keys_per_visit``, ``bombs_used_per_visit``,
``bomb_<stat>_per_visit``,
``cheated_bombs_per_visit``, ``cheated_keys_per_visit``, ``cheated_rupees_per_visit``, ``boss_stuns_per_visit``,
``boss_eats_per_visit``, ``drops_per_visit``,
``pickups_per_visit``), tagged with the ``specialist``.  Readers
merging several evaluations weight by visits.
"""

import argparse
import json
import time
import traceback
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Optional

from . import __version__, _native, registry, replay
from .events import LOGS_DIR_ENV, EventLog, logged_job, write_json_atomic
from .metrics import heart_change_total
from .provenance import code_version, data_hash, models_header
from .replay import ReplayDir
from .specialist import SpecialistConfig, load_specialist


FORMAT = 2
DEFAULT_EPISODES_PER_STATE = 100
DEFAULT_SEED = 0
# Dropped-item kinds of the visit records (zelda::visits::ITEM_KINDS).
ITEM_KINDS = ('bombs', 'rupee', 'blue_rupee', 'heart', 'fairy', 'clock')


def _room_key(room) -> str:
    level, location, in_cave = room
    return f"{level}:0x{location:02x}" + (":cave" if in_cave else "")


def _play(pool, policy, deterministic: bool, total: int, on_episode: Optional[Callable[[dict, int, int], None]],
          replays: Optional[ReplayDir], distributions: bool) -> list[dict]:
    """Lets the eval pool play until every job ran; returns the records in completion order.  ``replays``: the
    folder every episode's recording (a ``replay`` event) is written to."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    records = []
    pool.play([policy], deterministic=deterministic, distributions=distributions)
    while True:
        idle = pool.wait_idle(timeout=0.1)
        for event in pool.poll_events():
            if event['type'] == 'eval':
                records.append(event['record'])
                if on_episode is not None:
                    on_episode(records[-1], len(records), total)
            elif event['type'] == 'replay' and replays is not None:
                replays.add(int(event['job']), event['data'])
        if idle:
            return records


def _heart_stats(records: list[dict]) -> dict:
    """Means per episode of the total heart change (real + wasted) and the dropped-heart counts."""
    count = len(records)
    hearts = [r.get('hearts') or {} for r in records]
    return {'health_change_total': sum(heart_change_total(r['health'] - r['start_health'], h)
                                       for r, h in zip(records, hearts)) / count,
            'hearts_dropped': sum(h.get('dropped', 0) for h in hearts) / count,
            'hearts_taken': sum(h.get('taken', 0) for h in hearts) / count,
            'hearts_left': sum(h.get('left', 0) for h in hearts) / count}


def _linger_stats(records: list[dict]) -> dict:
    """Means and maxima over measured episodes only; historical absence is not a zero streak."""
    measured = [r['linger'] for r in records if r.get('linger')]
    if not measured:
        return {}
    result = {f'linger_{key}': sum(linger[key] for linger in measured) / len(measured)
              for key in ('max_steps', 'max_frames', 'tail_steps', 'tail_frames')}
    result.update({f'linger_worst_{unit}': max(linger[f'max_{unit}'] for linger in measured)
                   for unit in ('steps', 'frames')})
    return result


def _block_stats(records: list[dict]) -> dict:
    measured = [record['blocks'] for record in records if 'blocks' in record]
    return {'blocks': sum(measured) / len(measured)} if measured else {}


def _cheat_stats(records: list[dict]) -> dict:
    """Bombs placed and cheated items used per episode, over records that count them."""
    result = {}
    bombs = [record['bombs'] for record in records if 'bombs' in record]
    if bombs:
        result['mean_bombs_used'] = sum(b['used'] for b in bombs) / len(bombs)
        result.update({f'mean_bomb_{stat}': sum(b.get(stat, 0) for b in bombs) / len(bombs)
                       for stat in _native.BOMB_STATS})
    cheated = [record['cheated'] for record in records if 'cheated' in record]
    if cheated:
        result.update({f'mean_cheated_{item}': sum(c[item] for c in cheated) / len(cheated)
                       for item in _native.CHEAT_ITEMS})
    return result


def _boss_stats(records: list[dict]) -> dict:
    """Dodongo stuns and swallowed bombs per episode, over records that count them, and ``stun_kill_rate``: the share
    of those that succeeded with at least one stun and no swallow (absent without a measured success)."""
    measured = [record for record in records if 'boss' in record]
    if not measured:
        return {}
    result = {'mean_boss_stuns': sum(r['boss']['stuns'] for r in measured) / len(measured),
              'mean_boss_eats': sum(r['boss']['eats'] for r in measured) / len(measured)}
    wins = [r['boss'] for r in measured if r['success']]
    if wins:
        result['stun_kill_rate'] = sum(b['stuns'] > 0 and b['eats'] == 0 for b in wins) / len(wins)
    return result


def _group(records: list[dict], key) -> dict:
    groups = {}
    for record in records:
        groups.setdefault(key(record), []).append(record)
    return {name: {'episodes': len(items),
                   'success_rate': sum(r['success'] for r in items) / len(items),
                   'health_change': sum(r['health'] - r['start_health'] for r in items) / len(items),
                   **_heart_stats(items),
                   **_linger_stats(items),
                   **_block_stats(items),
                   **_cheat_stats(items),
                   **_boss_stats(items),
                   'endings': dict(Counter(r['ending'] for r in items)),
                   # how the unsuccessful episodes ended, e.g. {'death': 4}
                   'failures': dict(Counter(r['ending'] for r in items if not r['success']))}
            for name, items in sorted(groups.items())}


def _room_visit_stats(visits: list[dict]) -> dict:
    """Counts, exits and means over one room's visits."""
    count = len(visits)

    def mean(value) -> float:
        return sum(value(visit) for visit in visits) / count

    def items(field: str) -> dict:
        totals = Counter()
        for visit in visits:
            totals.update(visit[field])
        return {kind: totals[kind] / count for kind in ITEM_KINDS if totals[kind]}

    deaths = sum(visit['exit'] == 'death' for visit in visits)
    return {
        'level': visits[0]['level'], 'room': visits[0]['room'], 'in_cave': visits[0]['in_cave'],
        'visits': count, 'deaths': deaths, 'death_rate': deaths / count,
        'steps_per_visit': mean(lambda v: v['steps']),
        'health_lost_per_visit': mean(lambda v: v['health_lost']),
        'health_gained_per_visit': mean(lambda v: v['health_gained']),
        'health_change_per_visit': mean(lambda v: v['health_out'] - v['health_in']),
        'hits_per_visit': mean(lambda v: v['hits']),
        'no_hit_rate': mean(lambda v: v['hits'] == 0),
        'hits_dealt_per_visit': mean(lambda v: v['hits_dealt']),
        'keys_per_visit': mean(lambda v: v['keys_gained']),
        'bombs_used_per_visit': mean(lambda v: v.get('bombs_used', 0)),
        **{f'bomb_{stat}_per_visit': mean(lambda v, stat=stat: v.get(f'bomb_{stat}', 0))
           for stat in _native.BOMB_STATS},
        **{f'cheated_{item}_per_visit': mean(lambda v, item=item: v.get(f'cheated_{item}', 0))
           for item in _native.CHEAT_ITEMS},
        'boss_stuns_per_visit': mean(lambda v: v.get('boss_stuns', 0)),
        'boss_eats_per_visit': mean(lambda v: v.get('boss_eats', 0)),
        'drops_per_visit': items('drops'),
        'pickups_per_visit': items('pickups'),
        'exits': dict(Counter(visit['exit'] for visit in visits)),
    }


def _visited_rooms(records: list[dict], specialist: str) -> dict:
    """Per room any episode entered (``records[i]['visits']``): ``_room_visit_stats`` plus the number of
    episodes that entered it and the evaluated specialist."""
    rooms = {}
    for index, record in enumerate(records):
        for visit in record.get('visits', ()):
            key = _room_key((visit['level'], visit['room'], visit['in_cave']))
            rooms.setdefault(key, []).append((index, visit))
    return {key: {**_room_visit_stats([visit for _, visit in entries]),
                  'episodes': len({index for index, _ in entries}), 'specialist': specialist}
            for key, entries in sorted(rooms.items())}


def evaluate_specialist(specialist: str | SpecialistConfig, checkpoint, episodes_per_state: int = 100,
                        parallel: int = 1, seed: int = 0, deterministic: Optional[bool] = None,
                        on_episode: Optional[Callable[[dict, int, int], None]] = None, *,
                        threads: Optional[int] = None, states_dir=None, events=None, replays=None,
                        streams=(), state: str = 'embed',
                        keyframe_frames: int = replay.DEFAULT_KEYFRAME_FRAMES) -> dict:
    """Plays every seed start state `episodes_per_state` times on `parallel` envs and returns the summary dict
    (see module doc).  ``on_episode(record, done, total)`` is called as each episode finishes (completion order).
    ``deterministic``: argmax (True) or sampling (False); None follows the run's act mode (``registry.act_mode``).
    ``threads``: the pool's worker threads (default min(parallel, cores)); ``states_dir``: where start states
    resolve (default the states directory); ``events``: an ``EventLog`` that
    gets ``eval_start`` and ``eval_progress`` (at most once a second) for the training GUI (the caller emits
    ``eval``, ``eval_event`` fields, once it has written the result).  ``replays``: a folder every episode is
    recorded into as ``<seed>-<ending>.tfreplay`` (``streams``, ``state`` and ``keyframe_frames`` as
    ``replay.record_options``)."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    config = load_specialist(specialist) if isinstance(specialist, str) else specialist
    states_dir = None if states_dir is None else str(states_dir)
    states = sorted(Path(p).stem for p in config.resolve_start_states(states_dir))
    if not states:
        raise ValueError(f"{config.name} has no seed start states")
    episode_states = [state for state in states for _ in range(episodes_per_state)]
    jobs = [(state, seed + index) for index, state in enumerate(episode_states)]

    registry.require_compatible(checkpoint)
    policy = registry.load_model(checkpoint)[0].to_native()
    if deterministic is None:
        deterministic = registry.act_mode(checkpoint) == 'argmax'
    checkpoint_sha1 = registry.checkpoint_sha1(checkpoint)
    record = None
    if replays is not None:
        models = models_header([config.name], [policy], {config.name: str(checkpoint)}, [deterministic])
        record = replay.record_options(replay.meta('eval', models, specialist=config.name), streams, state,
                                       keyframe_frames)
        replays = ReplayDir(replays)
    task = _native.EvalTask(config, jobs, states_dir=states_dir)
    if events is not None:
        events.emit('eval_start', checkpoint=str(checkpoint), episodes_per_state=episodes_per_state,
                    episodes=len(jobs), parallel=parallel, deterministic=deterministic)
        on_episode = _progress_events(events, on_episode)
    with _native.EnvPool(task, num_envs=max(1, min(parallel, len(jobs))), threads=threads, seed=seed,
                         record=record) as pool:
        records = _play(pool, policy, deterministic, len(jobs), on_episode, replays,
                        bool(record and record['decisions']))
    records.sort(key=lambda r: r['seed'])

    count = len(records)
    hearts = _heart_stats(records)
    return {
        'format': FORMAT,
        'specialist': config.name, 'checkpoint': str(checkpoint), 'episodes_per_state': episodes_per_state,
        'seed': seed, 'deterministic': deterministic, 'act': 'argmax' if deterministic else 'sample',
        'checkpoint_sha1': checkpoint_sha1, 'model_hash': f'{policy.hash:016x}',
        'config': json.loads(config.to_json()), 'version': __version__, 'code': code_version(),
        'data': data_hash(),
        'episodes': count,
        'success_rate': sum(r['success'] for r in records) / count,
        'mean_health_fraction': sum(r['health'] / r['max_health'] if r['max_health'] else 0.0
                                    for r in records) / count,
        # the same at reset (hp is randomized there), so the end figure has something to compare with
        'mean_start_health_fraction': _start_health_fraction(records),
        'mean_health_change': sum(r['health'] - r['start_health'] for r in records) / count,
        'mean_health_change_total': hearts['health_change_total'],
        **hearts,
        **_linger_stats(records),
        **_block_stats(records),
        **_cheat_stats(records),
        **_boss_stats(records),
        'mean_steps': sum(r['steps'] for r in records) / count,
        'mean_frames': sum(r['frames'] for r in records) / count,
        'outcomes': dict(Counter(r['outcome'] for r in records)),
        'endings': dict(Counter(r['ending'] for r in records)),
        'by_room': _group(records, lambda r: _room_key(r['room'])),
        'by_state': _group(records, lambda r: r['start_state']),
        'visited_rooms': _visited_rooms(records, config.name),
        'records': records,
    }


def _start_health_fraction(records: list[dict]) -> float:
    """Mean start hearts over the max hearts at the start (``start_max_health``; ``max_health`` in records that
    predate it)."""
    def fraction(record):
        maximum = record.get('start_max_health', record['max_health'])
        return record['start_health'] / maximum if maximum else 0.0
    return sum(map(fraction, records)) / len(records)


def write_result(result: dict, path) -> Path:
    """Writes an evaluation result atomically (the file is the previous one or complete); returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_json_atomic(path, result)


def default_output(checkpoint, specialist: str, act: str, episodes_per_state: int, seed: int) -> Path:
    """Where an evaluation goes by default: ``<run dir>/eval.json`` for the canonical one, else
    ``eval[-<specialist>][-<act>][-e<N>][-s<seed>].json`` with each part only where it differs from canonical:
    ``specialist`` other than the checkpoint's, ``act`` (``model|sample|argmax``) other than the run's act mode,
    episodes per state other than 100, seed other than 0."""
    path = Path(checkpoint)
    path = path / registry.CHECKPOINT_NAME if path.is_dir() else path
    parts = ['eval']
    if registry.load_checkpoint(path, check_schema=False).get('specialist') != specialist:
        parts.append(specialist)
    if act not in registry.ACT_CHOICES:
        raise ValueError(f"act {act!r} is not one of {registry.ACT_CHOICES}")
    if act != 'model' and act != registry.act_mode(path):
        parts.append(act)
    if episodes_per_state != DEFAULT_EPISODES_PER_STATE:
        parts.append(f'e{episodes_per_state}')
    if seed != DEFAULT_SEED:
        parts.append(f's{seed}')
    return path.parent / ('-'.join(parts) + '.json')


def read_evaluation(run_dir, mode: Optional[str] = None) -> tuple[Path, dict]:
    """(path, result) of the evaluation of the run's current checkpoint in ``mode`` (default: the run's act mode):
    ``eval.json`` if it was measured in that mode and (when it records one) for the current ``ckpt.pt`` sha1, else
    ``eval-<mode>.json`` under the same checks.  Raises LookupError naming what it found otherwise."""
    path = Path(run_dir)
    checkpoint = path / registry.CHECKPOINT_NAME if path.is_dir() else path
    mode = registry.act_mode(checkpoint) if mode is None else mode
    if mode not in registry.ACT_MODES:
        raise ValueError(f"act mode {mode!r} is not one of {registry.ACT_MODES}")
    sha1 = registry.checkpoint_sha1(checkpoint)
    found = []
    for name in ('eval.json', f'eval-{mode}.json'):
        candidate = checkpoint.parent / name
        if not candidate.is_file():
            continue
        try:
            result = json.loads(candidate.read_text(encoding='utf-8'))
        except ValueError as error:
            found.append(f"{name} is unreadable ({error})")
            continue
        measured = registry.eval_mode(result)
        if measured != mode:
            found.append(f"{name} was measured by {measured}")
        elif result.get('checkpoint_sha1') not in (None, sha1):
            found.append(f"{name} is of other weights (sha1 {result['checkpoint_sha1']})")
        else:
            return candidate, result
    raise LookupError(f"{checkpoint.parent}: no {mode} evaluation of the current checkpoint "
                      f"({'; '.join(found) if found else f'no eval.json or eval-{mode}.json'})")


def _progress_events(events, on_episode):
    """``on_episode`` that also emits ``eval_progress`` (done, episodes, running success_rate) at most once a
    second and on the last episode."""
    progress = {'successes': 0, 'last': 0.0}

    def emit(record: dict, done: int, total: int):
        progress['successes'] += bool(record['success'])
        now = time.monotonic()
        if now - progress['last'] >= 1.0 or done == total:
            progress['last'] = now
            events.emit('eval_progress', done=done, episodes=total, success_rate=progress['successes'] / done)
        if on_episode is not None:
            on_episode(record, done, total)
    return emit


def eval_event(result: dict) -> dict:
    """The fields of an event stream's ``eval`` event (triforce/events.py) for an ``evaluate_specialist`` result:
    headline numbers, per-room and per-state success, room failures and the flat ``linger_*`` stats plus
    ``by_room_linger`` / ``by_state_linger``."""
    linger = {key: value for key, value in result.items() if key.startswith('linger_')}
    linger_groups = {
        f'{group}_linger': {name: {key: value for key, value in stats.items() if key.startswith('linger_')}
                            for name, stats in result[group].items() if 'linger_max_steps' in stats}
        for group in ('by_room', 'by_state')}
    return {'success_rate': result['success_rate'], 'episodes': result['episodes'],
            'deterministic': result['deterministic'],
            'health': result['mean_health_fraction'], 'start_health': result['mean_start_health_fraction'],
            'health_change': result['mean_health_change'], 'health_change_total': result['mean_health_change_total'],
            'hearts_dropped': result['hearts_dropped'], 'hearts_taken': result['hearts_taken'],
            'hearts_left': result['hearts_left'],
            'by_room': {room: stats['success_rate'] for room, stats in result['by_room'].items()},
            'by_room_failures': {room: {'episodes': stats['episodes'], 'failures': stats['failures']}
                                 for room, stats in result['by_room'].items()},
            'by_state': {state: stats['success_rate'] for state, stats in result['by_state'].items()},
            **linger, **linger_groups}


def format_report(result: dict) -> str:
    """Total line plus one line per room, worst success rate first.  Health change is real (total incl. hearts
    wasted at full health); hearts are dropped/taken/left per episode."""
    successes = sum(r['success'] for r in result['records'])
    lines = [f"{result['specialist']}: success {successes}/{result['episodes']} = {result['success_rate']:.0%}  "
             f"health change {result['mean_health_change']:+.2f} (total {result['mean_health_change_total']:+.2f}) "
             f"hearts  dropped/taken/left {_fmt_hearts(result)}  steps {result['mean_steps']:.0f}"
             f"{_fmt_linger(result)}"]
    for room, stats in sorted(result['by_room'].items(), key=lambda item: (item[1]['success_rate'], item[0])):
        top = max(stats['endings'].items(), key=lambda item: item[1])[0]
        lines.append(f"  {room:12} {stats['success_rate']:5.0%}  {stats['health_change']:+5.2f} "
                     f"({stats['health_change_total']:+5.2f}) hearts  {_fmt_hearts(stats)}  "
                     f"{stats['episodes']:4}  {top}{_fmt_linger(stats)}")
    return '\n'.join(lines)


def _fmt_hearts(stats: dict) -> str:
    return f"{stats['hearts_dropped']:.2f}/{stats['hearts_taken']:.2f}/{stats['hearts_left']:.2f}"


def _fmt_linger(stats: dict) -> str:
    if 'linger_max_steps' not in stats:
        return ''
    return (f"  linger max/tail/worst {stats['linger_max_steps']:.0f}/{stats['linger_tail_steps']:.0f}/"
            f"{stats['linger_worst_steps']:.0f} steps "
            f"({stats['linger_max_frames']:.0f}/{stats['linger_tail_frames']:.0f}/"
            f"{stats['linger_worst_frames']:.0f} frames)")


def main(argv=None):
    """Command line."""
    parser = argparse.ArgumentParser(prog='python -m triforce.evaluate', description=__doc__.split('\n\n')[0])
    parser.add_argument('--specialist', required=True)
    parser.add_argument('--checkpoint', default=None, help="checkpoint file or run dir (default: promoted)")
    parser.add_argument('--run-id', default=None, help="evaluate <models>/<specialist>/<run-id>/ckpt.pt")
    parser.add_argument('--models-dir', default=None, help=f"default ${registry.MODELS_DIR_ENV} or <repo>/models")
    parser.add_argument('--episodes-per-state', type=int, default=DEFAULT_EPISODES_PER_STATE)
    parser.add_argument('--parallel', type=int, default=1, help="envs in the evaluation pool")
    parser.add_argument('--threads', type=int, default=None, help="pool worker threads (default min(parallel, cores))")
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--act', choices=registry.ACT_CHOICES, default='model',
                        help="model: the run's act mode (registry); sample / argmax: act that way")
    parser.add_argument('--output', default=None,
                        help="JSON output (default: <checkpoint dir>/eval.json for the canonical evaluation, else "
                             "eval[-<specialist>][-<act>][-e<N>][-s<seed>].json)")
    replay.add_arguments(parser, default_state='embed')
    parser.add_argument('--logs-dir', default=None, help=f"event logs root (default ${LOGS_DIR_ENV} or <repo>/logs)")
    parser.add_argument('--no-log', action='store_true', help="no event-log folder (the training GUI won't show it)")
    parser.add_argument('--allow-incompatible-models', action='store_true',
                        help="load a model whose version differs from this code's in its first three numbers")
    args = parser.parse_args(argv)
    registry.set_allow_incompatible(args.allow_incompatible_models)
    try:
        streams = replay.parse_streams(args.streams)
    except ValueError as error:
        parser.error(str(error))
    if streams and args.replays is None:
        parser.error("--streams needs --replays")

    config = load_specialist(args.specialist)
    name = config.name
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        checkpoint = checkpoint / registry.CHECKPOINT_NAME if checkpoint.is_dir() else checkpoint
    elif args.run_id:
        checkpoint = registry.checkpoint_path(name, args.run_id, args.models_dir)
    else:
        checkpoint = registry.promoted_checkpoint(name, args.models_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    run_id = checkpoint.parent.name
    log = nullcontext() if args.no_log else logged_job(
        f'eval-{name}', 'eval', [{'specialist': name, 'run_id': run_id, 'events': f'{name}.jsonl',
                                  'run_dir': str(checkpoint.parent), 'status': 'evaluating'}], args.logs_dir)
    with log as plan:
        events = None if plan is None else EventLog(plan.directory / f'{name}.jsonl', specialist=name, run_id=run_id)
        try:
            result = evaluate_specialist(config, checkpoint, args.episodes_per_state, args.parallel, args.seed,
                                         registry.act_override(args.act), threads=args.threads, events=events,
                                         replays=args.replays, streams=streams, state=args.state,
                                         keyframe_frames=args.keyframe_frames)
        except Exception as error:
            if events is not None:
                message = f"evaluation failed: {type(error).__name__}: {error}"
                events.emit('error', message=message, traceback=traceback.format_exc())
                events.emit('end', status='eval-failed', message=message)
            raise
        output = (Path(args.output) if args.output else
                  default_output(checkpoint, name, args.act, args.episodes_per_state, args.seed))
        write_result(result, output)
        if events is not None:
            summary = {'eval_success_rate': result['success_rate'], 'eval_episodes': result['episodes']}
            events.emit('eval', **eval_event(result), path=str(output))
            events.emit('end', status='ok', **summary)
            plan.update_run(0, status='ok', **summary)
    print(format_report(result))
    print(f"(written to {output})")
    return 0


if __name__ == '__main__':
    main()
