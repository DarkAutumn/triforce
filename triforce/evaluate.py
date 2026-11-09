"""Evaluate one specialist on its own rooms: N episodes from every seed start state.

    python -m triforce.evaluate --specialist NAME [--checkpoint PATH | --run-id ID] [--models-dir DIR]
        [--episodes-per-state 100] [--parallel 1] [--threads N] [--device cpu] [--seed 0] [--deterministic]
        [--output PATH]

Training metrics mix start states and ``eval-chain`` plays the whole route; this measures one model in
isolation.  Every seed savestate of the specialist (``SpecialistConfig.resolve_start_states``) is played
``--episodes-per-state`` times in a native ``EvalTask`` pool of ``--parallel`` envs (same rewards, end conditions
and RAM randomization as training), episode ``i`` seeded with ``seed + i``.  Envs run asynchronously: each one
that is ready gets its action from a fixed-shape forward of the network (``inference.PoolInference``).  Actions are
sampled unless ``--deterministic``, with Gumbel noise keyed to (episode seed, decision), so an episode plays the
same whichever env or thread runs it.  The checkpoint defaults to the specialist's promoted model; results go to
``<checkpoint dir>/eval.json``, records sorted by seed.
``mean_health_change`` is hearts at the episode's end minus hearts at its start (after hp randomization).
Dropped hearts are kept per episode (``records[i]['hearts']``); ``mean_health_change_total``
adds the hearts wasted by picking them up at (or near) full health, and ``hearts_dropped`` / ``hearts_taken`` /
``hearts_left`` are means per episode (also per room and per start state).  ``mean_bombs_used`` are bombs placed per
episode (``records[i]['bombs']``) and ``mean_cheated_bombs`` / ``mean_cheated_keys`` / ``mean_cheated_rupees``
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
``cheated_bombs``/``cheated_keys``/``cheated_rupees`` (cheated items used),
``boss_stuns``/``boss_eats`` (Dodongo stunned, bombs swallowed) and ``exit``:
``n``/``s``/``e``/``w`` or ``cave`` when Link left the room, else how the episode ended there,
``death``/``success``/``timeout``/``other``).  ``visited_rooms`` aggregates them per
room entered (keyed like ``by_room``, which only knows start rooms): ``visits``, ``episodes``, ``deaths``,
``death_rate``, ``exits`` and per-visit means (``steps_per_visit``, ``health_lost/gained/change_per_visit``,
``hits_per_visit``, ``no_hit_rate``, ``hits_dealt_per_visit``, ``keys_per_visit``, ``bombs_used_per_visit``,
``cheated_bombs_per_visit``, ``cheated_keys_per_visit``, ``cheated_rupees_per_visit``, ``boss_stuns_per_visit``,
``boss_eats_per_visit``, ``drops_per_visit``,
``pickups_per_visit``), tagged with the ``specialist``.  Readers
merging several evaluations weight by visits.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import _native, registry
from .inference import PoolInference
from .metrics import heart_change_total
from .specialist import SpecialistConfig, load_specialist


FORMAT = 2
# Dropped-item kinds of the visit records (zelda::visits::ITEM_KINDS).
ITEM_KINDS = ('bombs', 'rupee', 'blue_rupee', 'heart', 'fairy', 'clock')


def _room_key(room) -> str:
    level, location, in_cave = room
    return f"{level}:0x{location:02x}" + (":cave" if in_cave else "")


def _load_network(checkpoint):
    network, _ = registry.load_model(checkpoint)
    return network.eval()


def _play(pool, network, deterministic: bool, total: int, device: str,
          on_episode: Optional[Callable[[dict, int, int], None]]) -> list[dict]:
    """Steps the eval pool asynchronously until every job ran; returns the records in completion order."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    buffers = pool.make_buffers()
    inference = PoolInference([network], pool.num_envs, device, deterministic)
    records = []

    def collect():
        for event in pool.poll_events():
            if event['type'] == 'eval':
                records.append(event['record'])
                if on_episode is not None:
                    on_episode(records[-1], len(records), total)

    while pool.active:
        count = pool.recv(buffers, min_ready=1)
        rows = np.flatnonzero(~buffers['finished'][:count])
        if len(rows):
            envs = buffers['env_ids'][rows]
            obs = {key: value[rows] for key, value in buffers['obs'].items()}
            pool.send(envs, inference.actions(envs, obs, np.zeros(len(rows), np.int64), buffers['job'][rows],
                                              buffers['step'][rows]))
        collect()
    collect()
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
                        parallel: int = 1, seed: int = 0, deterministic: bool = False,
                        on_episode: Optional[Callable[[dict, int, int], None]] = None, *,
                        threads: Optional[int] = None, device: str = 'cpu', states_dir=None) -> dict:
    """Plays every seed start state `episodes_per_state` times on `parallel` envs and returns the summary dict
    (see module doc).  ``on_episode(record, done, total)`` is called as each episode finishes (completion order).
    ``threads``: the pool's worker threads (default min(parallel, cores)); ``device``: where the network runs;
    ``states_dir``: where start states resolve (default the states directory)."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    config = load_specialist(specialist) if isinstance(specialist, str) else specialist
    states_dir = None if states_dir is None else str(states_dir)
    states = sorted(Path(p).stem for p in config.resolve_start_states(states_dir))
    if not states:
        raise ValueError(f"{config.name} has no seed start states")
    episode_states = [state for state in states for _ in range(episodes_per_state)]
    jobs = [(state, seed + index) for index, state in enumerate(episode_states)]

    network = _load_network(checkpoint)
    task = _native.EvalTask(config, jobs, states_dir=states_dir)
    with _native.EnvPool(task, num_envs=max(1, min(parallel, len(jobs))), threads=threads, seed=seed) as pool:
        records = _play(pool, network, deterministic, len(jobs), device, on_episode)
    records.sort(key=lambda r: r['seed'])

    count = len(records)
    hearts = _heart_stats(records)
    return {
        'format': FORMAT,
        'specialist': config.name, 'checkpoint': str(checkpoint), 'episodes_per_state': episodes_per_state,
        'seed': seed, 'deterministic': deterministic, 'episodes': count,
        'success_rate': sum(r['success'] for r in records) / count,
        'mean_health_fraction': sum(r['health'] / r['max_health'] if r['max_health'] else 0.0
                                    for r in records) / count,
        # the same at reset (hp is randomized there), so the end figure has something to compare with
        'mean_start_health_fraction': sum(r['start_health'] / r['max_health'] if r['max_health'] else 0.0
                                          for r in records) / count,
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
    parser.add_argument('--models-dir', default=None, help=f"default ${registry.MODELS_DIR_ENV} or ./models")
    parser.add_argument('--episodes-per-state', type=int, default=100)
    parser.add_argument('--parallel', type=int, default=1, help="envs in the evaluation pool")
    parser.add_argument('--threads', type=int, default=None, help="pool worker threads (default min(parallel, cores))")
    parser.add_argument('--device', default='cpu', help="torch device for the network")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true', help="argmax actions instead of sampling")
    parser.add_argument('--output', default=None, help="JSON output (default: <checkpoint dir>/eval.json)")
    args = parser.parse_args(argv)

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

    result = evaluate_specialist(config, checkpoint, args.episodes_per_state, args.parallel, args.seed,
                                 args.deterministic, threads=args.threads, device=args.device)
    output = Path(args.output) if args.output else checkpoint.parent / 'eval.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=2)
    print(format_report(result))
    print(f"(written to {output})")
    return 0


if __name__ == '__main__':
    main()
