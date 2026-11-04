"""Evaluate one specialist on its own rooms: N episodes from every seed start state.

    python -m triforce.evaluate --specialist NAME [--checkpoint PATH | --run-id ID] [--models-dir DIR]
        [--episodes-per-state 20] [--parallel 1] [--seed 0] [--deterministic] [--output PATH]

Training metrics mix start states and ``eval-chain`` plays the whole route; this measures one model in
isolation.  Every seed savestate of the specialist (``SpecialistConfig.resolve_start_states``) is played
``--episodes-per-state`` times in a ``ZeldaEnv`` (same rewards, end conditions and RAM randomization as
training), episode ``i`` seeded with ``seed + i``.  Actions are sampled unless ``--deterministic``.  The
checkpoint defaults to the specialist's promoted model; results go to ``<checkpoint dir>/eval.json``.
``mean_health_change`` is hearts at the episode's end minus hearts at its start (after hp randomization).
Dropped hearts (``triforce/hearts.py``) are kept per episode (``records[i]['hearts']``); ``mean_health_change_total``
adds the hearts wasted by picking them up at (or near) full health, and ``hearts_dropped`` / ``hearts_taken`` /
``hearts_left`` are means per episode (also per room and per start state).
Lingering records retain longest quiet streak, terminal tail, visit time and boss/pickup milestones. Overall,
room and start-state aggregates report mean longest/tail steps and frames plus the worst measured streak.
"""

import argparse
import json
import multiprocessing as mp
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from . import registry
from .env import ZeldaEnv
from .hearts import heart_change_total
from .session import GameSession
from .specialist import SpecialistConfig, load_specialist
from .start_states import state_label

_WORKER = {}


def _room_key(room) -> str:
    level, location, in_cave = room
    return f"{level}:0x{location:02x}" + (":cave" if in_cave else "")


def _load_network(checkpoint):
    network, _ = registry.load_model(checkpoint)
    network.eval()
    return network


def _play(env: ZeldaEnv, network, state: str, episode_seed: int, deterministic: bool) -> dict:
    """One episode from `state`; returns its record."""
    torch.manual_seed(episode_seed)
    obs, reset_info = env.reset(seed=episode_seed, options={'start_state': state})
    terminated = truncated = False
    steps = frames = 0
    info = reset_info
    while not (terminated or truncated):
        with torch.no_grad():
            action = network.get_action(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(np.asarray(action).reshape(-1)[:3])
        steps += 1
        frames += info['frames']
    record = {'start_state': state, 'room': list(reset_info['room']), 'seed': episode_seed,
              'outcome': info['outcome'], 'ending': info['ending'], 'success': info['outcome'] == 'success',
              'steps': steps, 'frames': frames, 'start_health': reset_info['health'], 'health': info['health'],
              'max_health': info['max_health'], 'hearts': info['hearts']}
    if info.get('linger') is not None:
        record['linger'] = info['linger']
    if 'blocks' in info:
        record['blocks'] = info['blocks']
    return record


def _worker_init(config: SpecialistConfig, checkpoint: str, deterministic: bool):
    torch.set_num_threads(1)    # N processes × all-core torch threads oversubscribes the machine
    _WORKER['env'] = ZeldaEnv(config, capture_dir=None)
    _WORKER['network'] = _load_network(checkpoint)
    _WORKER['deterministic'] = deterministic


def _worker_run(job):
    state, episode_seed = job
    return _play(_WORKER['env'], _WORKER['network'], state, episode_seed, _WORKER['deterministic'])


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
                   'endings': dict(Counter(r['ending'] for r in items))}
            for name, items in sorted(groups.items())}


def evaluate_specialist(specialist: str | SpecialistConfig, checkpoint, episodes_per_state: int = 20,
                        parallel: int = 1, seed: int = 0, deterministic: bool = False,
                        session: Optional[GameSession] = None,
                        on_episode: Optional[Callable[[dict, int, int], None]] = None) -> dict:
    """Plays every seed start state `episodes_per_state` times and returns the summary dict (see module doc).
    ``on_episode(record, done, total)`` is called as each episode finishes (completion order when parallel)."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    config = load_specialist(specialist) if isinstance(specialist, str) else specialist
    states = sorted(state_label(p) for p in config.resolve_start_states())
    if not states:
        raise ValueError(f"{config.name} has no seed start states")
    episode_states = [state for state in states for _ in range(episodes_per_state)]
    jobs = [(state, seed + index) for index, state in enumerate(episode_states)]

    if parallel <= 1:
        env = ZeldaEnv(config, session=session, capture_dir=None)
        network = _load_network(checkpoint)
        try:
            records = []
            for state, s in jobs:
                records.append(_play(env, network, state, s, deterministic))
                if on_episode is not None:
                    on_episode(records[-1], len(records), len(jobs))
        finally:
            env.close()
    else:
        ctx = mp.get_context('spawn')
        with ctx.Pool(parallel, _worker_init, (config, str(checkpoint), deterministic)) as pool:
            records = []
            for record in pool.imap_unordered(_worker_run, jobs):
                records.append(record)
                if on_episode is not None:
                    on_episode(record, len(records), len(jobs))
    records.sort(key=lambda r: r['seed'])

    count = len(records)
    hearts = _heart_stats(records)
    return {
        'specialist': config.name, 'checkpoint': str(checkpoint), 'episodes_per_state': episodes_per_state,
        'seed': seed, 'deterministic': deterministic, 'episodes': count,
        'success_rate': sum(r['success'] for r in records) / count,
        'mean_health_fraction': sum(r['health'] / r['max_health'] if r['max_health'] else 0.0
                                    for r in records) / count,
        'mean_health_change': sum(r['health'] - r['start_health'] for r in records) / count,
        'mean_health_change_total': hearts['health_change_total'],
        **hearts,
        **_linger_stats(records),
        **_block_stats(records),
        'mean_steps': sum(r['steps'] for r in records) / count,
        'mean_frames': sum(r['frames'] for r in records) / count,
        'outcomes': dict(Counter(r['outcome'] for r in records)),
        'endings': dict(Counter(r['ending'] for r in records)),
        'by_room': _group(records, lambda r: _room_key(r['room'])),
        'by_state': _group(records, lambda r: r['start_state']),
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
    parser.add_argument('--episodes-per-state', type=int, default=20)
    parser.add_argument('--parallel', type=int, default=1)
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
                                 args.deterministic)
    output = Path(args.output) if args.output else checkpoint.parent / 'eval.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=2)
    print(format_report(result))
    print(f"(written to {output})")
    return 0


if __name__ == '__main__':
    main()
