"""Evaluate one specialist on its own rooms: N episodes from every seed start state.

    python -m triforce.evaluate --specialist NAME [--checkpoint PATH | --run-id ID] [--models-dir DIR]
        [--episodes-per-state 20] [--parallel 1] [--seed 0] [--deterministic] [--output PATH]

Training metrics mix start states and ``eval-chain`` plays the whole route; this measures one model in
isolation.  Every seed savestate of the specialist (``SpecialistConfig.resolve_start_states``) is played
``--episodes-per-state`` times in a ``ZeldaEnv`` (same rewards, end conditions and RAM randomization as
training), episode ``i`` seeded with ``seed + i``.  Actions are sampled unless ``--deterministic``.  The
checkpoint defaults to the specialist's promoted model; results go to ``<checkpoint dir>/eval.json``.
"""

import argparse
import json
import multiprocessing as mp
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import registry
from .env import ZeldaEnv
from .session import GameSession
from .specialist import SpecialistConfig, load_specialist

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
    return {'start_state': state, 'room': list(reset_info['room']), 'seed': episode_seed,
            'outcome': info['outcome'], 'ending': info['ending'], 'success': info['outcome'] == 'success',
            'steps': steps, 'frames': frames, 'health': info['health'], 'max_health': info['max_health']}


def _worker_init(config: SpecialistConfig, checkpoint: str, deterministic: bool):
    torch.set_num_threads(1)    # N processes × all-core torch threads oversubscribes the machine
    _WORKER['env'] = ZeldaEnv(config, capture_dir=None)
    _WORKER['network'] = _load_network(checkpoint)
    _WORKER['deterministic'] = deterministic


def _worker_run(job):
    state, episode_seed = job
    return _play(_WORKER['env'], _WORKER['network'], state, episode_seed, _WORKER['deterministic'])


def _group(records: list[dict], key) -> dict:
    groups = {}
    for record in records:
        groups.setdefault(key(record), []).append(record)
    return {name: {'episodes': len(items),
                   'success_rate': sum(r['success'] for r in items) / len(items),
                   'endings': dict(Counter(r['ending'] for r in items))}
            for name, items in sorted(groups.items())}


def evaluate_specialist(specialist: str | SpecialistConfig, checkpoint, episodes_per_state: int = 20,
                        parallel: int = 1, seed: int = 0, deterministic: bool = False,
                        session: Optional[GameSession] = None) -> dict:
    """Plays every seed start state `episodes_per_state` times and returns the summary dict (see module doc)."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    config = load_specialist(specialist) if isinstance(specialist, str) else specialist
    states = sorted(os.path.basename(p)[:-len('.state')] for p in config.resolve_start_states())
    if not states:
        raise ValueError(f"{config.name} has no seed start states")
    episode_states = [state for state in states for _ in range(episodes_per_state)]
    jobs = [(state, seed + index) for index, state in enumerate(episode_states)]

    if parallel <= 1:
        env = ZeldaEnv(config, session=session, capture_dir=None)
        network = _load_network(checkpoint)
        try:
            records = [_play(env, network, state, s, deterministic) for state, s in jobs]
        finally:
            env.close()
    else:
        ctx = mp.get_context('spawn')
        with ctx.Pool(parallel, _worker_init, (config, str(checkpoint), deterministic)) as pool:
            records = pool.map(_worker_run, jobs)
    records.sort(key=lambda r: r['seed'])

    count = len(records)
    return {
        'specialist': config.name, 'checkpoint': str(checkpoint), 'episodes_per_state': episodes_per_state,
        'seed': seed, 'deterministic': deterministic, 'episodes': count,
        'success_rate': sum(r['success'] for r in records) / count,
        'mean_health_fraction': sum(r['health'] / r['max_health'] if r['max_health'] else 0.0
                                    for r in records) / count,
        'mean_steps': sum(r['steps'] for r in records) / count,
        'mean_frames': sum(r['frames'] for r in records) / count,
        'outcomes': dict(Counter(r['outcome'] for r in records)),
        'endings': dict(Counter(r['ending'] for r in records)),
        'by_room': _group(records, lambda r: _room_key(r['room'])),
        'by_state': _group(records, lambda r: r['start_state']),
        'records': records,
    }


def format_report(result: dict) -> str:
    """Total line plus one line per room, worst success rate first."""
    successes = sum(r['success'] for r in result['records'])
    lines = [f"{result['specialist']}: success {successes}/{result['episodes']} = {result['success_rate']:.0%}  "
             f"health {result['mean_health_fraction']:.2f}  steps {result['mean_steps']:.0f}"]
    for room, stats in sorted(result['by_room'].items(), key=lambda item: (item[1]['success_rate'], item[0])):
        top = max(stats['endings'].items(), key=lambda item: item[1])[0]
        lines.append(f"  {room:12} {stats['success_rate']:5.0%}  {stats['episodes']:4}  {top}")
    return '\n'.join(lines)


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

    name = load_specialist(args.specialist).name
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        checkpoint = checkpoint / registry.CHECKPOINT_NAME if checkpoint.is_dir() else checkpoint
    elif args.run_id:
        checkpoint = registry.checkpoint_path(name, args.run_id, args.models_dir)
    else:
        checkpoint = registry.promoted_checkpoint(name, args.models_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    result = evaluate_specialist(name, checkpoint, args.episodes_per_state, args.parallel, args.seed,
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
