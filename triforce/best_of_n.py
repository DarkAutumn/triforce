"""Best-of-N self-distillation (better-play §3): the policy's own best runs as behavior-cloning targets.

From every seed start state of a specialist, the current policy plays ``runs`` episodes in a native ``EvalTask``
pool (each with its own seed, so RAM/load randomization and the keyed sampling noise differ per run: the outcomes
spread over the futures the policy can't see, which favours robust play over lucky play).  Runs are ranked per start
state by the true objective: success first, then hearts lost, then frames.  The top ``keep`` successful runs of each
state supply ``(observation, mask, action)`` rows, kept per state so rare hard states aren't drowned out.
``collect_best`` returns them as ``ppo.PPO(demo_batch=...)`` tensors (the filtered-BC term during PPO).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import _native
from .model import MASK_KEY


@dataclass
class BestRuns:
    """What one collection produced: the demo tensors and how the runs went."""
    obs: dict[str, torch.Tensor]
    masks: torch.Tensor
    targets: torch.Tensor
    episodes: int            # runs played
    successes: int           # of which successful
    kept: int                # runs kept (top `keep` successes per state)
    states: int              # start states with at least one kept run
    steps: int               # rows in the batch

    @property
    def batch(self):
        """(obs, masks, targets), as ``PPO.demo_batch``."""
        return self.obs, self.masks, self.targets


def rank_key(record: dict) -> tuple:
    """Sort key, best first: success, then hearts lost, then frames."""
    return (not record['success'], record['start_health'] - record['health'], record['frames'])


def play_episodes(config, policy, runs: int, seed: int = 0, parallel: int = 4, threads: Optional[int] = None,
                  states_dir=None, deterministic: bool = False,
                  record: Optional[str] = None) -> tuple[dict[int, dict], dict[int, dict]]:
    """Plays ``runs`` episodes from every seed start state of ``config`` with ``policy`` (a ``_native.Policy``) on
    an ``EvalTask`` pool of ``parallel`` envs; episode ``i`` is seeded ``seed + i``.  Returns (eval records by seed,
    recorded plays by job; ``record`` as ``EnvPool.play``'s)."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    states_dir = None if states_dir is None else str(states_dir)
    states = sorted(Path(p).stem for p in config.resolve_start_states(states_dir))
    if not states:
        raise ValueError(f"{config.name} has no seed start states")
    jobs = [(state, seed + index) for index, state in enumerate(s for s in states for _ in range(runs))]
    records: dict[int, dict] = {}
    plays: dict[int, dict] = {}
    task = _native.EvalTask(config, jobs, states_dir=states_dir)
    with _native.EnvPool(task, num_envs=max(1, min(parallel, len(jobs))), threads=threads, seed=seed) as pool:
        pool.play([policy], deterministic=deterministic, record=record)
        while True:
            idle = pool.wait_idle(timeout=0.1)
            for event in pool.poll_events():
                if event['type'] == 'eval':
                    records[int(event['record']['seed'])] = event['record']
            plays.update((int(play['job']), play) for play in pool.poll_plays())
            if idle:
                break
    return records, plays


def success_by_state(config, policy, runs: int, seed: int = 0, parallel: int = 4, threads: Optional[int] = None,
                     states_dir=None, deterministic: bool = False) -> dict[str, tuple[int, int]]:
    """``play_episodes`` without recording: start state -> (successes, episodes)."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    records, _ = play_episodes(config, policy, runs, seed, parallel, threads, states_dir, deterministic)
    result: dict[str, tuple[int, int]] = {}
    for record in records.values():
        successes, episodes = result.get(record['start_state'], (0, 0))
        result[record['start_state']] = (successes + bool(record['success']), episodes + 1)
    return result


def collect_best(config, policy, runs: int, keep: int, seed: int = 0, parallel: int = 4,
                 threads: Optional[int] = None, states_dir=None, deterministic: bool = False) -> BestRuns:
    """Plays ``runs`` episodes from every seed start state of ``config`` with ``policy`` (a ``_native.Policy``)
    and returns the top ``keep`` successful runs per state as demo tensors.  Episode ``i`` is seeded
    ``seed + i``."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    records, plays = play_episodes(config, policy, runs, seed, parallel, threads, states_dir, deterministic,
                                   record='observations')
    by_state: dict[str, list[dict]] = {}
    for record in records.values():
        by_state.setdefault(record['start_state'], []).append(record)
    chosen = []
    for state in sorted(by_state):
        best = sorted((r for r in by_state[state] if r['success']), key=lambda r: (rank_key(r), r['seed']))[:keep]
        chosen += [plays[int(r['seed'])] for r in best]
    successes = sum(r['success'] for r in records.values())
    if not chosen:
        return BestRuns({}, torch.zeros(0, dtype=torch.bool), torch.zeros(0, 3, dtype=torch.int64), len(records),
                        successes, 0, 0, 0)
    keys = [k for k in chosen[0]['obs'] if k != MASK_KEY]
    obs = {k: torch.as_tensor(np.concatenate([np.asarray(p['obs'][k]) for p in chosen])) for k in keys}
    masks = torch.as_tensor(np.concatenate([np.asarray(p['obs'][MASK_KEY]) for p in chosen])).bool()
    targets = torch.as_tensor(np.concatenate([np.asarray(p['actions']) for p in chosen])).long()
    kept_states = len({by_state_key for by_state_key, rows in by_state.items() if any(r['success'] for r in rows)})
    return BestRuns(obs, masks, targets, len(records), successes, len(chosen), kept_states, int(targets.shape[0]))
