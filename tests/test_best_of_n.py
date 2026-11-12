"""``triforce.best_of_n``: ranking, per-state selection of the best runs into demo tensors (stubbed and on the native
pool), and ``train --best-of-n``'s collection schedule."""

import numpy as np
import pytest
import torch

from triforce import _native, best_of_n, train
from triforce.actions import MASK_SIZE, Action, ActionMask
from triforce.metrics import read_jsonl
from triforce.model import MASK_KEY

from .conftest import REPO_STATES_DIR, require_state
from .networks import scripted


def _record(seed, state, success, lost=0.0, frames=100, steps=2):
    return {'seed': seed, 'start_state': state, 'success': success, 'start_health': 3.0, 'health': 3.0 - lost,
            'frames': frames, 'steps': steps}


def _play(seed, steps):
    """A recorded play whose every row carries its seed (obs, mask, action)."""
    mask = np.zeros((steps, MASK_SIZE), dtype=bool)
    mask[:, 0] = True
    return {'job': seed, 'obs': {'link': np.full((steps, 2), seed, dtype=np.float32), MASK_KEY: mask},
            'actions': np.full((steps, 3), seed, dtype=np.int64)}


def test_rank_key_orders_success_then_hearts_lost_then_frames():
    records = [_record(1, 'a', False, frames=10), _record(2, 'a', True, lost=1.0, frames=10),
               _record(3, 'a', True, lost=0.5, frames=500), _record(4, 'a', True, lost=0.5, frames=200),
               _record(5, 'a', False, lost=0.0, frames=5)]
    assert [r['seed'] for r in sorted(records, key=best_of_n.rank_key)] == [4, 3, 2, 5, 1]


def test_collect_best_keeps_the_top_successes_of_each_state(monkeypatch):
    records = {r['seed']: r for r in [
        _record(10, 'a', True, lost=1.0, steps=3), _record(11, 'a', True, lost=0.0, frames=300, steps=4),
        _record(12, 'a', True, lost=0.0, frames=200, steps=5), _record(13, 'a', False, steps=6),
        _record(20, 'b', False, steps=2), _record(21, 'b', True, lost=0.5, steps=1),
        _record(22, 'b', True, lost=0.5, steps=2), _record(30, 'c', False, steps=3)]}
    plays = {seed: _play(seed, r['steps']) for seed, r in records.items()}
    calls = []

    def play_episodes(*args, **kwargs):
        calls.append((args, kwargs))
        return records, plays
    monkeypatch.setattr(best_of_n, 'play_episodes', play_episodes)
    best = best_of_n.collect_best('config', 'policy', 4, 2, seed=7, parallel=3)
    assert calls[0][1]['record'] == 'observations'
    assert (best.episodes, best.successes, best.kept, best.states) == (8, 5, 4, 2)
    # Per state (sorted), best first; ties on the rank key go to the lower seed.  Rows follow the chosen plays.
    chosen = [12, 11, 21, 22]
    rows = [seed for seed in chosen for _ in range(records[seed]['steps'])]
    assert best.steps == len(rows) == best.targets.shape[0] == best.masks.shape[0] == best.obs['link'].shape[0]
    assert best.targets[:, 0].tolist() == rows and best.obs['link'][:, 0].tolist() == rows
    assert MASK_KEY not in best.obs and best.masks.dtype == torch.bool
    assert best.batch == (best.obs, best.masks, best.targets)

    for record in records.values():
        record['success'] = False
    empty = best_of_n.collect_best('config', 'policy', 4, 2)
    assert (empty.episodes, empty.successes, empty.kept, empty.states, empty.steps) == (8, 0, 0, 0, 0)
    assert empty.obs == {} and empty.targets.shape == (0, 3)


@pytest.fixture(name='walk_north')
def fixture_walk_north():
    """ow-start's 0x1b east-entry states, where walking north succeeds in some runs and times out in others (Link
    enters at the east edge; west of his entry row is blocked in the rebuilt states, north is open in some)."""
    for chain in range(3):
        require_state(f'0_1be_{chain}')
    return _native.SpecialistConfig.load('ow-start').replace(start_states=['0_1be_*'], max_steps=30)


def test_collect_best_on_the_native_pool(walk_north):
    policy = scripted('MOVE N 16').to_native()
    # Seed 24: the 8-kind schema's noise stream (round 21) picks the blocked MOVE's direction differently than seed 11 did.
    records, _ = best_of_n.play_episodes(walk_north, policy, 4, seed=24, parallel=4, states_dir=REPO_STATES_DIR)
    by_state = {}
    for record in sorted(records.values(), key=lambda r: (best_of_n.rank_key(r), r['seed'])):
        if record['success']:
            by_state.setdefault(record['start_state'], []).append(record)
    # The precondition the selection checks below need: some runs fail, and some state has two successes (so
    # keeping one per state drops one). If rebuilt states break it, pick another scripted move or state set.
    assert not all(r['success'] for r in records.values()), "every run succeeded"
    assert any(len(runs) >= 2 for runs in by_state.values()), f"no state with two successes: {by_state.keys()}"
    best = best_of_n.collect_best(walk_north, policy, 4, 1, seed=24, parallel=4, states_dir=REPO_STATES_DIR)
    assert best.episodes == len(records) == 4 * 3
    assert best.successes == sum(r['success'] for r in records.values())
    kept = [r for runs in by_state.values() for r in runs[:1]]
    assert 0 < best.kept == len(kept) < best.successes and best.states == len(by_state)
    assert best.steps == sum(r['steps'] for r in kept)
    assert all(value.shape[0] == best.steps for value in best.obs.values())
    assert best.masks.shape == (best.steps, MASK_SIZE) and best.targets.shape == (best.steps, 3)
    assert all(ActionMask.from_flat(mask.numpy()).is_allowed(Action.from_indices(*target.tolist()))
               for mask, target in zip(best.masks, best.targets))


def test_collect_best_without_a_success_is_empty(walk_north):
    best = best_of_n.collect_best(walk_north.replace(max_steps=3), scripted('WAIT 8').to_native(), 2, 2,
                                  parallel=4, states_dir=REPO_STATES_DIR)
    assert (best.episodes, best.successes, best.kept, best.states, best.steps) == (6, 0, 0, 0, 0)
    assert best.batch[0] == {}


def test_play_episodes_ends_when_stop_says_so(walk_north):
    """``stop`` sees each record as it arrives; True ends the evaluation there (the pool closes mid-play)."""
    seen = []
    records, _ = best_of_n.play_episodes(walk_north, scripted('MOVE N 16').to_native(), 20, seed=11, parallel=4,
                                         states_dir=REPO_STATES_DIR, stop=lambda r: seen.append(r) or len(seen) == 2)
    assert len(seen) == len(records) == 2 and sorted(records) == sorted(r['seed'] for r in seen)
    by_state = best_of_n.success_by_state(walk_north, scripted('MOVE N 16').to_native(), 20, seed=11, parallel=4,
                                          states_dir=REPO_STATES_DIR, stop=lambda r: not r['success'])
    assert sum(e for _, e in by_state.values()) - sum(s for s, _ in by_state.values()) == 1


def test_train_refreshes_the_best_runs_every_best_every_iterations(states_dir, tmp_path, monkeypatch):
    collections = []

    def collect_best(_config, _policy, runs, keep, seed=0, parallel=4, states_dir=None):
        collections.append((runs, keep, seed, parallel, states_dir))
        empty = torch.zeros(0, dtype=torch.bool)
        return best_of_n.BestRuns({}, empty, torch.zeros(0, 3, dtype=torch.int64), 6, len(collections), 0, 0, 0)
    monkeypatch.setattr(train, 'collect_best', collect_best)
    run = train.main(['--specialist', 'ow-start', '--states-dir', str(states_dir), '--models-dir', str(tmp_path),
                      '--parallel', '2', '--steps-per-update', '16', '--iterations', '4', '--no-curriculum',
                      '--seed', '3', '--best-of-n', '3', '--best-keep', '1', '--best-every', '2',
                      '--best-parallel', '2', '--device', 'cpu'])
    # Collected before iteration 1 and after iteration 2; none after the last iteration (4).
    assert [(runs, keep, parallel) for runs, keep, _, parallel, _ in collections] == [(3, 1, 2), (3, 1, 2)]
    assert collections[0][2] != collections[1][2] and collections[0][4] == str(states_dir)
    records = read_jsonl(str(run / 'metrics.jsonl'))
    logged = [(r['iteration'], r['best/success_rate']) for r in records if 'best/episodes' in r]
    assert logged == [(1, 1 / 6), (3, 2 / 6)]
    assert not any('losses/demo_bc_loss' in r for r in records)     # nothing kept: no BC term
