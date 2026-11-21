"""Round 35: the learned guard (``triforce.guard``, act mode ``veto:K:P``) plays without the emulator's look-ahead.

The owner's condition: the look-ahead makes the guard's training labels and is never used when the game is played.
The Rust side is proven by ``crates/pool/tests/pool.rs`` ``a_learned_veto_plays_without_the_emulators_look_ahead``
(no ``PoolEnv::trial`` at all in a ``veto`` run; the guard handed exactly the decision's observation); these tests
check the same on the Zelda env and chains: no trial frames, plain play when the guard refuses nothing, and a
``veto`` run's replay verifying with the emulator alone."""

import shutil

import numpy as np
import pytest
import torch

from triforce import _native as native
from triforce import replay
from triforce.guard import GUARD_NAME, JOINT, make_guard, save_guard
from triforce.registry import load_model
from triforce.runner import Route, Runner

from .conftest import require_state
from .networks import random_network

BASE = 'temperature:0.75+relax:16'


def guarded(network, bias: torch.Tensor):
    """``network``'s guard with a hit logit per joint action fixed at ``bias`` (whatever the observation)."""
    guard = make_guard(network)
    with torch.no_grad():
        guard.value_head.bias.copy_(bias)
    return guard


def chain_runs(states_dir, mode, guards=None, seeds=(0, 3, 5)):
    """Chain runs from 0x5a W (ow-zora, then ow-moblin: Link is hit) by two random networks in ``mode``, each with
    its guard from ``guards`` (None: none): the run results and the plays by seed."""
    names = ['ow-moblin', 'ow-zora']
    networks = [random_network(1), random_network(6)]
    policies = [n.to_native() for n in networks]
    if guards is not None:
        policies = [p.with_guard(guards(n).to_native()) for p, n in zip(policies, networks)]
    task = native.ChainTask(names, 'd2first', seeds=list(seeds), start_from=str(require_state('0_5aw_0')),
                            max_steps=80, states_dir=str(states_dir))
    with native.EnvPool(task, num_envs=3, seed=0) as pool:
        pool.play(policies, act=mode, record='observations')
        events = []
        while not pool.wait_idle(timeout=0.05):
            events += pool.poll_events()
        events += pool.poll_events()
        plays = {p['job']: p for p in pool.poll_plays()}
    return {e['result']['seed']: e['result'] for e in events if e['type'] == 'run'}, plays


def foresight_of(run):
    """The run's foresight dicts, one per room that had one."""
    return [room['decisions']['foresight'] for room in run['rooms'] if room['decisions'].get('foresight')]


def test_veto_is_its_own_act_mode_never_combined_with_the_planner():
    assert native.act_mode('veto:16:0.50+temperature:0.5+relax:16') == 'temperature:0.5+relax:16+veto:16:0.5'
    for bad in ('temperature:0.5+relax:16+veto:16:0.5+plan:16:1',
                'temperature:0.5+relax:16+veto:16:0.5+low:2:argmax+plan:16:1',
                'temperature:0.5+relax:16+plan:16:1+low:2:argmax+veto:16:0.5', 'argmax+veto:16:1', 'argmax+veto:16'):
        with pytest.raises(ValueError):
            native.act_mode(bad)


def test_a_veto_that_refuses_nothing_plays_as_the_plain_mode(states_dir):
    """A guard that never refuses (every hit probability about 1e-9): the runs equal the plain mode's decision for
    decision and room record for room record; every decision is judged, none by a trial frame."""
    runs, plays = chain_runs(states_dir, BASE)
    vetoed, vetoed_plays = chain_runs(states_dir, f'{BASE}+veto:8:0.5',
                                      guards=lambda n: guarded(n, torch.full((JOINT,), -20.0)))
    for seed, run in runs.items():
        np.testing.assert_array_equal(vetoed_plays[seed]['actions'], plays[seed]['actions'])
        rooms = [{**room, 'decisions': {k: v for k, v in room['decisions'].items() if k != 'foresight'}}
                 for room in vetoed[seed]['rooms']]
        assert rooms == run['rooms'] and run['frames'] == vetoed[seed]['frames']
        seen = foresight_of(vetoed[seed])
        assert sum(sum(f['outcomes']) for f in seen) == len(plays[seed]['actions'])
        assert all(f['trial_frames'] == 0 and f['outcomes'][1] == 0 for f in seen)
        assert (vetoed_plays[seed]['tried']['future'] == -1).all()


def test_a_veto_run_replaces_on_the_guard_alone_and_replays_with_the_emulator(states_dir, models_dir, tmp_path):
    """``guard.pt`` beside two families' runs, a guard refusing about half the joint actions: the chain names its
    ``veto`` mode and each guard's hash, replaces decisions without a single trial frame, and every recorded run
    verifies with the emulator alone (a replay plays the recorded buttons; nothing of a guard or a look-ahead)."""
    models = tmp_path / 'models'
    shutil.copytree(models_dir, models)
    generator = torch.Generator().manual_seed(35)
    for family in ('ow-moblin', 'ow-zora'):
        run_dir = models / family / 'untrained'
        net, _ = load_model(run_dir)
        save_guard(run_dir / GUARD_NAME, guarded(net, 3 * torch.randn(JOINT, generator=generator)))
    mode = f'{BASE}+veto:8:0.5'
    runner = Runner(models_dir=models, max_steps=80, states_dir=states_dir, act=mode)
    route = Route.load('d2first', start_from=str(require_state('0_5aw_0')))
    record = replay.record_options(runner.record_meta(route))
    events = []
    with runner.pool(route, [3, 4], record=record, num_envs=2, threads=2) as chain:
        chain.play()
        while not chain.pool.wait_idle(timeout=0.05):
            events += chain.poll_events()
        events += chain.poll_events()
    results = chain.results(events)
    seen = [room.decisions['foresight'] for _, result in results for room in result.rooms
            if room.decisions.get('foresight')]
    assert seen and sum(f['outcomes'][1] for f in seen) > 0, 'the guard replaced decisions'
    assert all(f['trial_frames'] == 0 for f in seen)
    recordings = [native.Replay(e['data']) for e in events if e['type'] == 'replay']
    assert len(recordings) == 2
    for recording in recordings:
        models_meta = {m['name']: m for m in recording.meta['models']}
        assert {m['act'] for m in models_meta.values()} == {mode}
        assert 'guard' in models_meta['ow-zora'] and 'guard' in models_meta['ow-moblin']
        assert native.replay_verify(recording.data, states_dir=str(states_dir))['frames'] == recording.frames
