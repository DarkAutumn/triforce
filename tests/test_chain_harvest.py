"""Handoff harvesting (better-play §1b): a chain eval with ``harvest`` saves each handoff's pre-entry state under
``DIR/<specialist>/<level>_<room><side>_h<seed>_<n>.state``, and a training env with ``harvested_fraction`` 1.0
resets from those files with the health the upstream specialist handed over (no ``ram_randomization``)."""

import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

from triforce import _native as native, registry
from triforce.rollout_buffer import RolloutBuffer
from triforce.runner import Route, Runner

from .conftest import require_state
from .networks import scripted

# ow-zora walks E out of 0x5a into ow-moblin's 0x5b (entered from W): a handoff.
NAME = re.compile(r'^0_5bw_h(\d+)_(\d+)\.state$')
SEEDS = [0, 1]


def harvest(states_dir, models_dir, directory, parallel=2, threads=None):
    """Runs from 0x5a W: a scripted ow-zora walks E, the untrained ow-moblin takes over in 0x5b."""
    runner = Runner(models_dir=models_dir, max_steps=40, parallel=parallel, threads=threads, states_dir=states_dir,
                    networks={'ow-zora': scripted('MOVE E 16')}, harvest=directory)
    return runner.run_many(Route.load('d2first', None, str(require_state('0_5aw_0'))), SEEDS)


def handoffs(results):
    """Expected harvested file name → the health Link carried into the room."""
    expected = {}
    for result in results:
        for room in result.rooms[1:]:
            previous = result.rooms[room.index - 1]
            if (room.specialist, previous.specialist, previous.outcome) == ('ow-moblin', 'ow-zora', 'exit-correct'):
                expected[f'0_5bw_h{result.seed}_{room.index}.state'] = room.health_in
    return expected


@pytest.fixture(scope='module')
def harvested(states_dir, models_dir, tmp_path_factory):
    """(harvest dir, run results) of one two-seed harvest."""
    directory = tmp_path_factory.mktemp('harvest')
    return directory, harvest(states_dir, models_dir, directory)


def test_eval_chain_harvests_handoffs_named_by_seed_and_entry(states_dir, models_dir, harvested, tmp_path):
    directory, results = harvested
    assert {p.parent.name for p in directory.rglob('*.state')} == {'ow-moblin'}   # the new specialist's dir only
    files = sorted(p.name for p in (directory / 'ow-moblin').iterdir())
    assert files and all(NAME.match(f) for f in files), files
    assert set(files) == set(handoffs(results))
    assert {NAME.match(f).group(1) for f in files} == {str(seed) for seed in SEEDS}

    # Keyed by run seed and room index alone: one env on one thread writes the same files, byte for byte.
    serial = tmp_path / 'serial'
    harvest(states_dir, models_dir, serial, parallel=1, threads=1)
    assert sorted(p.name for p in (serial / 'ow-moblin').iterdir()) == files
    assert all((serial / 'ow-moblin' / f).read_bytes() == (directory / 'ow-moblin' / f).read_bytes() for f in files)


def test_eval_chain_command_harvests_into_the_dir_it_names(states_dir, models_dir, tmp_path):
    """``eval-chain --harvest DIR`` writes ``DIR/<specialist>/``; with DIR a states dir's ``harvested`` it is where each
    specialist's ``harvested_dir`` reads by default.  A bare ``--harvest`` is refused with a models folder (round 40:
    it wrote into the states every training draws; a store team's chain writes it into the job's folder)."""
    states = tmp_path / 'states'
    states.mkdir()
    shutil.copy(states_dir / 'start.state', states / 'start.state')
    walker = registry.save_checkpoint('ow-zora', 'walk-east', scripted('MOVE E 16'), root=tmp_path / 'models')
    output = tmp_path / 'chain.jsonl'
    command = [sys.executable, '-m', 'triforce.cli', '--models-dir', str(models_dir), 'eval-chain',
               '--route', 'd2first', '--runs', '2', '--parallel', '2', '--from', str(require_state('0_5aw_0')),
               '--no-log', '--checkpoint', f'ow-zora={walker}', '--output', str(output)]
    env = {**os.environ, 'TRIFORCE_STATES_DIR': str(states)}
    bare = subprocess.run([*command, '--harvest'], check=False, capture_output=True, text=True, env=env)
    assert bare.returncode == 2 and '--harvest needs a folder' in bare.stderr, bare.stderr
    subprocess.run([*command, '--harvest', str(states / 'harvested')], check=True, capture_output=True, text=True,
                   env=env)
    files = sorted(p.name for p in (states / 'harvested' / 'ow-moblin').iterdir())
    assert files and all(NAME.match(f) for f in files), files
    # ow-moblin's own config (no harvested_states) reads that dir: at harvested_fraction 1.0 every reset is one.
    config = native.SpecialistConfig.load('ow-moblin').replace(harvested_fraction=1.0)
    assert config.harvested_states is None
    _, info = native.Env(config, states_dir=str(states), seed=0).reset()
    assert os.path.basename(info['start_state']) in files
    # Harvested starts are opt-in (review F069): the canonical yaml (fraction 0) never draws them, and the config
    # reports the dir and its states for training's log.
    canonical = native.SpecialistConfig.load('ow-moblin')
    assert canonical.harvested_fraction == 0.0
    assert canonical.harvested_dir(str(states)) == str(states / 'harvested' / 'ow-moblin')
    assert [os.path.basename(p) for p in canonical.resolve_harvested_states(str(states))] == files
    shutil.copy(require_state('0_5bw_0'), states / '0_5bw_0.state')     # one seed state: the pool has seeds
    env = native.Env(canonical, states_dir=str(states), seed=0)
    drawn = [os.path.basename(env.reset()[1]['start_state']) for _ in range(20)]
    assert not set(drawn) & set(files), drawn


def test_an_every_room_harvest_also_saves_entries_from_the_same_family(states_dir, models_dir, tmp_path):
    """``harvest_every_room`` (``eval-chain --harvest-every-room``): ow-leever walks E out of 0x29 into its own 0x2a
    (``d3first`` step 82), no handoff; only the every-room harvest saves it."""
    files = {}
    for every_room in (False, True):
        directory = tmp_path / str(every_room)
        runner = Runner(models_dir=models_dir, max_steps=40, parallel=2, states_dir=states_dir,
                        networks={'ow-leever': scripted('MOVE E 16')}, harvest=directory,
                        harvest_every_room=every_room)
        runner.run_many(Route.load('d3first', None, str(require_state('d3first_0_29w_0'))), SEEDS)
        files[every_room] = sorted(str(p.relative_to(directory)) for p in directory.rglob('*.state'))
    entries = {f'ow-leever/d3first_0_2aw_h{seed}_1.state' for seed in SEEDS}
    assert entries <= set(files[True]) and not entries & set(files[False]), files
    assert set(files[False]) <= set(files[True]), files      # every handoff still is one


def test_training_resets_from_harvested_states_with_their_true_health(states_dir, harvested, tmp_path):
    directory, results = harvested
    health_in = handoffs(results)
    files = {str(p) for p in (directory / 'ow-moblin').iterdir()}
    config = native.SpecialistConfig.load('ow-moblin').replace(
        harvested_states=str(directory / 'ow-moblin'), harvested_fraction=1.0, max_steps=2)
    assert config.ram_randomization.hp is not None      # a seed state would get a random health
    env = native.Env(config, states_dir=str(states_dir), seed=3)
    seen = set()
    for _ in range(2 * len(files)):
        _, info = env.reset()
        assert info['start_state'] in files and info['room'] == [0, 0x5b, False]
        assert info['health'] == health_in[os.path.basename(info['start_state'])], info['start_state']
        seen.add(info['start_state'])
        env.step('WAIT 8')
    assert len(seen) > 1

    # A training pool draws them on every reset and plays them.
    task = native.TrainTask(config, str(tmp_path / 'run'), states_dir=str(states_dir))
    with native.EnvPool(task, num_envs=2, threads=2, seed=1) as pool:
        steps = 4
        views = RolloutBuffer(steps, pool.num_envs, pool.obs_spec, 0.99, 0.95).rollout_buffers()
        views['terminated'] = np.zeros((pool.num_envs, steps), bool)
        views['truncated'] = np.zeros((pool.num_envs, steps), bool)
        assert pool.rollout([scripted('WAIT 8').to_native()], steps, views) == pool.num_envs
        episodes = [e['info'] for e in pool.poll_events() if e['type'] == 'episode']
    labels = {os.path.basename(f).removesuffix('.state') for f in files}
    assert len(episodes) == 4 and all(info['start_state'] in labels for info in episodes), episodes
