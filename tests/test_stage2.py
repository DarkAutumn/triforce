"""Stage 2's tools (round 40, docs/lab.md "Stage 2: one network from its teachers"): the recorder's named teachers and
its refusal of a network reading a block whose switch is off (``scripts/experiments/r36_merge_record.py``), and the
imitation's hit loss, its refusal, its round-38 defaults (``scripts/experiments/r36_merge_distill.py``)."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from triforce import nets, registry
from triforce.lab import groups
from triforce.lab.recipe import create_network
from triforce.observation import observation_space
from triforce.specialist import load_specialist

ROOT = Path(__file__).resolve().parents[1]
GENERATION = {'tokens': ['type', 'features', 'rel', 'threat', 'chase'], 'feed': True, 'hidden_dim': 256}
SMALL = {'tokens': ['type', 'features', 'rel', 'threat', 'chase'], 'feed': True, 'hidden_dim': 32, 'depth': 1,
         'd_model': 16, 'layers': 1, 'link_out': 16, 'hit_dim': 32, 'hit_depth': 1}


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / 'experiments' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def recorder():
    return _script('r36_merge_record')


@pytest.fixture(scope='module')
def distill():
    return _script('r36_merge_distill')


def test_a_group_and_a_family_name_their_teachers(recorder):
    walkers = list(groups.get('walkers').members)
    keys, yamls, chosen = recorder.requested_keys(['zol@d3first=x.yaml', 'ow-start@d3first'], ['walkers'])
    assert keys == walkers + ['ow-start@d3first'] and yamls == {'zol@d3first': 'x.yaml'}
    assert [g.name for g in chosen] == ['walkers']
    named = recorder.named_teachers(keys, ['ow-start=ow-start/R1'], ['walkers=walkers/R2'])
    assert named == {**{k: 'walkers/R2' for k in walkers}, 'ow-start@d3first': 'ow-start/R1'}
    # by key or by family: the same
    assert recorder.named_teachers(['zol@d3first'], ['zol@d3first=a/b'], []) == {'zol@d3first': 'a/b'}


def test_a_missing_or_doubled_teacher_is_an_error(recorder):
    with pytest.raises(SystemExit, match='no teacher named for'):
        recorder.named_teachers(['zol@d3first', 'darknut@d3first'], ['zol=zol/R'], [])
    with pytest.raises(SystemExit, match='already has the teacher'):
        recorder.named_teachers(['zol@d3first'], ['zol=zol/R'], ['walkers=walkers/R'])
    with pytest.raises(SystemExit, match='none of the recorded keys'):
        recorder.named_teachers(['zol@d3first'], ['darknut=darknut/R'], [])
    with pytest.raises(SystemExit, match='no member of others'):
        recorder.named_teachers(['zol@d3first'], [], ['others=x/R'])
    assert recorder.named_teachers(['zol@d3first'], [], []) == {}      # none named: the team's (round 38)


def test_a_network_reading_a_block_whose_switch_is_off_is_named(recorder):
    task = load_specialist('zol')
    off = task.replace(obs={**task.to_dict()['obs'], 'threat': False, 'chase': False})
    on = task.replace(obs={**task.to_dict()['obs'], 'threat': True, 'chase': True})
    student = create_network('hitnet', GENERATION)
    assert recorder.switches_off(student, off) == ['obs.threat (entity_threat)',
                                                   'obs.chase (chase, entity_chase)']
    assert not recorder.switches_off(student, on)
    today = create_network(nets.ZELDA, {})          # today's networks read neither block: any task
    assert not recorder.switches_off(today, off) and not recorder.switches_off(today, on)


def test_the_generation_network_has_its_pinned_size():
    assert create_network('hitnet', GENERATION).parameter_count() == 1_608_836


def _recording(out: Path, key: str, shards: int, envs: int, obs_on: bool, seed: int, tried: int = 3):
    """A recording in ``r36_merge_record.py``'s layout: rows from ``observation_space`` (no entity in any slot), every
    action legal, the teacher's joint log-probs random, hit labels with ``tried`` candidates."""
    rng = np.random.default_rng(seed)
    space = observation_space()
    folder = out / key / 'resets'
    folder.mkdir(parents=True)
    rows = envs * 512
    for shard in range(shards):
        arrays = {}
        for name, box in space.spaces.items():
            if name == 'action_mask':
                continue
            if np.issubdtype(box.dtype, np.integer):
                value = rng.integers(0, 3, (rows, *box.shape)).astype(box.dtype)
            else:
                value = rng.random((rows, *box.shape)).astype(box.dtype) * 0.1
            arrays[f'obs/{name}'] = value
        arrays['obs/entity_features'][..., 0] = 0.0
        joint = torch.log_softmax(torch.from_numpy(rng.normal(size=(rows, 352)) * 2), -1).numpy().astype(np.float32)
        actions = np.stack([rng.integers(0, 11, rows), rng.integers(0, 8, rows), rng.integers(0, 4, rows)], 1)
        tried_actions = np.stack([rng.integers(0, 11, (rows, tried)), rng.integers(0, 8, (rows, tried)),
                                  rng.integers(0, 4, (rows, tried))], -1)
        arrays.update(action_mask=np.ones((rows, 143), bool), action=actions, teacher_joint=joint,
                      teacher_value=rng.normal(size=rows).astype(np.float32), family=np.zeros(rows, np.int64),
                      health_lost=(rng.random(rows) < 0.1).astype(np.float32), hits=np.zeros(rows, np.int64),
                      blocked=np.zeros(rows, bool), beam_lined=np.zeros(rows, np.int64),
                      tried_actions=tried_actions, tried_lost=(rng.random((rows, tried)) < 0.1).astype(np.int64),
                      tried_blocked=np.zeros((rows, tried), np.int64))
        np.savez_compressed(folder / f'shard-{shard:04d}.npz', **arrays)
    (folder / 'meta.json').write_text(json.dumps({'key': key, 'rows': rows * shards,
                                                  'obs': {'threat': obs_on, 'chase': obs_on}}))


def _args(cache: Path, **over) -> argparse.Namespace:
    values = dict(cache=str(cache), name='stage', run_id='s1', arch='hitnet', arch_config=json.dumps(SMALL),
                  init=None, epochs=None, batch=256, lr=None, value_coeff=0.02, aux=['hit=hit:1.0'], fraction=1.0,
                  sources=None, val_data=None, val_shards=4, seed=0, device='cpu')
    values.update(over)
    return argparse.Namespace(**values)


def test_imitation_trains_the_hit_head_in_the_same_pass(distill, tmp_path, monkeypatch):
    monkeypatch.setenv('TRIFORCE_MODELS_DIR', str(tmp_path / 'models'))
    monkeypatch.delenv('TRIFORCE_TEAM', raising=False)
    _recording(tmp_path / 'data', 'zol@d3first', 2, 2, True, 1)
    _recording(tmp_path / 'data', 'stalfos-keese-gel@d3first', 2, 2, True, 2)
    distill.build_cache([str(tmp_path / 'data')], tmp_path / 'cache')
    meta = json.loads((tmp_path / 'cache' / 'cache.json').read_text())
    assert 'tried_lost' in meta['fields'] and 'action' in meta['fields'] and 'family' not in meta['fields']
    assert [r['obs'] for r in meta['recorded']] == [{'threat': True, 'chase': True}] * 2
    held = np.load(tmp_path / 'cache' / 'held.npy')
    assert held.sum() == 4 * 512            # the last env of each of the 4 rollouts
    distill.train(_args(tmp_path / 'cache', epochs=2))
    run = tmp_path / 'models' / 'stage' / 's1'
    record = json.loads((run / 'distill.json').read_text())
    assert record['args']['lr'] == distill.FIRST['lr']
    first, last = record['history']
    assert 'train_aux_hit' in first and 'aux_hit' in last['val'] and 'aux_hit_pr_judged' in last['val']
    assert last['train_kl'] < first['train_kl']
    network, _ = registry.load_model(run)
    assert nets.arch_of(network.config.to_dict()) == 'hitnet' and network.config['feed']
    # a second pass from it takes round 38's later defaults (5 epochs at 3e-4)
    distill.train(_args(tmp_path / 'cache', run_id='s2', init=str(run), arch='zelda', arch_config='{}'))
    second = json.loads((tmp_path / 'models' / 'stage' / 's2' / 'distill.json').read_text())
    assert (second['args']['epochs'], second['args']['lr']) == (distill.LATER['epochs'], distill.LATER['lr'])
    assert second['init']['sha1'] == registry.checkpoint_sha1(run)


def test_imitation_refuses_a_student_reading_blocks_recorded_off(distill, tmp_path, monkeypatch):
    monkeypatch.setenv('TRIFORCE_MODELS_DIR', str(tmp_path / 'models'))
    _recording(tmp_path / 'data', 'zol@d3first', 1, 2, False, 1)
    distill.build_cache([str(tmp_path / 'data')], tmp_path / 'cache')
    with pytest.raises(SystemExit, match=r'data/zol@d3first/resets obs.threat'):
        distill.train(_args(tmp_path / 'cache', epochs=1))
    today = {'tokens': ['type', 'features', 'rel'], 'feed': True, 'hidden_dim': 32, 'depth': 1, 'd_model': 16,
             'layers': 1, 'link_out': 16, 'hit_dim': 32, 'hit_depth': 1}
    distill.train(_args(tmp_path / 'cache', epochs=1, arch_config=json.dumps(today)))   # reads neither: trains
