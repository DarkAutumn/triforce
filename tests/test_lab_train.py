"""Lab training (``triforce.lab``): several families in one run, teachers, imitation, the recorder."""

import json

import numpy as np
import torch

from triforce import nets, registry
from triforce.lab import record, train
from triforce.lab.collect import Teachers
from triforce.runner import Policies

from .conftest import ROOT
from .networks import native_spec, random_network

YAML = str(ROOT / 'specialists' / 'ow-start.yaml')


def _teachers(root):
    """Two promoted ow-start runs: the default and one for route d3first."""
    for run, seed in (('teach-a', 1), ('teach-b', 2)):
        registry.save_checkpoint('ow-start', run, random_network(seed), root=root)
    registry.promote('ow-start', 'teach-a', root)
    registry.promote('ow-start', 'teach-b', root, route='d3first')


def test_teachers_label_with_each_familys_network(tmp_path):
    _teachers(tmp_path)
    teachers = Teachers(['ow-start', 'ow-start@d3first'], 'cpu', tmp_path)
    assert teachers.paths['ow-start@d3first'].endswith('teach-b/ckpt.pt')
    obs = {k: torch.zeros(3, *v['shape'], dtype=torch.long if v['vocab'] else torch.float32)
           for k, v in native_spec().items() if k != 'action_mask'}
    masks = torch.ones(3, 11 * 13, dtype=torch.bool)
    joint, values = teachers.label('ow-start', obs, masks)
    assert joint.shape == (3, 352) and values.shape == (3,)
    want = random_network(1).joint_log_probs(obs, masks)
    assert torch.allclose(joint, want, atol=1e-5)


def test_a_two_family_run_with_teachers_trains_links_and_loads_once(tmp_path):
    _teachers(tmp_path)
    common = ['--families', 'ow-start', f'ow-start@d3first={YAML}', '--parallel', '2', '--steps-per-update', '64',
              '--device', 'cpu', '--models-dir', str(tmp_path), '--seed', '3', '--no-wait', '--save-every', '1']
    run = train.main(['--name', 'set', '--run-id', 'r1', '--iterations', '2', '--teacher-coeff', '1.0',
                      '--teacher-fade', '1', '--arch', 'wide', '--arch-config', '{"hidden_dim": 32, "depth": 1}',
                      *common])
    assert run == tmp_path / 'set' / 'r1'
    lines = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    assert len(lines) == 2
    metrics = lines[-1]
    metrics = metrics.get('metrics', metrics)
    assert 'losses/teacher_kl' in metrics and metrics['charts/teacher_coeff'] == 0.0     # faded at iteration 2
    assert any(k.startswith('family/ow-start/') for k in metrics)
    assert any(k.startswith('family/ow-start@d3first/') for k in metrics)
    checkpoint = registry.load_checkpoint(run)
    assert checkpoint['families'] == ['ow-start', 'ow-start@d3first'] and checkpoint['model_config']['arch'] == 'wide'
    assert (tmp_path / 'ow-start' / 'r1').is_symlink()
    registry.promote('ow-start', 'r1', tmp_path)
    registry.promote('ow-start', 'r1', tmp_path, route='d3first')
    policies = Policies(tmp_path, names=['ow-start'], route='d3first')
    assert policies.policies[0] is policies.policies[-1]
    assert run_summary(run)['status'] == 'completed'


def run_summary(run):
    return json.loads((run / 'train_summary.json').read_text())


def test_imitation_alone_with_the_teachers_acting_and_the_recorder(tmp_path):
    _teachers(tmp_path)
    run = train.main(['--name', 'imit', '--run-id', 'r2', '--families', 'ow-start', '--parallel', '2',
                      '--steps-per-update', '64', '--device', 'cpu', '--models-dir', str(tmp_path), '--seed', '4',
                      '--no-wait', '--iterations', '2', '--mode', 'imitate', '--actor', 'teacher', '--epochs', '2'])
    lines = (run / 'metrics.jsonl').read_text().splitlines()
    assert len(lines) == 2 and 'charts/teacher_agreement' in lines[-1]
    assert nets.arch_of(registry.load_checkpoint(run)['model_config']) == nets.ZELDA
    out = record.main(['--families', 'ow-start', '--rollouts', '2', '--steps-per-env', '16', '--parallel', '2',
                       '--device', 'cpu', '--models-dir', str(tmp_path), '--out', str(tmp_path / 'data'),
                       '--no-wait'])
    meta, data = record.load(out)
    assert meta['families'] == ['ow-start'] and data['teacher_joint'].shape == (64, 352)
    assert set(np.unique(data['family'])) == {0} and data['obs/grid'].shape[0] == 64
    legal = np.isfinite(data['teacher_joint'])
    assert np.allclose(np.exp(np.where(legal, data['teacher_joint'], -np.inf)).sum(-1), 1.0, atol=1e-4)
