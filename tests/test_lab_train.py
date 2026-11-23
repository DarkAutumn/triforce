"""Lab training (``triforce.lab``): several families in one run, teachers, imitation, the recorder."""

import json

import numpy as np
import torch

from triforce import nets, registry
from triforce.lab import record, train
from triforce.lab.collect import Teachers, task_yaml
from triforce.specialist import load_specialist
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


def test_a_route_run_trains_on_its_own_yaml_where_there_is_one():
    """``family@d3first`` reads ``specialists/d3first/<family>.yaml`` (the wood-sword runs' tasks): each loads, on
    the route's states; a key without one (or without a route) reads the family's yaml."""
    yamls = sorted((ROOT / 'specialists' / 'd3first').glob('*.yaml'))
    assert len(yamls) == 19
    for path in yamls:
        assert task_yaml(f'{path.stem}@d3first') == str(path)
        config = load_specialist(str(path))
        assert config.name == path.stem and list(config.state_routes) == ['d3first']
    assert task_yaml('zol') == 'zol' and task_yaml('ow-lynel@d3first') == 'ow-lynel'


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


def test_a_two_family_run_with_teachers_trains_links_and_loads_once(tmp_path, logs_dir):
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
    # its log folder: a lab-train plan with a run a family, and a stream a family the training GUI draws
    folder, = logs_dir.glob('*_lab-set')
    plan = json.loads((folder / 'training.json').read_text())
    keys = ['ow-start', 'ow-start@d3first']
    assert plan['kind'] == train.KIND == 'lab-train' and plan['status'] == 'completed' and plan['families'] == keys
    assert [(r['specialist'], r['status'], r['target_steps']) for r in plan['runs']] == [
        (key, 'completed', 128) for key in keys]
    for key in keys:
        events = [json.loads(line) for line in (folder / f'{key}.jsonl').read_text().splitlines()]
        assert [e['event'] for e in events] == ['start', 'progress', 'progress', 'train_end', 'end']
        assert events[0]['target_iterations'] == 2 and events[0]['specialist'] == key
        last = events[2]
        assert (last['iteration'], last['iterations'], last['steps'], last['target_steps']) == (2, 2, 128, 128)
        assert last['metrics']['rows_share'] == metrics[f'family/{key}/rows_share']
        assert last['metrics']['teacher_coeff'] == 0.0 and 'losses/teacher_kl' in last['metrics']
        assert events[-1]['status'] == 'completed'


def run_summary(run):
    return json.loads((run / 'train_summary.json').read_text())


def test_imitation_alone_with_the_teachers_acting_and_the_recorder(tmp_path, logs_dir):
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
    # the recorder's job: a log folder whose one line counts the decisions recorded
    folder, = logs_dir.glob('*_lab-record')
    events = [json.loads(line) for line in (folder / 'progress.jsonl').read_text().splitlines()]
    assert events[0]['event'] == 'eval_start' and (events[0]['episodes'], events[0]['unit']) == (64, 'decisions')
    assert [e['done'] for e in events if e['event'] == 'eval_progress'] == [32, 64]
    assert events[-1]['event'] == 'end' and events[-1]['status'] == 'completed'


def _cache(data: dict, cache, key: str):
    """A distillation cache (``r36_merge_distill.py cache``'s layout) of one key from a ``lab.record`` dataset."""
    cache.mkdir()
    fields = [f for f in data if f not in ('action', 'family')]
    for field in fields:
        np.save(cache / f'{field.replace("/", "_")}.npy', data[field])
    np.save(cache / 'key.npy', np.zeros(data['teacher_value'].shape[0], dtype=np.uint8))
    (cache / 'cache.json').write_text(json.dumps({'keys': [key], 'fields': fields}))


def test_teacher_weights_need_targets_and_rehearsal(tmp_path):
    """``--teacher-weight KEY=0`` takes a family's rows out of the teacher term, ``--need-target-for`` sets its need
    target, ``--rehearse`` adds the KL on stored teacher rows of keys outside the rollout."""
    _teachers(tmp_path)
    out = record.main(['--families', 'ow-start', '--rollouts', '1', '--steps-per-env', '16', '--parallel', '2',
                       '--device', 'cpu', '--models-dir', str(tmp_path), '--out', str(tmp_path / 'data'),
                       '--no-wait'])
    _, data = record.load(out)
    _cache(data, tmp_path / 'cache', 'ow-start')
    run = train.main(['--name', 'reh', '--run-id', 'r3', '--families', 'ow-start@d3first=' + YAML, '--parallel', '1',
                      '--steps-per-update', '32', '--device', 'cpu', '--models-dir', str(tmp_path), '--seed', '5',
                      '--no-wait', '--iterations', '2', '--teacher-coeff', '1.0',
                      '--teacher-weight', 'ow-start@d3first=0', '--sampling', 'need', '--need-target', '0.5',
                      '--need-target-for', 'ow-start@d3first=0.99', '--rehearse', str(tmp_path / 'cache'),
                      '--rehearse-rows', '20'])
    metrics = json.loads((run / 'metrics.jsonl').read_text().splitlines()[-1])
    metrics = metrics.get('metrics', metrics)
    assert metrics['losses/teacher_kl'] == 0.0           # its only family has no teacher term
    assert metrics['losses/rehearse_kl'] > 0.0
    assert abs(metrics['family/ow-start@d3first/need'] - 0.99 + metrics.get(
        'family/ow-start@d3first/recent_success', 0.0)) < 1e-6


def test_weighted_joint_kl():
    from triforce.ppo import joint_kl
    old = torch.log_softmax(torch.randn(4, 352), -1)
    new = torch.log_softmax(torch.randn(4, 352), -1)
    rows = torch.stack([joint_kl(old[i:i + 1], new[i:i + 1]) for i in range(4)])
    weight = torch.tensor([1.0, 0.0, 2.0, 0.5])
    assert torch.allclose(joint_kl(old, new, weight), (rows * weight).mean())
    assert torch.allclose(joint_kl(old, new), rows.mean())


def test_route_only_keeps_the_routes_states_and_tracks(tmp_path):
    """``route_only``: ``state_routes`` set, the globs the route has no state for dropped, the curriculum on the
    tracks explored from the route's states (none: off)."""
    from triforce.lab.collect import route_only
    config = route_only(load_specialist('ow-lynel'), 'd3first', states_dir=ROOT / 'triforce' / 'states',
                        tracks_dir=tmp_path)
    assert list(config.state_routes) == ['d3first'] and '0_0a*_[0-9]' not in config.start_states
    if config.curriculum.enabled:
        tracks = list((tmp_path / 'd3first' / 'ow-lynel').iterdir())
        assert tracks and all(json.loads((t / 'trajectory.json').read_text())['start_state'].startswith('d3first_')
                              for t in tracks)
