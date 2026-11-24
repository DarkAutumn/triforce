"""A lab run's recipe (``triforce.lab.recipe``): its manifest, ``--same-recipe-as``, the refusals and ``--resume``.

Every run here is small: ow-start from the bare state ``start`` (``start_task``), one env, 32 decisions an update, on
the CPU (``--all-routes``: the task names no state_routes)."""

import json
import os
import signal

import numpy as np
import pytest
import torch

from triforce import manifest, paths, registry, store
from triforce.lab import recipe, train
from triforce.lab.collect import parse_families
from triforce.lab.groups import Group
from triforce.metrics import MetricsLogger
from triforce.specialist import load_specialist

from .conftest import start_task
from .networks import native_spec


def _common(models, *families) -> list[str]:
    """The runs' shared arguments; ``families``: KEY=YAML items beside ``start_task``'s ow-start."""
    return ['--families', f'ow-start={start_task()}', *families, '--parallel', str(1 + len(families)),
            '--steps-per-update', '32', '--device', 'cpu', '--models-dir', str(models), '--all-routes']


def _run(models, run_id, *extra, seed='1', iterations='1', name='gen'):
    return train.main(['--name', name, '--run-id', run_id, '--seed', seed, '--iterations', iterations,
                       *_common(models), *extra])


def _task(tmp_path, name, **changes):
    """``start_task``'s config with ``changes`` (``obs`` merged into its obs block), written as ``<name>.yaml``."""
    config = load_specialist(start_task())
    if 'obs' in changes:
        changes['obs'] = {**config.to_dict()['obs'], **changes['obs']}
    path = tmp_path / f'{name}.yaml'
    path.write_text(config.replace(**changes).to_json())
    return path


def test_a_lab_run_records_its_manifest_and_two_seeds_differ_only_in_seed_and_identity(tmp_path):
    first, second = _run(tmp_path, 's1', seed='1'), _run(tmp_path, 's2', seed='2')
    a, b = manifest.read(first), manifest.read(second)
    assert a['kind'] == 'lab-train' and a['seeds']['seed'] == 1
    task = a['task']['tasks']['ow-start']
    assert task['yaml_sha256'] == manifest.file_sha256(start_task()) and task['effective']['obs']['target']
    assert a['data']['tasks']['ow-start']['start_states']['count'] == 1
    assert a['networks']['student']['arch'] == 'zelda' and a['networks']['student']['parameters'] > 0
    assert a['training']['ppo']['target_steps'] == 32 and a['training']['mode'] == 'ppo'
    changed = {d['path'] for d in manifest.diff(a, b)}
    assert changed == {'arguments.run_id', 'arguments.seed', 'seeds.seed', 'seeds.pools'}, changed
    assert manifest.same_recipe(b, a) == ([], [])
    # --same-recipe-as: a third seed passes; a changed learning rate is refused unless --except names it
    _run(tmp_path, 's3', '--same-recipe-as', str(first), seed='3')
    with pytest.raises(recipe.Refused, match='learning_rate'):
        _run(tmp_path, 's4', '--same-recipe-as', str(first), '--learning-rate', '0.001', seed='4')
    _run(tmp_path, 's4', '--same-recipe-as', str(first), '--learning-rate', '0.001', '--except', 'learning_rate',
         seed='4')


def _recipe(tmp_path, families, *argv, group=None) -> dict:
    """``recipe.build`` of a run of ``families`` (KEY=YAML items) without training it."""
    args = train.parse_args(['--name', 'x', '--families', *families, '--all-routes', '--models-dir', str(tmp_path),
                             *argv])
    tasks = parse_families(args.families, len(args.families))
    network = recipe.create_network(args.arch, json.loads(args.arch_config), tasks)
    return recipe.build(args, tasks, network, group=group, envs={t.key: t.envs for t in tasks}, training={},
                        seeds={'seed': 0})


def test_manifest_diff_names_a_task_switch_a_group_member_and_an_architecture_key(tmp_path):
    off = _task(tmp_path, 'target-off', obs={'target': False})
    base = _recipe(tmp_path, [f'ow-start={start_task()}'])
    switched = _recipe(tmp_path, [f'ow-start={off}'])
    assert any(line.strip().startswith('task.tasks.ow-start.effective.obs.target: true -> false')
               for line in manifest.report(base, switched))
    wide = _recipe(tmp_path, [f'ow-start={start_task()}'], '--arch', 'hitnet', '--arch-config', '{"hidden_dim": 32}')
    wider = _recipe(tmp_path, [f'ow-start={start_task()}'], '--arch', 'hitnet', '--arch-config', '{"hidden_dim": 64}')
    assert 'networks.student.config.hidden_dim' in {d['path'] for d in manifest.diff(wide, wider)}
    assert any(line.startswith('networks (') for line in manifest.report(wide, wider))
    one = Group('pair', 'd3first', ('ow-start@d3first',))
    two = Group('pair', 'd3first', ('ow-start@d3first', 'ow-moblin@d3first'))
    a = _recipe(tmp_path, [f'ow-start@d3first={start_task()}'], group=one)
    b = _recipe(tmp_path, [f'ow-start@d3first={start_task()}', f'ow-moblin@d3first={start_task()}'], group=two)
    paths_ = {d['path'] for d in manifest.diff(a, b)}
    assert 'task.group.members' in paths_ and any(p.startswith('task.tasks.ow-moblin@d3first.') for p in paths_)


def test_the_refusals_say_what_to_change(tmp_path):
    # a task drawing every route's states
    with pytest.raises(recipe.Refused, match='no state_routes .* --state-route d3first'):
        train.main(['--name', 'r', '--families', f'ow-start={start_task()}', '--models-dir', str(tmp_path),
                    '--device', 'cpu'])
    # a task's warm_start (the lab never loads it)
    warm = _task(tmp_path, 'warm', warm_start='ow-moblin')
    with pytest.raises(recipe.Refused, match=r'warm_start.*set the task.s warm_start to null.*--allow-warm-start'):
        train.main(['--name', 'r', *_common(tmp_path, f'ow-start@d3first={warm}')])
    recipe.check_warm_start(parse_families([f'ow-start={warm}'], 1), allow=True)
    # tasks that disagree on a switch
    off = _task(tmp_path, 'target-off', obs={'target': False})
    with pytest.raises(recipe.Refused, match=r'obs\.target: True in ow-start; False in ow-start@d3first'):
        train.main(['--name', 'r', *_common(tmp_path, f'ow-start@d3first={off}')])
    # the network reads a block its tasks leave at zero, or the tasks fill one it does not read
    tokens = json.dumps({'tokens': ['type', 'features', 'rel', 'threat'], 'hidden_dim': 32})
    with pytest.raises(recipe.Refused, match='reads entity_threat, which obs.threat fills, and obs.threat is off'):
        train.main(['--name', 'r', *_common(tmp_path), '--arch', 'hitnet', '--arch-config', tokens])
    threat = _task(tmp_path, 'threat', obs={'threat': True})
    with pytest.raises(recipe.Refused, match=r"obs\.threat is on in the tasks and the network \(zelda\) reads none"):
        train.main(['--name', 'r', '--families', f'ow-start={threat}', '--all-routes', '--models-dir', str(tmp_path),
                    '--device', 'cpu'])
    both = _task(tmp_path, 'both', obs={'threat': True, 'chase': True})
    generation = recipe.create_network('hitnet', {'tokens': ['type', 'features', 'rel', 'threat', 'chase'],
                                                  'feed': True}, parse_families([f'ow-start={both}'], 1))
    assert {'entity_threat', 'chase', 'entity_chase'} <= recipe.network_reads(generation)


@pytest.mark.parametrize('flag', ['--name', '--run-id'])
def test_a_name_no_team_spec_could_hold_is_refused_at_once(flag, capsys):
    """A team spec splits on '+' (``store.parse_team_spec``): a run named with one could never be chained or judged."""
    argv = ['--name', 'zol-skg', '--families', 'zol', flag, 'zol+skg']
    with pytest.raises(SystemExit) as stopped:
        train.parse_args(argv)
    assert stopped.value.code == 2
    assert f"{flag}: '+' separates the slots of a team spec" in capsys.readouterr().err


def _demo_file(path, network, zero=(), drop=()):
    """A demonstration file holding every key ``network`` reads but ``drop``, ``zero`` all zeros, the rest ones."""
    spec = native_spec()
    rows = 4
    arrays = {}
    for key in recipe.network_reads(network) - set(drop):
        shape = (rows, *spec[key]['shape'])
        arrays[f'obs/{key}'] = (np.zeros if key in zero else np.ones)(shape, dtype=np.int64 if spec[key]['vocab']
                                                                       else np.float32)
    arrays['mask'] = np.ones((rows, 143), dtype=bool)
    arrays['actions'] = np.zeros((rows, 3), dtype=np.int64)
    np.savez(path, **arrays)
    return str(path)


def test_demonstrations_recorded_without_a_key_the_network_reads_are_refused(tmp_path):
    both = _task(tmp_path, 'both', obs={'threat': True, 'chase': True})
    families = parse_families([f'ow-start={both}'], 1)
    network = recipe.create_network('hitnet', {'tokens': ['type', 'features', 'rel', 'threat', 'chase']}, families)
    good = _demo_file(tmp_path / 'good.npz', network)
    recipe.check_demos(network, families, [good])
    old = _demo_file(tmp_path / 'old.npz', network, drop=('chase', 'entity_chase'))
    with pytest.raises(recipe.Refused, match=r'old\.npz: no chase, entity_chase'):
        recipe.check_demos(network, families, [good, old])
    off = _demo_file(tmp_path / 'off.npz', network, zero=('entity_threat',))
    with pytest.raises(recipe.Refused, match=r'off\.npz: entity_threat all zeros \(recorded with obs\.threat off\)'):
        recipe.check_demos(network, families, [off])


def _stop_at(monkeypatch, iteration: int):
    """SIGTERM this process right after iteration ``iteration``'s metrics are written: the queue's stop, mid-run."""
    original = MetricsLogger.log

    def log(self, it, steps, metrics):
        original(self, it, steps, metrics)
        if it == iteration:
            os.kill(os.getpid(), signal.SIGTERM)
    monkeypatch.setattr(MetricsLogger, 'log', log)


def test_a_stopped_run_resumes_to_the_same_iterations_and_steps_as_one_straight_through(tmp_path, monkeypatch):
    """Four iterations straight against two, a SIGTERM, then ``--resume`` for the rest: the same iteration and step
    counts (the weights are not compared: the resumed pools draw fresh episodes, ``RESUME_SEED_STRIDE``)."""
    extra = ['--sampling', 'need', '--need-target', '1.0', '--save-every', '1']
    straight = _run(tmp_path, 'straight', *extra, seed='9', iterations='4')
    with monkeypatch.context() as patch:
        _stop_at(patch, 2)
        stopped = _run(tmp_path, 'r1', *extra, seed='9', iterations='4')
    assert json.loads((stopped / 'train_summary.json').read_text())['status'] == 'interrupted'
    saved = registry.load_checkpoint(stopped)
    assert saved['iterations'] == 2 and saved['lab']['sampler'] is not None
    assert (stopped / 'tasks' / 'ow-start' / 'curriculum.json').is_file()
    with pytest.raises(recipe.Refused, match='learning_rate'):
        _run(tmp_path, 'r1', *extra, '--resume', 'gen/r1', '--learning-rate', '0.001', seed='9', iterations='4')
    resumed = _run(tmp_path, 'r1', *extra, '--resume', 'gen/r1', seed='9', iterations='4')
    assert resumed == stopped
    mine, theirs = (json.loads((run / 'train_summary.json').read_text()) for run in (resumed, straight))
    assert (mine['status'], mine['iterations'], mine['steps'], mine['start_iteration']) == (
        'completed', theirs['iterations'], theirs['steps'], 2)
    assert [json.loads(line)['iteration'] for line in (resumed / 'metrics.jsonl').read_text().splitlines()] == [
        1, 2, 3, 4]
    record = json.loads((resumed / 'run.json').read_text())
    assert [(r['iteration'], r['previous']['status']) for r in record['resumes']] == [(2, 'interrupted')]
    assert record['resumes'][0]['seeds']['pools'] == 9 + train.RESUME_SEED_STRIDE * 2
    assert registry.load_checkpoint(resumed)['iterations'] == 4
    with pytest.raises(SystemExit, match='the run ended'):
        _run(tmp_path, 'r1', *extra, '--resume', 'gen/r1', seed='9', iterations='4')


@pytest.fixture
def shared(tmp_path, monkeypatch):
    """An empty store root (as ``tests/test_store.py``'s), its team variable restored after the test."""
    root = tmp_path / 'shared'
    monkeypatch.setenv(paths.SHARED_ENV, str(root))
    monkeypatch.setenv(store.TEAM_ENV, 'recorded-so-it-is-restored')
    monkeypatch.delenv(store.TEAM_ENV)
    monkeypatch.setattr(store, '_TEAMS', {})
    monkeypatch.setattr(store, '_VIEWS', {})
    store.init()
    yield root
    store.make_writable(root)


def test_an_interrupted_store_run_stays_open_and_resumes_then_seals(shared, tmp_path, monkeypatch):
    teacher = store.allocate_run('ow-start', 't')
    torch.save({'state_dict': {}, 'tag': 't'}, teacher.path / 'ckpt.pt')
    store.seal_run(teacher)
    store.create_team('t1', {'ow-start': store.slot_of(teacher)}, 'test')
    argv = ['--name', 'gen', '--run-id', 'r1', '--team', 't1', '--families', f'ow-start={start_task()}',
            '--parallel', '1', '--steps-per-update', '32', '--device', 'cpu', '--all-routes', '--no-curriculum',
            '--seed', '5', '--iterations', '3']
    with monkeypatch.context() as patch:
        _stop_at(patch, 1)
        out = train.main(argv)
    run = store.Run('gen', 'r1')
    assert out == run.path and store.get_run(run)['status'] == store.OPEN
    assert registry.load_checkpoint(out)['iterations'] == 1
    out = train.main([*argv, '--resume', 'gen/r1'])
    assert store.get_run(run)['status'] == store.SEALED
    assert json.loads((out / 'train_summary.json').read_text())['iterations'] == 3
