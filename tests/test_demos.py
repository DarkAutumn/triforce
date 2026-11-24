"""``triforce.demos``: demonstration rows from a scripted rule, and ``train --demos``'s cloning term."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from triforce import _native, demos, manifest, paths, train
from triforce.actions import MASK_SIZE, NUM_DIRECTIONS, NUM_DURATIONS, NUM_KINDS
from triforce.inputs import MissingInput, task_demos
from triforce.metrics import read_jsonl
from triforce.model import MASK_KEY
from triforce.specialist import load_specialist

from .conftest import start_task

KINDS, DIRECTIONS, DURATIONS = list(_native.ACTION_KINDS), list(_native.DIRECTIONS), list(_native.DURATIONS)


def _mask(kind: str, directions=(), durations=()) -> np.ndarray:
    """A mask offering one kind with the given directions and durations."""
    mask = np.zeros(MASK_SIZE, dtype=bool)
    k = KINDS.index(kind)
    mask[k] = True
    for d in directions:
        mask[NUM_KINDS + k * NUM_DIRECTIONS + DIRECTIONS.index(d)] = True
    for u in durations:
        mask[NUM_KINDS + NUM_KINDS * NUM_DIRECTIONS + k * NUM_DURATIONS + [int(x) for x in DURATIONS].index(u)] = True
    return mask


def test_action_text_becomes_the_policy_indices():
    assert demos.action_indices('MOVE W 8') == (KINDS.index('MOVE'), DIRECTIONS.index('W'), 1)
    assert demos.action_indices('SWORD N') == (KINDS.index('SWORD'), DIRECTIONS.index('N'), 0)
    assert demos.action_indices('WAIT 16') == (KINDS.index('WAIT'), 0, 2)


def test_legal_reads_the_kind_and_its_rows():
    mask = _mask('MOVE', ['N', 'W'], [4, 8])
    assert demos.legal(mask, demos.action_indices('MOVE W 8'))
    assert not demos.legal(mask, demos.action_indices('MOVE E 8'))
    assert not demos.legal(mask, demos.action_indices('MOVE N 32'))
    assert not demos.legal(mask, demos.action_indices('SWORD N'))


def test_recorder_keeps_only_kept_episodes_and_round_trips(tmp_path):
    recorder = demos.DemoRecorder()
    obs = {'link': np.arange(3, dtype=np.float32), MASK_KEY: _mask('MOVE', ['N'], [4])}
    recorder.add(obs, 'MOVE N 4')
    recorder.drop()
    for value in (1.0, 2.0):     # the second row written twice
        recorder.add({**obs, 'link': np.full(3, value, dtype=np.float32)}, 'MOVE N 4', repeat=int(value))
    recorder.add({**obs, 'link': np.full(3, 3.0, dtype=np.float32)}, 'MOVE N 4', any_duration=True)
    recorder.keep()
    with pytest.raises(ValueError, match='masked'):
        recorder.add(obs, 'MOVE S 4')
    assert recorder.save(tmp_path / 'd.npz') == 4
    loaded, masks, actions = demos.load_demos([tmp_path / 'd.npz', tmp_path / 'd.npz'])
    assert list(loaded) == ['link'] and loaded['link'][:, 0].tolist() == [1.0, 2.0, 2.0, 3.0] * 2
    move = KINDS.index('MOVE')
    assert masks.shape == (8, MASK_SIZE)
    assert actions.tolist() == [[move, 0, 0], [move, 0, 0], [move, 0, 0], [move, 0, demos.ANY_DURATION]] * 2


def test_train_clones_the_demonstrations(states_dir, tmp_path):
    """--demos adds the cloning term every update (and --ent-coeff sets the entropy bonus)."""
    env = _native.Env('ow-start', states_dir=str(states_dir), seed=0)
    obs, _ = env.reset(seed=0)
    recorder = demos.DemoRecorder()
    recorder.add(obs, 'MOVE N 8')
    recorder.keep()
    recorder.save(tmp_path / 'demos.npz')
    common = ['--specialist', start_task(), '--states-dir', str(states_dir), '--models-dir', str(tmp_path),
              '--parallel', '2', '--steps-per-update', '16', '--iterations', '2', '--no-curriculum', '--seed', '3',
              '--device', 'cpu', '--demos', str(tmp_path / 'demos.npz'), '--demo-coeff', '0.5',
              '--ent-coeff', '0.03']
    with pytest.raises(SystemExit, match='--best-of-n'):
        train.main(common + ['--run-id', 'both', '--best-of-n', '2'])
    run = train.main(common + ['--run-id', 'demos'])
    records = [r for r in read_jsonl(str(run / 'metrics.jsonl')) if 'losses/demo_bc_loss' in r]
    assert len(records) == 2 and all(r['charts/demo_bc_coeff'] == 0.5 for r in records)
    assert all(r['charts/ent_coeff'] == pytest.approx(0.03) for r in records)


def test_the_demos_recipe_merges_the_tasks_and_the_command_line(tmp_path):
    """``demos.recipe``: the tasks' files then ``--demos``; ``--demo-coeff`` over the tasks' weight, which must agree;
    no file at all: None, the default weight; groups only when some task names them (the others' files and
    ``--demos``' are ``UNGROUPED``; several tasks' group names carry the task's index)."""
    a, b = tmp_path / 'a.npz', tmp_path / 'b.npz'
    assert demos.recipe([], None, None) == (None, demos.DEFAULT_COEFF, None)
    assert demos.recipe([([], None, None)], ['x.npz'], 0.5) == (['x.npz'], 0.5, None)
    assert demos.recipe([([a], 0.05, None)], ['x.npz'], None) == ([str(a), 'x.npz'], 0.05, None)
    assert demos.recipe([([a], 0.05, None), ([b], 0.05, None)], None, 0.3) == ([str(a), str(b)], 0.3, None)
    with pytest.raises(SystemExit, match='--demo-coeff'):
        demos.recipe([([a], 0.05, None), ([b], 0.1, None)], None, None)
    assert demos.recipe([([a, b], 0.05, ['g', 'h'])], ['x.npz'], None) == (
        [str(a), str(b), 'x.npz'], 0.05, ['g', 'h', demos.UNGROUPED])
    assert demos.recipe([([a], 0.05, ['g']), ([b], 0.05, None)], None, None)[2] == ['0:g', demos.UNGROUPED]


def _fake_demos(path, rows: int, value: float):
    """``rows`` demonstration rows whose ``link`` observation is ``value`` (which file a drawn row came from)."""
    obs = {'link': np.full(3, value, dtype=np.float32), MASK_KEY: _mask('MOVE', ['N'], [4])}
    demos.save_demos(path, [obs] * rows, [demos.action_indices('MOVE N 4')] * rows)
    return path


def test_grouped_rows_are_drawn_by_group_and_ungrouped_as_before(tmp_path):
    """``demos.row_weights`` and ``ppo.demo_pick``: three groups of very different sizes (100 rows; 10 + 1,000 rows
    pooled; 50 rows) each get a third of a seeded draw of 6,000 rows, within 3 points, and inside the pooled group
    its files in proportion to their rows; without groups no weights, and the draw is the uniform one it always was
    (``randperm`` for a seed)."""
    import torch  # pylint: disable=import-outside-toplevel
    from triforce.ppo import demo_pick  # pylint: disable=import-outside-toplevel
    sizes = [100, 10, 1000, 50]
    files = [_fake_demos(tmp_path / f'{i}.npz', rows, float(i)) for i, rows in enumerate(sizes)]
    obs, _, _ = demos.load_demos(files)
    weights = demos.row_weights(files, ['a', 'b', 'b', 'c'])
    assert weights.shape == (sum(sizes),) and weights.sum().item() == pytest.approx(1.0)
    torch.manual_seed(7)
    pick = demo_pick(sum(sizes), 6000, weights.float(), 'cpu')
    source = obs['link'][pick, 0].long()
    share = {g: sum((source == i).sum().item() for i in ids) / 6000 for g, ids in (('a', [0]), ('b', [1, 2]),
                                                                                ('c', [3]))}
    assert all(abs(s - 1 / 3) < 0.03 for s in share.values()), share
    inside = (source == 1).sum().item() / max(1, ((source == 1) | (source == 2)).sum().item())
    assert abs(inside - 10 / 1010) < 0.01, inside
    with pytest.raises(ValueError, match='one a file'):
        demos.row_weights(files, ['a'])

    assert demos.row_weights(files, None) is None
    torch.manual_seed(7)
    uniform = demo_pick(sum(sizes), 64, None, 'cpu')
    torch.manual_seed(7)
    assert torch.equal(uniform, torch.randperm(sum(sizes))[:64])
    assert demo_pick(10, 64, None, 'cpu') is None     # fewer rows than a minibatch: every row, as before


def test_a_tasks_demo_groups_reach_the_manifest(tmp_path, monkeypatch):
    """A task's ``groups:`` comes back from ``task_demos`` and is each ``data.demos`` entry's ``group``
    (``triforce manifest diff`` names it); a task without stays without the key."""
    task = _task_with_demos(tmp_path, monkeypatch, ['a.npz', 'b.npz'])
    _fake_demos(tmp_path / 'demos-data' / 'a.npz', 4, 0.0)
    _fake_demos(tmp_path / 'demos-data' / 'b.npz', 4, 1.0)
    plain = task_demos(load_specialist(task))
    assert plain[2] is None
    text = json.loads(Path(task).read_text(encoding='utf-8'))
    text['demos']['groups'] = ['x', 'y']
    Path(task).write_text(json.dumps(text), encoding='utf-8')
    files, coeff, groups = task_demos(load_specialist(task))
    assert groups == ['x', 'y'] and coeff == 0.5
    entries = manifest.demo_entries(files, groups)
    assert [e['group'] for e in entries] == ['x', 'y'] and manifest.demo_groups({'demos': entries}) == ['x', 'y']
    assert 'group' not in manifest.demo_entries(files)[0] and manifest.demo_groups({'demos': []}) is None


def _task_with_demos(tmp_path, monkeypatch, names, coeff=0.5) -> str:
    """The training tests' task with ``demos: {files: names, coeff}``, the demos data folder a fresh one under
    ``tmp_path`` (this checkout's ``target/demos`` in the rule's place); returns the task's path."""
    folder = tmp_path / 'demos-data'
    folder.mkdir(exist_ok=True)
    monkeypatch.setitem(paths._LEGACY_DATA, 'demos', folder)  # pylint: disable=protected-access
    task = json.loads(Path(start_task()).read_text(encoding='utf-8'))
    task['demos'] = {'files': names, 'coeff': coeff}
    path = tmp_path / 'ow-start-demos.yaml'
    path.write_text(json.dumps(task), encoding='utf-8')
    return str(path)


def test_a_task_demo_file_that_is_not_there_refuses_the_run(tmp_path, monkeypatch):
    """A name in the task's ``demos:`` that the demos data folder lacks is a missing input: the refusal names the
    folder looked in and both places of the rule (the checkout's folder, the store's current snapshot)."""
    task = load_specialist(_task_with_demos(tmp_path, monkeypatch, ['gone.npz']))
    with pytest.raises(MissingInput) as error:
        task_demos(task)
    message = str(error.value)
    assert "['gone.npz']" in message and str(tmp_path / 'demos-data') in message
    assert str(paths.data_root('demos') / paths.CURRENT) in message
    assert task_demos(load_specialist(start_task())) == ([], None, None)


def test_train_clones_the_task_demonstrations_and_records_them(states_dir, tmp_path, monkeypatch):
    """A task's ``demos:`` is cloned at its weight and listed in the run's manifest (``data.demos``) with its
    checksum; ``--demo-coeff`` overrides the weight; ``--no-task-demos`` leaves it out."""
    env = _native.Env('ow-start', states_dir=str(states_dir), seed=0)
    obs, _ = env.reset(seed=0)
    recorder = demos.DemoRecorder()
    recorder.add(obs, 'MOVE N 8')
    recorder.keep()
    task = _task_with_demos(tmp_path, monkeypatch, ['d.npz'])
    recorder.save(tmp_path / 'demos-data' / 'd.npz')
    common = ['--specialist', task, '--states-dir', str(states_dir), '--models-dir', str(tmp_path / 'models'),
              '--parallel', '2', '--steps-per-update', '16', '--iterations', '1', '--no-curriculum', '--seed', '3',
              '--device', 'cpu']

    def cloned(run):
        return [r['charts/demo_bc_coeff'] for r in read_jsonl(str(run / 'metrics.jsonl'))
                if 'losses/demo_bc_loss' in r]

    run = train.main(common + ['--run-id', 'task'])
    assert cloned(run) == [0.5]
    sha = hashlib.sha256((tmp_path / 'demos-data' / 'd.npz').read_bytes()).hexdigest()
    assert [entry['sha256'] for entry in manifest.read(run)['data']['demos']] == [sha]
    assert cloned(train.main(common + ['--run-id', 'weight', '--demo-coeff', '0.25'])) == [0.25]
    plain = train.main(common + ['--run-id', 'plain', '--no-task-demos'])
    assert cloned(plain) == [] and 'demos' not in manifest.read(plain)['data']
