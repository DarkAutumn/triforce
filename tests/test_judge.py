"""Judging a teacher in chains (triforce.judge, eval-batch --teacher): where a group plays, the comparison with a
base measurement, the files found in a folder or in the store."""

import json
from pathlib import Path

import pytest

from triforce import _native, eval_batch, judge, paths, store

ROUTE = 'd3first'
PLAN = list(_native.route_steps(ROUTE))
CUT = 53                # d3first: 0:0x45 out of Dungeon 4 (0_45c_0)
GLEEOK = 51             # 4:0x13


def _visit(step, outcome='exit-correct', lost=0.0):
    s = PLAN[step]
    return {'level': s['level'], 'room': f"0x{s['room']:02x}", 'in_cave': s['in_cave'], 'step': step,
            'outcome': outcome, 'health_in': 3.0, 'health_out': 3.0 - lost}


def _won(seed, stopped):
    """A run through Dungeon 4: stopped at the cut (``--stop-at step:53``) or playing on past it."""
    rooms = [_visit(31), _visit(GLEEOK, lost=1.0), _visit(52)]
    if stopped:
        return {'seed': seed, 'route': ROUTE, 'result': 'success', 'reason': f'reached route step {CUT} (0:0x45)',
                'rooms': rooms}
    return {'seed': seed, 'route': ROUTE, 'result': 'failure', 'reason': 'died later',
            'rooms': [*rooms, _visit(CUT), _visit(CUT + 1, 'death')]}


def _died(seed):
    return {'seed': seed, 'route': ROUTE, 'result': 'failure', 'reason': 'died in (level 4, room 0x13)',
            'rooms': [_visit(31), _visit(GLEEOK, 'death', lost=3.0)]}


def _write(path: Path, runs: list[dict], start='d3first_4_71s_0'):
    path.write_text(''.join(json.dumps(run) + '\n' for run in runs))
    Path(f'{path}.meta.json').write_text(json.dumps({'provenance': {'route': ROUTE, 'start_from': start,
                                                                    'team': {'name': path.stem}}}))


def test_a_group_is_judged_where_its_families_play_from_both_starts():
    plan = judge.teacher_plan(['boss-gleeok'], ROUTE, PLAN)
    assert [(j.dungeon, j.start_kind, j.scope.file, j.cut_step) for j in plan] == [
        ('d4', 'leg', 'post-d3', 53), ('d4', 'entrance', 'd4', 53), ('d8', 'leg', 'post-d6', 229),
        ('d8', 'entrance', 'd8', 229)]
    assert plan[0].chain_args(20, ROUTE) == ['--runs', '20', '--act', 'model', '--route', ROUTE, '--from', '0_74c_0',
                                             '--stop-at', 'step:53']
    overworld = {(j.dungeon, j.start_kind) for j in judge.teacher_plan(['ow-tektite'], ROUTE, PLAN)}
    assert ('d3', 'leg') in overworld and ('d3', 'entrance') not in overworld     # the overworld: legs only
    last = judge.teacher_plan(['ganon'], ROUTE, PLAN)
    assert [(j.dungeon, j.scope.file, j.cut_step) for j in last] == [('d9', 'post-d8', None), ('d9', 'd9', None)]
    assert judge.teacher_plan(['d1-melee'], ROUTE, PLAN) == []                    # a family the route never plays


def test_the_comparison_pairs_by_seed_and_names_the_rooms_runs_end_in(tmp_path):
    arm, base = tmp_path / 'arm', tmp_path / 'base'
    arm.mkdir()
    base.mkdir()
    _write(arm / 't-d4.jsonl', [_won(0, True), _won(1, True), _died(2), _died(3)])
    _write(base / 'b-d4.jsonl', [_won(0, False), _won(1, False), _won(2, False), _died(3), _died(4)])
    result = judge.compare('t', 'b', arm, base, ROUTE)
    row, = result['rows']
    assert (row['dungeon'], row['start'], row['scope']) == ('d4', 'entrance', 'd4')
    n = row['numbers']
    assert (n['arm']['rate'], n['base']['rate'], n['base_on_seeds']['rate']) == (0.5, 0.6, 0.75)
    assert n['rate_diff']['paired'] and n['rate_diff']['n'] == 4 and n['rate_diff']['diff'] == pytest.approx(-0.25)
    assert n['arm_ends'] == {'4:0x13': 0.5} and n['base_ends'] == {'4:0x13': 0.4}
    assert n['arm_hearts_lost']['mean'] == pytest.approx(2.0) and n['arm_goals'] is None
    assert result['teams'] == {'measured': 't-d4', 'base': 'b-d4'}
    text = judge.report(result)
    assert '| d4 | entrance | 60.0% of 5 | 75.0% | 50.0% of 4 | -25.0 ± 25.0 (paired, 4) | 4x13 50.0% (40.0%) |' \
        in text and 'no tracked goals' in text


def test_without_shared_seeds_the_difference_is_unpaired(tmp_path):
    _write(tmp_path / 't-d4.jsonl', [_won(0, True), _died(1)])
    _write(tmp_path / 'b-d4.jsonl', [_won(5, False), _won(6, False)])
    n = judge.compare('t', 'b', tmp_path, route=ROUTE)['rows'][0]['numbers']
    assert not n['rate_diff']['paired'] and n['rate_diff']['diff'] == pytest.approx(-0.5)


@pytest.fixture
def shared(tmp_path, monkeypatch):
    """An empty store (as tests/test_store.py's)."""
    root = tmp_path / 'shared'
    monkeypatch.setenv(paths.SHARED_ENV, str(root))
    monkeypatch.setenv(store.TEAM_ENV, 'restored')
    monkeypatch.delenv(store.TEAM_ENV)
    monkeypatch.setattr(store, '_TEAMS', {})
    monkeypatch.setattr(store, '_VIEWS', {})
    store.init()
    yield root
    store.make_writable(root)


def test_a_base_filed_in_the_store_is_read_from_there(shared, tmp_path):
    """The baseline's records live in a job's folder of the store (``chain/``, ``chain-meta/``), no worktree."""
    made = tmp_path / 'made'
    made.mkdir()
    _write(made / 'b-d4.jsonl', [_won(0, False), _died(1)])
    store.file_output(made / 'b-d4.jsonl', 'chain', team='b')
    store.file_output(made / 'b-d4.jsonl.meta.json', 'chain-meta', team='b')
    found = judge.find_files('b', tmp_path / 'nowhere')
    path, meta = found['d4']
    assert path.parent.name == 'chain' and meta.parent.name == 'chain-meta' and path.is_relative_to(shared)
    _write(tmp_path / 't-d4.jsonl', [_won(0, True), _won(1, True)])
    n = judge.compare('t', 'b', tmp_path, route=ROUTE)['rows'][0]['numbers']
    assert n['rate_diff']['diff'] == pytest.approx(0.5) and n['base']['n'] == 2


def _sealed(family, name):
    import torch  # pylint: disable=import-outside-toplevel
    run = store.allocate_run(family, name)
    torch.save({'state_dict': {'w': torch.zeros(1)}}, run.path / 'ckpt.pt')
    store.seal_run(run)
    return run


def test_eval_batch_plans_a_teacher_without_running(shared, tmp_path, capsys):
    store.create_team('t1', {'boss-gleeok': store.slot_of(_sealed('boss-gleeok', 'r1'))}, 'test')
    _sealed('boss-gleeok', 'r9')
    argv = ['judged', '--teacher', 'boss-gleeok=boss-gleeok/r9', '--base', 't1', '--runs', '5', '--chains-dir',
            str(tmp_path / 'chains'), '--plan']
    assert eval_batch.main(argv) == 0
    out = capsys.readouterr().out
    assert 'team: t1+@boss-gleeok=boss-gleeok/r9' in out
    assert '2 slots changed: boss-gleeok@d3first, boss-gleeok@d3first:late' in out
    assert '--from 0_74c_0 --stop-at step:53' in out and '--from 8_7es_0 --stop-at step:229' in out
    assert eval_batch.main(['judged', '--teacher', 'nope=a/b', '--plan']) == 2
    assert eval_batch.main(['judged', '--teacher', 'ganon=a/b', '--team', 't1']) == 2
    assert eval_batch.main([*argv, '--dungeons', 'd3']) == 2                  # Gleeok plays no step of Dungeon 3
    capsys.readouterr()
    assert eval_batch.main([*argv, '--dungeons', 'd8']) == 0
    out = capsys.readouterr().out
    assert 'judged-post-d6.jsonl' in out and 'judged-post-d3' not in out and 'judged-d4' not in out
