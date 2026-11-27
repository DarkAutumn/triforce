"""Judging a teacher in chains (triforce.judge, eval-batch --teacher): where a group plays, the comparison with a
base measurement, the files found in a folder or in the store."""

import hashlib
import json
import shlex
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


def _training(family, name, iterations=(2, 4)):
    """An open run with numbered checkpoints as the lab trainer leaves them (``registry.snapshot_checkpoint``)."""
    import os  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    from triforce import registry  # pylint: disable=import-outside-toplevel
    run = store.allocate_run(family, name)
    (run.path / 'run.json').write_text(json.dumps({'manifest': {'schema': 1, 'kind': 'lab-train'}}))
    for iteration in iterations:
        torch.save({'state_dict': {'w': torch.tensor([float(iteration)])}, 'iterations': iteration,
                    'steps': 1000 * iteration}, run.path / 'ckpt.tmp')
        os.replace(run.path / 'ckpt.tmp', run.path / 'ckpt.pt')
        registry.snapshot_checkpoint(run.path / 'ckpt.pt', iteration)
    return run


def test_eval_batch_plays_a_numbered_checkpoint_made_at_the_jobs_turn(shared, tmp_path, capsys, monkeypatch):
    store.create_team('t1', {'boss-gleeok': store.slot_of(_sealed('boss-gleeok', 'r1'))}, 'test')
    parent = _training('gen1-gleeok', 's1')
    argv = ['judged', '--teacher', 'boss-gleeok=gen1-gleeok/s1@it2', '--base', 't1', '--runs', '5', '--dungeons',
            'd4', '--chains-dir', str(tmp_path / 'chains')]
    assert eval_batch.main([*argv, '--plan']) == 0
    out = capsys.readouterr().out
    assert ('teacher boss-gleeok: gen1-gleeok/s1@it2 is gen1-gleeok/s1-it000002 (iteration 2 of gen1-gleeok/s1; '
            'sha1 ') in out and 'would be made: copied from the open run' in out
    assert 'team: t1+@boss-gleeok=gen1-gleeok/s1-it000002' in out and '2 slots changed' in out
    assert store.run_row(store.checkpoint_run_name(parent, 2)) is None            # a plan makes nothing
    args = eval_batch.build_parser().parse_args([*argv, '--scopes', 'd4', '--no-report'])
    monkeypatch.setenv('TRIFORCE_JOB_ESTIMATE', str(tmp_path / 'estimate.json'))
    eval_batch.prepare_teacher(args)                                              # the queue's dry call
    assert not args.checkpoints[0]['made'] and store.run_row(store.checkpoint_run_name(parent, 2)) is None
    monkeypatch.delenv('TRIFORCE_JOB_ESTIMATE')
    args = eval_batch.build_parser().parse_args([*argv, '--scopes', 'd4', '--no-report'])
    eval_batch.prepare_teacher(args)                                              # the job's turn
    assert args.checkpoints[0]['made'] and args.team == 't1+@boss-gleeok=gen1-gleeok/s1-it000002'
    assert [item.scope.name for item in args.judged] == ['d4']
    assert [entry.name for entry in eval_batch.build_entries(args)] == ['d4']       # one chain, no report
    assert store.effective_team(args.team)['slots']['boss-gleeok@d3first']['run'] == 'gen1-gleeok/s1-it000002'
    assert eval_batch.main([*argv[:2], 'boss-gleeok=gen1-gleeok/s1@it6', *argv[3:], '--plan']) == 2
    assert 'no numbered checkpoint of iteration 6' in capsys.readouterr().err
    assert eval_batch.main([*argv, '--scopes', 'to-d9', '--plan']) == 2
    # --with: another group's checkpoint in its slots too, not judged; a group twice is refused.
    _training('gen1-walkers', 's1')
    args = eval_batch.build_parser().parse_args([*argv, '--with', 'walkers=gen1-walkers/s1@latest'])
    eval_batch.prepare_teacher(args)
    assert args.team == 't1+@boss-gleeok=gen1-gleeok/s1-it000002+@walkers=gen1-walkers/s1-it000004'
    assert [item.dungeon for item in args.judged] == ['d4', 'd4'] and args.checkpoints[1]['made']
    team = store.effective_team(args.team)
    assert team['slots']['zol@d3first']['run'] == 'gen1-walkers/s1-it000004'
    assert eval_batch.main([*argv, '--with', 'boss-gleeok=gen1-walkers/s1@it2', '--plan']) == 2


def _played(path: Path, runs: list[dict], team: dict):
    path.write_text(''.join(json.dumps(run) + '\n' for run in runs))
    Path(f'{path}.meta.json').write_text(json.dumps({'provenance': {'route': ROUTE, 'team': team}}))


def _job_lines(out: str) -> list:
    """The plan's queue calls: lines of ``<the program beside the interpreter> job run ...``."""
    return [line for line in out.splitlines() if line.split(' ', 1)[0].endswith('triforce') and ' job run ' in line]


def test_checkpoints_are_planned_one_chain_a_job_then_collected_in_one_table(shared, tmp_path, capsys):
    store.create_team('t1', {'boss-gleeok': store.slot_of(_sealed('boss-gleeok', 'r1'))}, 'test')
    parent = _training('gen1-gleeok', 's1')
    _training('gen1-walkers', 's1')
    chains, base = tmp_path / 'chains', tmp_path / 'base'
    chains.mkdir()
    base.mkdir()
    for name in ('post-d3', 'd4'):
        _write(base / f'b-{name}.jsonl', [_won(0, False), _won(1, False), _won(2, False), _died(3)])
    common = ['checkpoints', 'boss-gleeok=gen1-gleeok/s1', '--dungeons', 'd4', '--runs', '4', '--chains-dir',
              str(chains), '--base', 'b', '--base-dir', str(base), '--agent', 'Tester', '--team-base', 't1']
    assert judge.main([*common, '--plan']) == 0
    out = capsys.readouterr().out
    assert '0 of 4 chains played; the control control-t1-r4 (t1): 0 of 2' in out
    lines = _job_lines(out)
    assert len(lines) == 6 and all('--express --timeout 10m --detach' in line for line in lines)
    assert "eval-batch control-t1-r4 --team t1 --chain 'post-d3: --runs 4 --act model --route d3first --from " \
        "0_74c_0 --stop-at step:53 --output" in lines[0]
    assert ("eval-batch gen1-gleeok-s1-r4-it000002 --teacher boss-gleeok=gen1-gleeok/s1@it2 --base t1 --scopes to-d4 "
            "--runs 4 --no-report --chains-dir") in lines[2] and '--subtask boss-gleeok-it000004-d4' in lines[5]
    # With the walkers' checkpoint beside: every chain and the control carry it.
    assert judge.main([*common, '--plan', '--with', 'walkers=gen1-walkers/s1@it2']) == 0
    out = capsys.readouterr().out
    tag = judge.control_tag('t1', ['walkers=gen1-walkers/s1-it000002'], 4)
    assert tag == f"control-t1-w{hashlib.sha256(b'walkers=gen1-walkers/s1-it000002').hexdigest()[:8]}-r4"
    assert f'the control {tag} (t1+@walkers=gen1-walkers/s1-it000002)' in out
    assert '--with: gen1-walkers/s1-it000002 (iteration 2 of gen1-walkers/s1) is made by --submit' in out
    assert '--teacher boss-gleeok=gen1-gleeok/s1@it2 --with walkers=gen1-walkers/s1@it2 --base t1' in out
    # The pieces of iteration 2 played (their jobs made its run); iteration 4's and the control not yet.
    team = {'name': 't1+@boss-gleeok=gen1-gleeok/s1-it000002',
            'overrides': {'boss-gleeok@d3first': store.checkpoint_run(parent, 2)['run']}}
    _played(chains / 'gen1-gleeok-s1-r4-it000002-post-d3.jsonl', [_won(0, True), _died(1), _died(2), _died(3)], team)
    _played(chains / 'gen1-gleeok-s1-r4-it000002-d4.jsonl', [_won(0, True), _won(1, True), _died(2), _died(3)], team)
    assert judge.main(common) == 0
    out = capsys.readouterr().out
    assert ('| 2 | 0.00M | d4 | previous triforce | 25.0% ± 21.7 of 4 | 75.0% | -50.0 ± 28.9 (paired, 4) | '
            'not played | - |') in out
    assert '| 2 | 0.00M | d4 | entrance | 50.0% ± 25.0 of 4 | 75.0% | -25.0 ± 25.0 (paired, 4) |' in out
    assert ('Not yet played (4): gen1-gleeok-s1-r4-it000004-post-d3, gen1-gleeok-s1-r4-it000004-d4, '
            'control-t1-r4-post-d3, control-t1-r4-d4') in out
    assert 'nothing filed yet' in out and not store.outputs(kind='judge')
    assert (chains / 'gen1-gleeok-s1-r4-checkpoints-vs-b.md').is_file()                # another base: named
    assert judge.main([*common, '--plan']) == 0                                    # only what is missing
    assert len(_job_lines(capsys.readouterr().out)) == 4
    # Every piece and the control in: the table is filed in the store.
    team = {'name': 't1+@boss-gleeok=gen1-gleeok/s1-it000004',
            'overrides': {'boss-gleeok@d3first': store.checkpoint_run(parent, 4)['run']}}
    for name in ('post-d3', 'd4'):
        _played(chains / f'gen1-gleeok-s1-r4-it000004-{name}.jsonl', [_won(s, True) for s in range(4)], team)
        _played(chains / f'control-t1-r4-{name}.jsonl', [_won(0, True), _died(1), _died(2), _died(3)], {'name': 't1'})
    assert judge.main(common) == 0
    out = capsys.readouterr().out
    assert ('| 4 | 0.00M | d4 | entrance | 100.0% ± 0.0 of 4 | 75.0% | +25.0 ± 25.0 (paired, 4) | 25.0% | '
            '+75.0 ± 25.0 (paired, 4) |') in out and 'Not yet played' not in out
    assert sorted(Path(row['path']).name for row in store.outputs(kind='judge')) == [
        'gen1-gleeok-s1-r4-checkpoints-vs-b.json', 'gen1-gleeok-s1-r4-checkpoints-vs-b.md']
    # The comparison of one piece names the weights it played.
    text = judge.report(judge.compare('gen1-gleeok-s1-r4-it000004', 'b', chains, base, ROUTE))
    assert '- `gen1-gleeok/s1-it000004`: iteration 4 of `gen1-gleeok/s1`, 4,000 steps' in text
    assert '| Measured (it 4, 0.00M steps) |' in text
    assert judge.main([*common[:1], 'boss-gleeok=gen1-gleeok/s1', '--its', '2,6', '--runs', '4']) == 2


# Ten other teachers beside the judged one, each a long-named run at a numbered checkpoint: the control's team spec
# alone runs past 381 characters (the lead's ten `--with`s, Nov 27).
OTHERS = ('overworld', 'walkers', 'darknut', 'goriya', 'wizzrobe', 'others', 'rooms', 'boss-manhandla',
          'boss-aquamentus', 'boss-dodongo')


def test_every_job_the_checkpoints_plan_writes_fits_the_queue_and_the_file_system(shared, tmp_path, capsys):
    store.create_team('t1', {'boss-gleeok': store.slot_of(_sealed('boss-gleeok', 'r1'))}, 'test')
    judged = 'gen1-boss-gleeok-a-teacher-with-a-long-name/s1r3-continued-from-s1r2'
    _training(*judged.split('/'))
    withs = []
    for group in OTHERS:
        family = f'gen1-{group}-a-teacher-with-a-long-name'
        _training(family, 's1r3-continued-from-s1r2')
        withs += ['--with', f'{group}={family}/s1r3-continued-from-s1r2@it2']
    plan = judge.checkpoints_plan(f'boss-gleeok={judged}', None, 100, 'd4', chains_dir=tmp_path / 'chains',
                                  base='t1', with_=withs[1::2])
    # Every chain's files (`<tag>-<scope file>.jsonl.meta.json`, `.stats.json`, `.log`) and every batch's folder in
    # the store (the tag) stay well under a file name's 255 bytes; the control's tag is short and stable.
    names = [f"{piece.tag}-{piece.judged.scope.file}.jsonl.stats.json" for piece in plan.pieces]
    names += [f"{plan.control}-{item.scope.file}.jsonl.stats.json" for item in plan.control_missing]
    assert max(len(name.encode()) for name in names) < 150, names
    resolved = [item.replace('@it2', '-it000002') for item in withs[1::2]]
    assert plan.control == judge.control_tag('t1', resolved, 100) == judge.control_tag('t1', resolved[::-1], 100)
    assert plan.control.startswith('control-t1-w') and len(plan.control) < 30 and len(plan.control_team) > 381
    assert len(judge._short_tag('x' * 300)) == judge.TAG_LIMIT  # pylint: disable=protected-access
    assert judge.main(['checkpoints', f'boss-gleeok={judged}', '--dungeons', 'd4', '--runs', '100', '--chains-dir',
                       str(tmp_path / 'chains'), '--agent', 'Tester', '--team-base', 't1', '--plan', *withs]) == 0
    lines = _job_lines(capsys.readouterr().out)
    descs = [(words := shlex.split(line))[words.index('--desc') + 1] for line in lines]
    assert len(descs) == 6 and max(len(desc) for desc in descs) <= 120, descs
    teams = [(words := shlex.split(line))[words.index('--team') + 1] for line in lines if ' --team ' in line]
    assert teams and min(len(team) for team in teams) > 381                    # the team spec rides in the command
    assert "the control of boss-gleeok's checkpoints: t1 with 10 other teachers' runs on to-d4, 100 runs" in descs
    assert f"{judged} it 4 on d4 with 10 other teachers' runs, 100 runs" in descs


def test_the_darknut_rooms_of_dungeon_3_are_counted_by_pass_cycle_and_bombs():
    def step_of(room, n=0):
        return [i for i, s in enumerate(PLAN) if (s['level'], s['room'], s['in_cave']) == (3, room, False)][n]
    first, lock, in_69, out_69 = step_of(0x5b), step_of(0x59), step_of(0x69), step_of(0x69, 1)

    def visit(step, outcome, **fields):
        return {**_visit(step, outcome), **fields}
    runs = [{'seed': 0, 'rooms': [visit(first, 'exit-correct', goal_cycle_in=0, goal_carrier_kills=1,
                                        goal_carrier_bombs=4, goal_carrier_bombs_taken=4, hits=1),
                                  visit(lock, 'exit-correct', bombs_in=8), visit(in_69, 'exit-correct', hits=2),
                                  visit(out_69, 'death', hits=3)]},
            {'seed': 1, 'rooms': [visit(first, 'death', goal_cycle_in=0, goal_carrier_kills=0, hits=3)]},
            {'seed': 2, 'rooms': [visit(first, 'exit-correct', goal_cycle_in=1, hits=0),
                                  visit(lock, 'death', bombs_in=4)]}]
    rooms = judge.darknut_rooms(runs, PLAN)
    assert rooms['5b_first'] == {'visits': 3, 'cycle0': 2, 'exits': 1, 'killed': 1, 'dropped': 4, 'taken': 4,
                                 'hits': 2.0}
    assert rooms['59_n'] == {'4': {'visits': 1, 'exits': 0}, '8': {'visits': 1, 'exits': 1}}
    assert rooms['69'] == [{'visits': 1, 'exits': 1, 'hits': 2.0}, {'visits': 1, 'exits': 0, 'hits': 3.0}]
    line = judge.darknut_rooms_lines([{'iteration': 228, 'start': 'entrance', 'rooms': rooms,
                                       'curriculum': judge.curriculum_points({'curriculum': {'iteration': 230, 'tasks': {
                                           'darknut@d3first': {'d3first_3_59n_0': {'point': 42, 'length': 140}}}}})}])[-1]
    assert line == ('| 228 | entrance | 2 (3), 1, 1, 4 / 4, 2.00 | 4: 1 / 0, 8: 1 / 1 | 1, 1, 2.00 | 1, 0, 3.00 | '
                    '42/140 (read at it 230) |')
    assert judge.darknut_rooms([{'seed': 0, 'rooms': [_visit(31)]}], PLAN) is None
