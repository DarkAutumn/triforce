"""scripts/full_runs.py: the 250-run test's ranking, kept runs, long-term record and analysis on synthetic chain
records (no emulator)."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from triforce import build

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('full_runs', ROOT / 'scripts' / 'full_runs.py')
full_runs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full_runs)

# A tiny route: step 1 takes Dungeon 3's triforce, step 3 Dungeon 4's, step 4 is the last.
STEPS = [{'level': 0, 'exit': 'n'}, {'level': 3, 'exit': 'triforce'}, {'level': 0, 'exit': 'e'},
         {'level': 4, 'exit': 'triforce'}, {'level': 9, 'exit': 'triforce'}]
ROUTE = full_runs.route_facts(STEPS)


def room(step, level, name, outcome='exit-correct', frames=100, hits=0, health_in=3.0, health_out=3.0,
         family='fam', **cheats):
    return {'step': step, 'level': level, 'room': name, 'in_cave': False, 'specialist': family, 'outcome': outcome,
            'frames': frames, 'hits': hits, 'health_in': health_in, 'health_out': health_out, 'hearts_healed': 0.0,
            **{f'cheated_{k}': v for k, v in cheats.items()}}


def run(seed, rooms, result='failure', reason=''):
    return {'seed': seed, 'result': result, 'reason': reason, 'frames': sum(r['frames'] for r in rooms),
            'steps': 10 * len(rooms), 'rooms': rooms}


def records():
    path = [room(0, 0, '0x77'), room(1, 3, '0x4d', family='boss'), room(2, 0, '0x10'), room(3, 4, '0x13')]
    return [
        run(0, path + [room(4, 9, '0x32')], 'success', 'the game ended in (level 9, room 0x32)'),
        run(1, path[:1] + [room(1, 3, '0x4d', 'death', hits=3, health_out=0.0, family='boss')],
            reason='died in (level 3, room 0x4d)'),
        run(2, path[:2] + [room(2, 0, '0x10', 'stuck')],
            reason='bounce-limit: ow left into (level 0, room 0x11), the 5th return in a row'),
        run(3, path[:2] + [room(2, 0, '0x10', 'stuck')], reason='wrong-exit-limit: ow left into x, the 20th'),
        run(4, path[:1] + [room(None, 0, '0x99', 'stuck')], reason='ow exceeded 2000 steps'),
        run(5, [room(0, 0, '0x77', 'stuck', bombs=2)], reason='run exceeded 50000 steps'),
        run(6, [room(0, 0, '0x76', 'stuck', frames=500)], reason='ow no-room-progress: 900 quiet actions'),
        run(7, path[:1] + [room(1, 3, '0x4d', 'death', health_out=0.0, family='boss', keys=1)],
            reason='died in (level 3, room 0x4d)'),
    ]


def test_endings_by_kind():
    kinds = [full_runs.ending(r) for r in records()]
    assert kinds == ['win', 'death', 'bounce-limit', 'wrong-exit-limit', 'room-step-cap', 'run-step-cap',
                     'no-room-progress', 'death']
    assert full_runs.ending({'result': 'error', 'reason': 'x'}) == 'error'
    assert full_runs.ending({'result': 'failure', 'reason': 'ow stalled: 10 actions'}) == 'stalled'
    assert full_runs.ending({'result': 'failure',
                             'reason': 'ow-ghini left the route into untagged (level 0, room 0x30)'}) == 'off-route'


def test_best_worst_and_three_stuck_runs_in_different_rooms():
    rs = records()
    scores = [full_runs.score(r, ROUTE) for r in rs]
    assert full_runs.best(scores)['seed'] == 0
    # least far (step 0): seeds 5 and 6; the most cheated first
    assert full_runs.worst(scores)['seed'] == 5
    kept = full_runs.keep(rs, scores)
    # stuck rooms: 0:0x10 twice (seeds 2, 3: the lowest kept), then by lowest seed 0:0x99 (4) and 0:0x77 (5)
    assert kept == [{'seed': 0, 'reasons': ['best']},
                    {'seed': 5, 'reasons': ['worst', 'stuck-0-0x77']},
                    {'seed': 2, 'reasons': ['stuck-0-0x10']},
                    {'seed': 4, 'reasons': ['stuck-0-0x99']}]


def test_the_record_counts_endings_rooms_and_triforces():
    record = full_runs.summarize(records(), {'provenance': {'version': '9.9.9.9', 'models': [
        {'name': 'boss', 'checkpoint': 'boss/r1/ckpt.pt', 'checkpoint_sha1': 'ab', 'hash': 'cd', 'act': 'x'}]},
        'batches': [{'first_seed': 0, 'runs': 8}]}, ROUTE, seconds=10.0, total_seconds=12.0)
    assert record['version'] == '9.9.9.9' and record['seeds'] == {'first': 0, 'runs': 8}
    assert record['team'][0]['checkpoint_sha1'] == 'ab'
    assert record['endings'] == {'win': 1, 'death': 2, 'stuck': 5, 'error': 0}
    assert record['stuck_kinds'] == {'bounce-limit': 1, 'no-room-progress': 1, 'room-step-cap': 1,
                                     'run-step-cap': 1, 'wrong-exit-limit': 1}
    assert record['triforces']['D3'] == 3 and record['triforces']['D4'] == 1 and record['triforces']['D9'] == 1
    assert record['triforce_frames_median']['D3'] == 200 and record['triforce_frames_median']['D9'] == 500
    assert record['deaths'] == [{'room': '3:0x4d', 'step': 1, 'family': 'boss', 'runs': 2}]
    assert record['decisions'] == sum(r['steps'] for r in records()) and record['decisions_per_second']
    rooms = {(row[0], row[1]): dict(zip(record['room_fields'], row)) for row in record['rooms']}
    boss = rooms[('3:0x4d', 1)]
    assert (boss['visits'], boss['runs'], boss['deaths'], boss['hits'], boss['hearts_lost']) == (5, 5, 2, 3, 6.0)
    assert rooms[('0:0x99', None)]['stucks'] == 1
    fields = record['run_fields']
    by_seed = {row[0]: dict(zip(fields, row)) for row in record['runs']}
    assert by_seed[5]['cheated_bombs'] == 2 and by_seed[7]['cheated_keys'] == 1
    assert by_seed[0]['triforces'] == '349' and by_seed[0]['furthest_step'] == len(STEPS)
    json.loads(full_runs.dump(record))                  # the committed file is plain JSON


def test_analyze_compares_records_with_standard_errors(tmp_path):
    base = full_runs.summarize(records(), {'provenance': {'version': '1'}}, ROUTE)
    other = full_runs.summarize(records()[:4], {'provenance': {'version': '2'}}, ROUTE)
    paths = []
    for name, record in (('1.json', base), ('2_one.json', {**other, 'team_name': 'one'})):
        (tmp_path / name).write_text(full_runs.dump(record))
        paths.append(tmp_path / name)
    text = full_runs.analyze(paths, tmp_path / 'summary.md')
    assert (tmp_path / 'summary.md').read_text() == text
    assert '2 one − 1 main' in text
    # D3: 3/8 vs 3/4: +37.5 points
    assert '| D3 | 37.5 ± 17.1 | 75.0 ± 21.7 | +37.5 ± 27.6 (z +1.4) |' in text
    assert '3:0x4d (1) | boss | 2/5 (40.0%; 2 died, 0 stuck)' in text


def test_freeze_models_copies_pointers_and_links_runs(tmp_path):
    source = tmp_path / 'models'
    (source / 'fam' / 'r1').mkdir(parents=True)
    (source / 'fam' / 'promoted').write_text('r1')
    (source / 'fam' / 'late.d3first').write_text('r1')
    (source / 'fam' / 'notes.json').write_text('{}')
    (source / 'results').mkdir()                         # no pointers: not a family
    frozen = full_runs.freeze_models(source, tmp_path / 'frozen')
    assert sorted(p.name for p in frozen.iterdir()) == ['fam']
    assert sorted(p.name for p in (frozen / 'fam').iterdir()) == ['late.d3first', 'promoted', 'r1']
    assert (frozen / 'fam' / 'r1').resolve() == (source / 'fam' / 'r1').resolve()
    (source / 'fam' / 'promoted').write_text('r2')       # a later promotion doesn't reach the frozen team
    assert (frozen / 'fam' / 'promoted').read_text() == 'r1'


def test_runs250_plays_from_power_on(tmp_path, monkeypatch):
    (tmp_path / 'models' / 'fam').mkdir(parents=True)
    (tmp_path / 'models' / 'fam' / 'promoted').write_text('r1')
    seen = {}

    class Done:
        returncode = 2
        stdout = 'abc123\n'                              # git rev-parse HEAD

    class Help:
        returncode = 0
        stdout = '--clips {on,off} --events EVENTS --express'   # eval-chain --help (chain_options)

    def fake_run(command, **_kw):
        if '--help' in command:
            return Help()
        if 'eval-chain' in command:
            seen['command'] = command
        return Done()
    monkeypatch.setattr(full_runs.subprocess, 'run', fake_run)
    # The module check (``triforce.build``) asks git through the same subprocess.run this test answers by hand.
    monkeypatch.setattr(build, 'problem', lambda root=None: None)
    out = tmp_path / 'out'
    assert full_runs.main(['runs250', '--models-dir', str(tmp_path / 'models'), '--out', str(out)]) == 2
    command = seen['command']
    assert command[command.index('--runs') + 1] == '250' and '--power-on' in command
    assert '--express' not in command and command[command.index('--models-dir') + 1] == str(out / 'models')
    assert command.index('--models-dir') < command.index('eval-chain')     # the triforce command's own option
    assert json.loads((out / 'timing.json').read_text())['commit'] == 'abc123'    # the code that played


def test_the_wrapper_dry_call_writes_the_chain_work_and_creates_nothing(tmp_path):
    """``triforce job run`` asks ``run`` its work first (``$TRIFORCE_JOB_ESTIMATE``, no job id yet): the route steps
    of its runs, written as ``{kind, work, name}`` before any output folder."""
    from triforce import jobs
    from triforce.runner import steps_played
    estimate, out = tmp_path / 'estimate.json', tmp_path / 'out'
    environ = {**{k: v for k, v in os.environ.items() if k != jobs.JOB_ID_ENV}, jobs.JOB_ESTIMATE_ENV: str(estimate)}
    done = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'full_runs.py'), 'run', '--runs', '3', '--out',
                           str(out)], env=environ, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    assert json.loads(estimate.read_text()) == {
        'kind': 'chain', 'work': {'chain_steps': 3 * steps_played(full_runs.ROUTE)}, 'name': 'full-runs'}
    assert not out.exists()


def test_an_old_versions_runs_read_with_its_own_route(tmp_path):
    """The back-test: the route's steps come from OUT_DIR/route.json (the playing code's), its triforce order from
    them, and room records without ``step`` (before 1.4.1.0) count it from the exits taken correctly."""
    d2first = [{'level': 0, 'exit': 'n'}, {'level': 2, 'exit': 'triforce'}, {'level': 1, 'exit': 'triforce'}]
    (tmp_path / 'route.json').write_text(json.dumps({'route': 'd2first', 'steps': d2first}))
    rooms = [{k: v for k, v in room(None, level, name, outcome).items() if k != 'step'}
             for level, name, outcome in ((0, '0x77', 'exit-correct'), (2, '0x7d', 'exit-wrong'),
                                          (2, '0x7d', 'exit-correct'), (1, '0x73', 'death'))]
    (tmp_path / 'runs.jsonl').write_text(json.dumps({'seed': 0, 'result': 'failure', 'reason': 'died in 1:0x73',
                                                     'frames': 400, 'steps': 40, 'rooms': rooms}) + '\n')
    route = full_runs.live_route(tmp_path)
    assert full_runs.order(route) == [2, 1, 9]
    record = full_runs.read_runs(tmp_path)[0]
    assert [r['step'] for r in record['rooms']] == [0, 1, 1, 2] and record['rooms'][0]['step_derived']
    assert full_runs.score(record, route)['triforces'] == [2] and full_runs.score(record, route)['furthest_step'] == 2
