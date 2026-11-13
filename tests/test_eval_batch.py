"""``triforce eval-batch`` (``triforce.eval_batch``): the plan, order, statuses, failures, side-by-side jobs, SIGTERM
and resume on fake evaluation processes; entry building; the niceness of the entry points (``triforce.jobs``)."""

import ast
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from triforce import eval_batch, jobs
from triforce.eval_batch import Entry

ROOT = Path(__file__).resolve().parents[1]

# A fake evaluation: argv[1] is its spec (JSON), then `--events STREAM` (added by the batch).
FAKE = r'''
import json, sys, time
spec = json.loads(sys.argv[1])
events = sys.argv[sys.argv.index('--events') + 1]
def emit(kind, **fields):
    with open(events, 'a') as file:
        file.write(json.dumps({'event': kind, 'ts': time.time(), **fields}) + '\n')
def note(what):
    if spec.get('order'):
        with open(spec['order'], 'a') as file:
            file.write(f"{what} {spec['name']} {time.time()}\n")
note('start')
emit('eval_start', episodes=4)
time.sleep(spec.get('sleep', 0))
emit('eval_progress', done=4, episodes=4, success_rate=0.75)
emit('eval', success_rate=0.75, episodes=4, deterministic=False, health_change=-0.5,
     by_room_failures={'1:0x22': {'episodes': 4, 'failures': {'death': 1}}})
note('end')
print('fake evaluation done', flush=True)
sys.exit(spec.get('code', 0))
'''


def fake(name: str, kind: str = 'eval', parallel: int = 1, **spec) -> Entry:
    """An entry running FAKE; its headline comes from the fake 'eval' event."""
    return Entry(name, kind, [[sys.executable, '-c', FAKE, json.dumps({'name': name, **spec})]], parallel=parallel,
                 total=4, complete=lambda events: eval_batch.eval_headline(eval_batch._last(events, 'eval')))


def plan_of(logs: Path) -> tuple[Path, dict]:
    folder = next(p for p in logs.iterdir() if p.is_dir())
    return folder, json.loads((folder / 'training.json').read_text())


def stream(folder: Path, name: str) -> list[dict]:
    return [json.loads(line) for line in (folder / f'{name}.jsonl').read_text().splitlines() if line.strip()]


def order(path: Path) -> list[tuple[str, str, float]]:
    return [(what, name, float(ts)) for what, name, ts in (line.split() for line in path.read_text().splitlines())]


def test_batch_runs_in_order_with_one_plan_and_a_stream_each(tmp_path):
    log = tmp_path / 'order.txt'
    entries = [fake('a', order=str(log)), fake('b', order=str(log)), fake('c', order=str(log))]
    lines = []
    assert eval_batch.run_batch('my-batch', entries, logs_dir=tmp_path / 'logs', cores=4, nice=7,
                                say=lines.append) == 0
    folder, plan = plan_of(tmp_path / 'logs')
    assert folder.name.endswith('_my-batch') and plan['kind'] == 'eval-batch' and plan['name'] == 'my-batch'
    assert plan['status'] == 'completed' and plan['nice'] == 7
    assert [run['name'] for run in plan['runs']] == ['a', 'b', 'c']
    assert all(run['status'] == 'ok' and run['eval_success_rate'] == 0.75 and run['eval_episodes'] == 4
               and run['started'] and run['finished'] and run['wall_seconds'] is not None for run in plan['runs'])
    assert [(what, name) for what, name, _ in order(log)] == [
        ('start', 'a'), ('end', 'a'), ('start', 'b'), ('end', 'b'), ('start', 'c'), ('end', 'c')]
    for name in 'abc':
        events = stream(folder, name)
        assert [e['event'] for e in events] == ['eval_start', 'eval_progress', 'eval', 'end']
        assert events[-1]['status'] == 'ok' and events[-1]['eval_success_rate'] == 0.75
        assert 'fake evaluation done' in (folder / f'{name}.log').read_text()
    kinds = [e['event'] for e in stream(folder, 'round')]
    assert kinds[0] == 'start' and kinds[-1] == 'end' and kinds.count('entry_start') == kinds.count('entry_end') == 3
    assert any('a: ok' in line and 'success 75.0% of 4 episodes' in line for line in lines)


def test_a_failure_in_the_middle_is_marked_and_the_batch_goes_on(tmp_path):
    entries = [fake('a'), fake('b', code=3), fake('c')]
    assert eval_batch.run_batch('b', entries, logs_dir=tmp_path, say=lambda _: None) == 1
    folder, plan = plan_of(tmp_path)
    assert [run['status'] for run in plan['runs']] == ['ok', 'failed', 'ok']
    assert plan['status'] == 'failed' and 'b' in plan['error'] and 'exit code 3' in plan['runs'][1]['error']
    events = stream(folder, 'b')
    assert [e['event'] for e in events][-2:] == ['error', 'end'] and events[-1]['status'] == 'failed'
    assert 'fake evaluation done' in events[-2]['log_tail']


def test_an_entry_without_its_result_event_fails(tmp_path):
    def complete(events):
        if eval_batch._last(events, 'eval') is None:
            raise RuntimeError("no 'eval' event")
        return {}
    entry = Entry('silent', 'eval', [[sys.executable, '-c', 'import sys']], complete=complete)
    assert eval_batch.run_batch('x', [entry], logs_dir=tmp_path, say=lambda _: None) == 1
    _, plan = plan_of(tmp_path)
    assert plan['runs'][0]['status'] == 'failed' and "no 'eval' event" in plan['runs'][0]['error']


def test_family_evaluations_share_the_cores_and_chains_run_alone(tmp_path):
    log = tmp_path / 'order.txt'
    entries = [fake('e1', parallel=2, sleep=1.0, order=str(log)), fake('e2', parallel=2, sleep=1.0, order=str(log)),
               fake('e3', parallel=2, sleep=0.2, order=str(log)), fake('ch', kind='chain', order=str(log))]
    assert eval_batch.run_batch('side', entries, logs_dir=tmp_path / 'logs', cores=4, jobs=3,
                                say=lambda _: None) == 0
    times = {(what, name): ts for what, name, ts in order(log)}
    assert times[('start', 'e2')] < times[('end', 'e1')]          # e1 and e2 side by side (4 cores)
    assert times[('start', 'e3')] > min(times[('end', 'e1')], times[('end', 'e2')])   # a third doesn't fit
    assert times[('start', 'ch')] > max(times[('end', name)] for name in ('e1', 'e2', 'e3'))   # the chain alone


def test_fits_packs_only_family_evaluations():
    ev = Entry('e', 'eval', [], parallel=4)
    chain = Entry('c', 'chain', [], parallel=20)
    assert eval_batch._fits(chain, [], 20, 1)
    assert not eval_batch._fits(ev, [chain], 20, 4)
    assert not eval_batch._fits(chain, [ev], 20, 4)
    assert eval_batch._fits(ev, [ev], 20, 2) and not eval_batch._fits(ev, [ev], 20, 1)
    assert not eval_batch._fits(ev, [ev] * 4, 16, 8)            # 16 + 4 envs > 16 cores


DRIVER = r'''
import json, sys
sys.path.insert(0, sys.argv[2])
from test_eval_batch import fake
from triforce import eval_batch
sys.exit(eval_batch.run_batch('stopped', [fake('a'), fake('b', sleep=60, order=sys.argv[3]), fake('c')],
                              logs_dir=sys.argv[1], say=lambda _: None))
'''


def test_sigterm_stops_the_running_evaluation_and_marks_the_rest_interrupted(tmp_path):
    logs, order_file = tmp_path / 'logs', tmp_path / 'order.txt'
    process = subprocess.Popen([sys.executable, '-c', DRIVER, str(logs), str(Path(__file__).parent),  # noqa: S603
                                str(order_file)])
    deadline = time.time() + 60
    while time.time() < deadline:
        if order_file.exists() and 'start b' in order_file.read_text():
            break
        time.sleep(0.1)
    else:
        process.kill()
        pytest.fail("the second evaluation never started")
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=60) == eval_batch.EXIT_INTERRUPTED
    folder, plan = plan_of(logs)
    assert plan['status'] == 'interrupted'
    assert [run['status'] for run in plan['runs']] == ['ok', 'interrupted', 'interrupted']
    for name in 'bc':
        assert stream(folder, name)[-1]['event'] == 'end' and stream(folder, name)[-1]['status'] == 'interrupted'
    assert 'end b' not in order_file.read_text()                # the fake evaluation was stopped, not waited for
    assert stream(folder, 'round')[-1] == {**stream(folder, 'round')[-1], 'event': 'end', 'status': 'interrupted'}


def _chain_output(path: Path, results: list[str]):
    from triforce.runner import RunResult, write_jsonl
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, [RunResult('d2first', seed, result) for seed, result in enumerate(results)])


def test_a_chain_whose_output_exists_is_done_and_the_rest_runs(tmp_path):
    options = eval_batch.Options('again', cores=4, eval_parallel=2)
    out = tmp_path / 'chains'
    _chain_output(out / 't-d4.jsonl', ['success', 'failure', 'success'])
    done = eval_batch.chain_entry('d4', ['--from', '4_71s_0', '--runs', '3'], options, output=out / 't-d4.jsonl')
    assert done.resume() is not None and done.resume()[0]['chain_successes'] == 2
    entries = [done, fake('family')]
    assert eval_batch.run_batch('again', entries, logs_dir=tmp_path / 'logs', say=lambda _: None) == 0
    folder, plan = plan_of(tmp_path / 'logs')
    assert plan['runs'][0]['status'] == 'ok' and plan['runs'][0]['resumed'] is True
    assert plan['runs'][0]['chain_runs'] == 3 and plan['runs'][0]['chain_success_rate'] == pytest.approx(2 / 3)
    assert [e['event'] for e in stream(folder, 'd4')] == ['chain', 'end']
    assert not (folder / 'd4.log').exists()                       # it never ran
    assert plan['runs'][1]['status'] == 'ok'
    assert eval_batch.chain_entry('d4', ['--from', '4_71s_0'], eval_batch.Options('again', 4, 2, redo=True),
                                  output=out / 't-d4.jsonl').resume() is None


def test_a_finished_chain_moves_its_output_into_place_and_counts_arrivals(tmp_path):
    options = eval_batch.Options('b', cores=4, eval_parallel=2)
    out = tmp_path / 'seg.jsonl'
    entry = eval_batch.chain_entry('d3', ['--from', '3_7cs_0', '--runs', '2'], options, output=out)
    assert entry.fields['target'] == '0:0x74'                     # the next segment's start room
    assert entry.commands[0][-2:] == ['--output', f'{out}.tmp'] and '--parallel' in entry.commands[0]
    tmp = Path(f'{out}.tmp')
    room = {'level': 0, 'room': '0x74', 'in_cave': False}
    tmp.write_text(json.dumps({'route': 'd2first', 'seed': 0, 'result': 'failure', 'rooms': [room, room]}) + '\n'
                   + json.dumps({'route': 'd2first', 'seed': 1, 'result': 'failure', 'rooms': [room]}) + '\n')
    Path(f'{tmp}.meta.json').write_text('{}')
    fields = entry.complete([{'event': 'chain', 'runs': 2, 'successes': 0, 'success_rate': 0.0}])
    assert out.is_file() and Path(f'{out}.meta.json').is_file() and not tmp.exists()
    assert fields == {'chain_runs': 2, 'chain_successes': 0, 'chain_success_rate': 0.0, 'chain_target': '0:0x74',
                      'chain_arrived': 1, 'chain_arrival_rate': 0.5}
    with pytest.raises(RuntimeError, match='summary'):
        entry.complete([])


def test_chain_entries_check_their_arguments(tmp_path):
    options = eval_batch.Options('my batch', cores=20, eval_parallel=4)
    entry = eval_batch.chain_entry(None, ['--from', '0_74c_0', '--runs', '5', '--parallel', '8',
                                          '--output=x.jsonl'], options)
    assert entry.name == 'chain-from-0_74c_0' and entry.parallel == 8 and entry.total == 5
    assert entry.output == Path('x.jsonl') and '--output=x.jsonl' not in entry.commands[0]
    assert entry.fields['target'] == '4:0x71'
    plain = eval_batch.chain_entry(None, ['--stop-at', '1:0x73'], options)
    assert plain.output == Path('runs') / 'my-batch' / 'chain-to-1:0x73.jsonl'.replace(':', '-')
    assert plain.fields['target'] is None and plain.parallel == 20
    for reserved in ('--append', '--no-log', '--events=x', '--logs-dir'):
        with pytest.raises(eval_batch.BatchError, match="can't be used in a batch"):
            eval_batch.chain_entry('x', [reserved] + (['y'] if reserved == '--logs-dir' else []), options)


def test_split_entry_and_option_helpers():
    assert eval_batch.split_entry('d3: --from 3_7cs_0 --runs 100') == ('d3', ['--from', '3_7cs_0', '--runs', '100'])
    assert eval_batch.split_entry('--from 3_7cs_0') == (None, ['--from', '3_7cs_0'])
    assert eval_batch._drop_option(['--output', 'a', '--runs', '2', '--output=b'], '--output') == ['--runs', '2']
    assert eval_batch._has_option(['--parallel=3'], '--parallel') and not eval_batch._has_option(['--par'], '--parallel')


def test_chains_tag_is_the_eight_segments_then_the_report(tmp_path):
    options = eval_batch.Options('r71', cores=20, eval_parallel=4)
    entries = eval_batch.chains_entries('r71', options, act='sample', runs=120, segments=eval_batch.SEGMENTS,
                                        chains_dir=tmp_path, chain_args=['--first-seed', '0'], report=True,
                                        only_report=False)
    assert [e.name for e in entries] == ['full', 'd2', 'post-d2', 'd1', 'post-d1', 'd3', 'post-d3', 'd4', 'report']
    full, d2 = entries[0], entries[1]
    assert full.output == tmp_path / 'r71-full.jsonl' and full.log == tmp_path / 'r71-full.log'
    assert '--from' not in full.commands[0] and d2.commands[0][d2.commands[0].index('--from') + 1] == '2_7ds_0'
    assert d2.fields['target'] == '0:0x3c' and full.fields['target'] is None and entries[7].fields['target'] is None
    assert all('--first-seed' in e.commands[0] and 'sample' in e.commands[0] for e in entries[:8])
    only = eval_batch.chains_entries('r71', options, act='sample', runs=1, segments='d2:2_7ds_0',
                                     chains_dir=tmp_path, chain_args=[], report=False, only_report=True)
    assert [e.name for e in only] == ['report']
    _chain_output(tmp_path / 'r71-full.jsonl', ['success'])
    yardstick, room_stats = eval_batch.report_commands('r71', ['full', 'd2'], tmp_path, None)
    chains = [yardstick[i + 1] for i, arg in enumerate(yardstick) if arg == '--chain']
    assert chains[0] == f"full={tmp_path / 'r71-full.jsonl'}"          # d2 has no output yet: left out
    assert chains[-1].startswith('full→d4=') and chains[-1].endswith('@2:0x7d,0:0x3c,1:0x73,0:0x37,3:0x7c,0:0x74,4:0x71')
    assert room_stats[-4:] == ['room-stats', 'r71', '--chains-dir', str(tmp_path)]


def test_same_measurement_needs_every_field():
    expected = {'specialist': 'goriya', 'act': 'sample', 'episodes_per_state': 100, 'seed': 0,
                'checkpoint_sha1': 'abc', 'code': 'v1', 'data': 'd1'}
    result = {**expected, 'act': 'sample', 'deterministic': False}
    assert eval_batch.same_measurement(result, expected)
    for key, other in (('checkpoint_sha1', 'def'), ('code', 'v2'), ('seed', 1), ('episodes_per_state', 50)):
        assert not eval_batch.same_measurement({**result, key: other}, expected)
    assert not eval_batch.same_measurement({**result, 'act': 'argmax', 'deterministic': True}, expected)


def test_duplicate_entry_names_are_refused(tmp_path):
    with pytest.raises(eval_batch.BatchError, match='twice'):
        eval_batch.run_batch('dup', [fake('a'), fake('a')], logs_dir=tmp_path, say=lambda _: None)


def test_arrival_and_headline_text():
    room = lambda level, number, cave=False: {'level': level, 'room': number, 'in_cave': cave}  # noqa: E731
    run = {'result': 'failure', 'rooms': [room(0, '0x3c'), room(0, '0x3c', True), room(2, '0x7d'), room(0, '0x3c')]}
    assert eval_batch.arrival(run, '0:0x3c') is run['rooms'][3]
    assert eval_batch.arrival(run, '2:0x7d,0:0x3c') is run['rooms'][3]
    with pytest.raises(ValueError, match='not L:0xRR'):
        eval_batch.parse_target('0:3c')
    text = eval_batch.headline_text({'chain_runs': 1000, 'chain_successes': 220, 'chain_success_rate': 0.22,
                                     'chain_target': '0:0x74', 'chain_arrived': 388, 'chain_arrival_rate': 0.388,
                                     'wall_seconds': 259})
    assert text == 'arrivals 388/1000 = 38.8% at 0:0x74 · success 220/1000 = 22.0% · 4m 19s'


# ------------------------------------------------------------------------------------------------------ priority

NICE_PROBE = r'''
import os, sys
from triforce import jobs
before = os.nice(0)
after = jobs.lower_priority(None if sys.argv[1] == 'default' else int(sys.argv[1]))
print(before, after, os.nice(0))
'''


@pytest.mark.parametrize('arg,env,expected', [
    ('default', None, 15), ('default', '9', 9), ('4', '9', 4), ('0', None, 0), ('default', 'x', 15)])
def test_lower_priority_sets_the_niceness_once_and_never_raises_priority(arg, env, expected):
    environ = {k: v for k, v in os.environ.items() if k != jobs.NICE_ENV}
    if env is not None:
        environ[jobs.NICE_ENV] = env
    out = subprocess.run([sys.executable, '-c', NICE_PROBE, arg], env=environ, capture_output=True, text=True,
                         check=True).stdout.split()
    before, after, now = map(int, out)
    assert after == now == max(before, expected)
    # Already lower: unchanged (a child of a niced batch).
    out = subprocess.run(['nice', '-n', '17', sys.executable, '-c', NICE_PROBE, arg], env=environ,
                         capture_output=True, text=True, check=True).stdout.split()
    assert int(out[1]) == max(int(out[0]), expected) >= 17


def test_child_env_passes_the_niceness_on():
    assert jobs.child_env(15, {'A': '1'}) == {'A': '1', jobs.NICE_ENV: '15'}
    assert jobs.child_env(None, {})[jobs.NICE_ENV] == str(os.nice(0))


# Only the training and evaluation entry points lower the priority, never library code that play and replay tools
# share (play-zelda, zelda-headless, create-video, zelda-map, tf replay, explore, entry_states).
ENTRY_POINTS = {('triforce/train.py', 'cli'), ('triforce/evaluate.py', 'main'),
                ('triforce/cli.py', '_cmd_eval_chain'), ('triforce/eval_batch.py', 'main'),
                ('scripts/train_round.py', 'main')}


def test_only_the_training_and_evaluation_entry_points_call_lower_priority():
    calls = set()
    files = [*ROOT.glob('triforce/*.py'), *ROOT.glob('scripts/**/*.py'), *ROOT.glob('*.py')]
    for path in files:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(function):
                if isinstance(node, ast.Call) and getattr(node.func, 'id', getattr(node.func, 'attr', None)) == \
                        'lower_priority':
                    calls.add((str(path.relative_to(ROOT)), function.name))
    assert calls == ENTRY_POINTS


PLAY_PROBE = r'''
import importlib.util, os, sys
before = os.nice(0)
import triforce.play, triforce.replay, triforce.explore, triforce.entry_states, triforce.evaluate, triforce.cli
for script in ('play-zelda.py', 'zelda-headless.py', 'create-video.py', 'zelda-map.py'):
    spec = importlib.util.spec_from_file_location(script.replace('-', '_')[:-3], os.path.join(sys.argv[1], script))
    try:
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    except ImportError:
        pass
print(before, os.nice(0))
'''


def test_play_and_replay_tools_keep_their_priority():
    environ = {k: v for k, v in os.environ.items() if k != jobs.NICE_ENV}
    environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    out = subprocess.run([sys.executable, '-c', PLAY_PROBE, str(ROOT)], env=environ, capture_output=True, text=True,
                         check=True, cwd=ROOT).stdout.split()
    assert out[-2] == out[-1]
