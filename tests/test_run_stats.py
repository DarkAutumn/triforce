"""Run statistics (``triforce.run_stats``): the file from a chain's records, an evaluation and a training; the result
flags; the store; the reader and the paired comparison."""

import json

import pytest

from triforce import run_stats as rs

STEPS = [{'index': i, 'level': 3, 'room': room, 'in_cave': False, 'specialist': family, 'exit': exit_, 'way': None}
         for i, (room, family, exit_) in enumerate([(0x7c, 'zol', 'W'), (0x7b, 'zol', 'N'), (0x6b, 'darknut', 'N'),
                                                    (0x5b, 'zol', 'triforce')])]


def visit(step, outcome='exit-correct', frames=100, health_in=3.0, health_out=3.0, new=True, **extra):
    room = STEPS[step]
    v = {'level': 3, 'room': f"0x{room['room']:02x}", 'in_cave': False, 'specialist': room['specialist'],
         'policy': f"{room['specialist']}@d3first", 'step': step, 'frames': frames, 'health_in': health_in,
         'health_out': health_out, 'max_health': 3, 'outcome': outcome, 'hits': 0, 'keys_in': 1, 'keys_out': 0,
         'bombs_used': 0, 'cheated_bombs': 0, 'cheated_keys': 0, 'cheated_rupees': 0, 'linger': {'max_frames': 5,
                                                                                            'tail_frames': 0},
         'decisions': {'n': 4}, 'beam_shots': 2, 'beam_shots_lined': 1, 'beam_lined': 3, 'beam_lined_unfired': 1,
         'beam_kills_far': 1, 'beam_kills_near': 1, 'beam_kills_other': 0, 'beam_kill_distance': 60,
         'beam_hearts_lost': health_in - health_out, 'beam_beams_lost': int(health_out < health_in)}
    if new:
        v.update({'beam_shots_hit': 1, 'beam_held_frames': frames if health_out == 3.0 else 40,
                  'beam_lost_frame': None if health_out == 3.0 else 40})
    v.update(extra)
    return v


def won(seed, new=True, slow=0):
    rooms = [visit(0, new=new), visit(1, new=new, frames=100 + slow), visit(2, new=new), visit(3, new=new)]
    rooms.append({**visit(3, new=new), 'specialist': '-', 'step': None, 'outcome': 'milestone', 'frames': 0})
    return {'route': 'd3first', 'seed': seed, 'result': 'success', 'reason': 'took the level 3 triforce',
            'frames': sum(v['frames'] for v in rooms), 'rooms': rooms}


def died(seed, new=True):
    rooms = [visit(0, new=new), visit(1, new=new, health_out=2.0),
             visit(2, 'death', new=new, health_in=2.0, health_out=0.0)]
    return {'route': 'd3first', 'seed': seed, 'result': 'failure', 'reason': 'died in (level 3, room 0x6b)',
            'frames': 300, 'rooms': rooms}


def meta(runs, models=('zol@d3first', 'darknut@d3first')):
    return {'provenance': {'route': 'd3first', 'clips': True, 'stop_step': 4,
                           'models': [{'name': m, 'run': 'r'} for m in models]},
            'batches': [{'first_seed': 0, 'runs': runs, 'finished': True}]}


def chain(runs, **kwargs):
    return rs.chain_stats(runs, kwargs.pop('meta', meta(len(runs))), STEPS, 4, source='/x/d3.jsonl', **kwargs)


def codes(stats, level=None):
    return sorted(f['code'] for f in stats['flags'] if level in (None, f['level']))


def test_chain_rate_endings_groups_time_and_beams():
    s = chain([won(0), died(1), won(2, slow=50), died(3)])
    assert s['name'] == 'd3' and s['kind'] == 'chain'
    assert s['rate'] == {'n': 4, 'successes': 2, 'rate': 0.5, 'se': 0.25}
    assert (s['endings']['deaths'], s['endings']['death_rooms']) == (2, {'3:0x6b': 2})
    assert s['triforces'] == {'3': 2}
    step2 = s['by_step']['2']
    assert (step2['visits'], step2['exits'], step2['deaths'], step2['family']) == (4, 2, 2, 'darknut')
    assert step2['hearts_lost'] == {'n': 4, 'sum': 4.0, 'sq': 8.0, 'mean': 1.0, 'se': 0.5774}
    assert step2['enter'] == {'n': 4, 'sum': 850, 'sq': 182500.0, 'mean': 212.5, 'se': 12.5, 'median': 200, 'p90': 250}
    assert s['runs']['furthest'] == [3, 2, 3, 2] and s['time']['3']['sum'] == 650
    assert s['by_family']['zol']['visits'] == 10 and s['all']['keys_used']['mean'] == 1.0
    assert s['all']['frames']['median'] == 100 and s['all']['shots_hit']['sum'] == 14
    # Frames to entering each step, per run, aligned with the seeds; the end for the winners.
    assert s['runs']['seeds'] == [0, 1, 2, 3] and s['runs']['result'] == 'wdwd'
    assert s['runs']['enter']['2'] == [200, 200, 250, 200] and s['runs']['enter']['3'] == [300, None, 350, None]
    assert s['time']['3']['n'] == 2 and s['runs']['end'] == [400, None, 450, None]
    # Beams: the room-level measure (the milestone counts) and round 40's exact frame.
    assert s['runs']['beam_rooms'] == [5, 1, 5, 1] and s['runs']['beam_frames'] == [400, 100, 450, 100]
    assert s['runs']['beam_lost_at'] == [None, 140, None, 140] and s['beams']['never_lost'] == 2
    assert s['records']['counters'] == 'round40' and codes(s) == []
    json.dumps(s)


def test_old_records_are_null_never_zero():
    s = chain([won(0, new=False), died(1, new=False)])
    assert s['all']['shots_hit'] is None and s['all']['held_frames'] is None
    assert s['runs']['shots_hit'] == [None, None] and s['beams']['held_frames'] is None
    assert codes(s) == ['records-old'] and s['flags'][0]['level'] == 'note'


def test_result_flags():
    runs = [won(i) for i in range(120)]
    assert codes(chain(runs)) == ['rate-extreme']
    runs[5] = {**died(5), 'result': 'error', 'reason': 'zol chose masked action'}
    for i in range(6, 16):
        runs[i] = {**died(i), 'reason': 'bounce-limit: zol left into (level 3, room 0x7b)'}
    s = chain(runs, meta=meta(150))
    assert codes(s, 'red') == ['run-error', 'runs-short', 'stuck-share']
    assert s['endings']['stuck_reasons'] == {'bounce-limit': 10}
    shifted = [{**won(i), 'seed': i + 1000} for i in range(3)]
    assert codes(chain(shifted, meta=meta(3))) == ['seeds-nonstandard']
    assert codes(chain([won(0), won(0)], meta=meta(2))) == ['seeds-duplicate']


def test_route_canaries():
    # Step 1 left by its exit in 30 runs and step 2 never entered: the records or the route are broken.
    runs = [{**won(i), 'rooms': [r for r in won(i)['rooms'] if r['step'] != 2]} for i in range(30)]
    s = chain(runs, meta=meta(30, ('zol@d3first',)))
    assert 'step-never-entered' in codes(s, 'red') and 'family-never-played' in codes(s, 'red')
    # A network the team lists for the steps played that made no decision (another played its rooms).
    runs = [won(i) for i in range(30)]
    for run in runs:
        for v in run['rooms']:
            if v['specialist'] == 'darknut':
                v['policy'] = 'darknut'
    assert 'network-no-decisions' in codes(chain(runs, meta=meta(30)), 'red')


def test_evaluation_statistics():
    episodes = [{'start_state': 's0', 'seed': i, 'success': i % 2 == 0, 'ending': 'exit-correct' if i % 2 == 0 else
                 'death', 'frames': 100, 'health': 3.0, 'visits': [{**visit(0), 'room': 0x7c, 'health_lost': 1.0}]}
                for i in range(30)]
    result = {'specialist': 'zol', 'checkpoint': '/m/zol/r1/ckpt.pt', 'records': episodes,
              'config': {'rooms': ['(3, 0x7c)', '(3, 0x7b)']}}
    s = rs.eval_stats(result, '/m/zol/r1/eval.json')
    assert s['rate']['successes'] == 15 and s['endings']['deaths'] == 15
    assert s['by_room']['3:0x7c']['hearts_lost']['mean'] == 1.0
    assert s['runs']['seeds'][0] == 's0#0' and codes(s, 'red') == ['family-room-never-played']
    assert rs.eval_name(result, '/m/zol/r1/eval.json') == 'zol-r1-eval'


def test_training_statistics():
    metrics = [{'iteration': i, 'steps': 1000 * i, 'balanced_success_rate': i / 20, 'success_rate': i / 20}
               for i in range(1, 11)]
    run = {'specialist': 'zol', 'run_id': 'r1', 'args': {'stop_success': 0.95, 'start_state': None},
           'harvested': {'dir': '/h/zol', 'states': 5, 'fraction': 0.25}, 'curriculum': '/t/zol',
           'curriculum_tracks': {}}
    s = rs.train_stats(run, metrics, {'status': 'completed', 'steps_per_second': 900.0}, None)
    assert s['reached']['0.5'] == {'iteration': 10, 'steps': 10000} and s['reached']['0.95'] is None
    assert s['steps_per_second'] == 900.0 and s['final']['success_rate']['n'] == 10
    assert codes(s, 'red') == ['curriculum-empty', 'harvested-none']
    metrics[3]['harvested/episodes'] = 7
    run['curriculum_tracks'] = {'t': 'x'}
    assert codes(rs.train_stats(run, metrics, None, None)) == []


def test_written_registered_moved_listed_shown_and_compared(tmp_path, capsys):
    out = tmp_path / 'd3.jsonl.tmp'
    out.write_text(''.join(json.dumps(r) + '\n' for r in [won(0), died(1)]))
    (tmp_path / 'd3.jsonl.tmp.meta.json').write_text(json.dumps({**meta(2), 'provenance': {'route': 'nowhere'}}))
    written = rs.write_chain(out)
    assert written == tmp_path / 'd3.jsonl.tmp.stats.json'
    stored = rs.store_path() / '0-d3.stats.json'
    assert json.loads(stored.read_text())['job'] == {'id': 0}
    # A batch moves its chain from .tmp: the statistics follow, under the same key.
    final = tmp_path / 'd3.jsonl'
    out.rename(final)
    rs.move_chain(out, final)
    assert not written.exists() and rs.stats_path(final).is_file()
    assert json.loads(stored.read_text())['source'] == str(final.resolve())
    arm = tmp_path / 'arm.jsonl'
    arm.write_text(''.join(json.dumps({**r, 'route': 'nowhere'}) + '\n' for r in [won(0, slow=60), won(1)]))
    rs.write_chain(arm)
    assert rs.main(['list']) == 0
    listed = capsys.readouterr().out
    assert '0-d3' in listed and '0-arm' in listed
    assert rs.main(['show', '0-d3', '--rooms', '2']) == 0
    assert '## Route steps' in capsys.readouterr().out
    assert rs.main(['compare', str(final), str(arm), '--min-pairs', '1']) == 0
    text = capsys.readouterr().out
    assert '| 0-arm | 2 | 2 | 100.0% | 2 | +50.0 ± 50.0 |' in text
    assert '| 2 | 3:0x6b darknut | 2 | 200 | 230 | +30 ± 30 | +0.5 |' in text
    with pytest.raises(SystemExit):
        rs.load('no-such-file')


def test_ingest_registers_without_writing_beside(tmp_path, capsys):
    src = tmp_path / 'theirs.jsonl'
    src.write_text(json.dumps(won(0)) + '\n')
    assert rs.main(['ingest', str(src), '--agent', 'Other', '--task', 'T']) == 0
    assert not rs.stats_path(src).exists()
    stats = rs.load('0-theirs')
    assert (stats['job']['agent'], stats['job']['task']) == ('Other', 'T')
    assert '0-theirs: chain' in capsys.readouterr().out


def test_what_a_row_is():
    command = ['triforce', 'eval-batch', 'x', '--chain',
               'd3: --from 3_7cs_0 --checkpoint zol@d3first=/m/zol/r40-x-s401 --output /o/a.jsonl',
               '--chain', 'd4: --from 4_71s_0 --output /o/b.jsonl']
    assert rs.checkpoint_overrides(command, '/o/a.jsonl') == ['zol@d3first = r40-x-s401']
    assert rs.checkpoint_overrides(command, '/o/b.jsonl') == []
    assert rs.checkpoint_overrides(['eval-chain', '--checkpoint', 'zol=/m/zol/r1/ckpt.pt']) == ['zol = r1']
    s = rs.chain_stats([won(0)], {**meta(1), 'provenance': {**meta(1)['provenance'],
                                                             'start_from': 'd3first_3_7cs_0'}},
                       STEPS, 4, segment='d3', overrides=['zol@d3first = r40-x-s401'])
    assert s['description'] == 'd3 (3_7cs_0 to step 4), 1 runs, zol@d3first = r40-x-s401'


def test_compare_summary_in_the_owners_order():
    base = chain([won(i) if i % 2 else died(i) for i in range(40)], meta=meta(40))
    arm = chain([won(i, slow=30) for i in range(40)], meta=meta(40))
    lines = rs.summary_lines(base, arm, 20)
    assert lines[0].startswith('- rate: 50.0% -> 100.0%, +50.0 ± 8.0 points (40 pairs): **better**')
    assert lines[1].endswith('**better**') and 'hearts lost' in lines[1]
    assert 'within noise' in rs.verdict(0.1, 0.06, True) and rs.verdict(-0.2, 0.06, False) == 'better'
    assert lines[-1].startswith('- time to enter step 3 (3:0x5b)') and lines[-1].endswith('**worse**')


def test_an_evaluations_wrong_exits_are_not_stuck_runs():
    episodes = [{'start_state': 's0', 'seed': i, 'success': False, 'ending': 'exit-wrong', 'frames': 100,
                 'health': 3.0, 'visits': [{**visit(0), 'room': 0x7c, 'health_lost': 0.0}]} for i in range(30)]
    s = rs.eval_stats({'specialist': 'zol', 'checkpoint': '/m/zol/r1/ckpt.pt', 'records': episodes,
                       'config': {'rooms': ['(3, 0x7c)']}})
    assert s['endings']['stuck'] == 30 and codes(s, 'red') == []


def test_route_canaries_skip_the_stop_step_and_other_route_versions():
    runs = [won(i) for i in range(30)]
    stop = {**meta(30), 'provenance': {**meta(30)['provenance'], 'stop_step': 3}}
    for run in runs:            # the stop step is reached (the milestone), never played; a few runs go beyond
        run['rooms'] = [r for r in run['rooms'] if r['step'] != 3]
    runs[0]['rooms'].insert(-1, {**visit(3), 'step': 3})
    assert 'step-never-entered' not in codes(chain(runs, meta=stop))
    renumbered = [won(i) for i in range(30)]
    for run in renumbered:
        run['rooms'][1]['room'] = '0x11'
    s = chain(renumbered, meta=meta(30))
    assert 'route-renumbered' in codes(s, 'note') and 'family-never-played' not in codes(s)
