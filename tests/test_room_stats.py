"""triforce.room_stats: per-room statistics by scope from chain files (full run, segments) and evaluations (room)."""

import json

import pytest

from triforce import _native, registry, room_stats
from triforce.room_stats import Scope

# A small route: 0:0x10 -> 0:0x11 -> 0:0x12 -> down into level 1 (1:0x20, the triforce in 1:0x10) -> out of 0:0x12
# (the second segment's start, '0_12c_0') -> back through 0:0x11 (on both segments) -> 0:0x01.
PLAN = [{'index': i, 'level': level, 'room': room, 'in_cave': False, 'entry': entry, 'exit': exit_}
        for i, (level, room, entry, exit_) in enumerate((
            (0, 0x10, 'start', 'E'), (0, 0x11, 'W', 'E'), (0, 0x12, 'W', 'cave'), (1, 0x20, 'S', 'N'),
            (1, 0x10, 'S', 'triforce'), (0, 0x12, 'cave', 'W'), (0, 0x11, 'E', 'N'), (0, 0x01, 'S', 'N')))]
FULL = Scope('full', 'full', 'Full', 'full', 'start', None)
FIRST = Scope('seg-a', 'segment', 'Start -> out of level 1', 'full', 'start', '0_12c_0')
SECOND = Scope('seg-b', 'segment', 'Out of level 1 -> end', 'post', '0_12c_0', None)


def visit(level, room, outcome, health_in=3.0, health_out=3.0, specialist='ow', **extra):
    return {'index': 0, 'level': level, 'room': f'0x{room:02x}', 'in_cave': False, 'specialist': specialist,
            'steps': 10, 'frames': 100, 'health_in': health_in, 'health_out': health_out, 'max_health': 3,
            'outcome': outcome, 'hearts_healed': 0.0, 'bombs_used': 0, 'cheated_bombs': 0, **extra}


def run(seed, rooms, result='failure'):
    return {'route': 'test', 'seed': seed, 'result': result, 'reason': '', 'rooms': rooms}


def ok(level, room, **kw):
    return visit(level, room, 'exit-correct', **kw)


THROUGH = [ok(0, 0x10), ok(0, 0x11), ok(0, 0x12), ok(1, 0x20), ok(1, 0x10)]
FULL_RUNS = [
    # Wins: through everything, the milestone marker after the last room.
    run(0, [*THROUGH, ok(0, 0x12), ok(0, 0x11), ok(0, 0x01), visit(0, 0x01, 'milestone')], 'success'),
    # Dies in 0:0x11 on its second visit (step 6, after the cut).
    run(1, [*THROUGH, ok(0, 0x12), visit(0, 0x11, 'death', health_in=1.0, health_out=0.0)]),
    # A wrong exit off the route and back, then stuck in 1:0x20.
    run(2, [*THROUGH[:3], visit(1, 0x20, 'exit-wrong'), visit(1, 0x21, 'exit-wrong'), visit(1, 0x20, 'stuck')]),
    # Dies in 0:0x11 on its first visit.
    run(3, [ok(0, 0x10), visit(0, 0x11, 'death', health_in=2.0, health_out=0.0)]),
]


def test_a_run_that_dies_in_a_room_visited_twice_counts_once_per_run():
    scope = room_stats.chain_scope(FULL_RUNS, PLAN, FULL)
    assert (scope['runs'], scope['arrived'], scope['endings']) == (4, 1, {'death': 2, 'stuck': 1})
    room = scope['rooms']['0:0x11']
    assert (room['entered'], room['visits'], room['deaths'], room['death_rate']) == (4, 6, 2, 0.5)
    assert room['reach_rate'] == 1.0 and room['through_rate'] == 0.5 and room['death_rate_entered'] == 0.5
    assert room['hearts_lost_per_visit'] == pytest.approx(3.0 / 6) and room['hearts_change_per_visit'] < 0
    # Per route step: the first visit's death on step 1, the second's on step 6.
    assert {key: (s['visits'], s['deaths']) for key, s in room['steps'].items()} == {'1': (4, 1), '6': (2, 1)}
    assert list(scope['rooms'])[:3] == ['0:0x10', '0:0x11', '0:0x12']        # route order
    assert room['hits_per_visit'] is None and room['clips_per_visit'] is None


def test_a_stuck_run_and_a_visit_off_the_route():
    scope = room_stats.chain_scope(FULL_RUNS, PLAN, FULL)
    room = scope['rooms']['1:0x20']
    assert (room['stuck'], room['stuck_rate'], room['deaths'], room['wrong_exits']) == (1, 0.25, 0, 1)
    assert room['endings'] == {'stuck': 1} and room['outcomes'] == {'exit-correct': 2, 'exit-wrong': 1, 'stuck': 1}
    assert list(scope['rooms']['1:0x21']['steps']) == ['off'] and scope['off_route_visits'] == 1


def test_a_segment_stops_at_the_first_entry_into_the_next_segments_start_step():
    scope = room_stats.chain_scope(FULL_RUNS, PLAN, FIRST)
    assert (scope['start_step'], scope['cut_step'], scope['arrived'], scope['arrival_rate']) == (0, 5, 2, 0.5)
    # 0:0x12 is entered on step 2 and again on step 5 (the cut): only the first counts here.
    assert list(scope['rooms']['0:0x12']['steps']) == ['2'] and scope['rooms']['0:0x12']['visits'] == 3
    # Run 1 died in 0:0x11 after the cut: in this segment only run 3 died there.
    room = scope['rooms']['0:0x11']
    assert (room['visits'], room['deaths'], room['death_rate'], list(room['steps'])) == (4, 1, 0.25, ['1'])
    assert '0:0x01' not in scope['rooms']
    assert scope['endings'] == {'death': 1, 'stuck': 1}


def test_a_room_on_two_segments_has_an_entry_in_each():
    post = [run(0, [ok(0, 0x12), visit(0, 0x11, 'death')]),
            run(1, [ok(0, 0x12), ok(0, 0x11), ok(0, 0x01), visit(0, 0x01, 'milestone')], 'success')]
    first = room_stats.chain_scope(FULL_RUNS, PLAN, FIRST)['rooms']['0:0x11']
    second = room_stats.chain_scope(post, PLAN, SECOND)
    assert second['start_step'] == 5 and second['arrival_rate'] == 0.5
    room = second['rooms']['0:0x11']
    assert (room['runs'], room['deaths'], room['death_rate'], list(room['steps'])) == (2, 1, 0.5, ['6'])
    assert first['death_rate'] == 0.25


def test_steps_follow_the_real_route_through_wrong_exits_and_retreats():
    plan = list(_native.route_steps('d2first'))

    def step_of(level, room, entry):
        return next(s['index'] for s in plan if (s['level'], s['room'], s['entry']) == (level, room, entry))
    for state, (level, room, entry) in (('0_3cc_0', (0, 0x3c, 'cave')), ('0_37c_0', (0, 0x37, 'cave')),
                                        ('0_74c_0', (0, 0x74, 'cave')), ('2_7ds_0', (2, 0x7d, 'S')),
                                        ('start', (0, 0x77, 'start'))):
        assert room_stats.state_step(plan, state) == step_of(level, room, entry)
    with pytest.raises(room_stats.RoomStatsError):
        room_stats.state_step(plan, '9_00s_0')
    # Dungeon 1: a wrong exit N out of 0x53 skips the route's 0x52 and 0x42; the game's objective goes on from 0x43.
    start = step_of(1, 0x63, 'S')
    records = [ok(1, 0x63), visit(1, 0x53, 'exit-wrong'), ok(1, 0x43), ok(1, 0x33)]
    assert room_stats.assign_steps(plan, records, start) == [start, start + 1, step_of(1, 0x43, 'W'),
                                                             step_of(1, 0x33, 'S')]
    # 0x0a S before the white sword is a retreat: back on 0x1a's first step, then N again.
    start = step_of(0, 0x1a, 'E')
    records = [ok(0, 0x1a), visit(0, 0x0a, 'retreat'), ok(0, 0x1a), ok(0, 0x0a)]
    assert room_stats.assign_steps(plan, records, start) == [start, start + 1, start, start + 1]


def test_recorded_steps_hold_on_both_ways_of_both_stretches():
    """Round 22's route: a chain record names its step, so a visit is counted on the way it played: ``d3-approach``
    (0x67 out of the cave: S to 0x77, 0x76, 0x75 and 0x74 from E with clips; W along row 6, 0x63 S, 0x73 and 0x74
    from W without) and ``d4-island`` (4:0x31 from E: N onto 0x21's island, or W round by 0x30 and 0x20)."""
    plan = list(_native.route_steps('d2first'))
    for name, state in (('d3-approach', '0_67c_0'), ('d4-island', '4_31e_0')):
        ways = {way: [s for s in plan if s.get('stretch') == name and s.get('way') == way] for way in ('clips', 'plain')}
        assert ways['clips'] and ways['plain'], name
        first = room_stats.state_step(plan, state)
        assert first == ways['clips'][0]['index'], name
        rejoin = plan[ways['plain'][-1]['index'] + 1]
        for way, steps in ways.items():
            records = [ok(s['level'], s['room'], step=s['index']) for s in steps]
            records.append(ok(rejoin['level'], rejoin['room'], step=rejoin['index']))
            assert room_stats.assign_steps(plan, records, first) == [*(s['index'] for s in steps), rejoin['index']], \
                (name, way)
    rooms = {(s['stretch'], s['way']): [] for s in plan if s.get('stretch')}
    for s in plan:
        if s.get('stretch'):
            rooms[s['stretch'], s['way']].append((s['level'], s['room'], s['entry'], s['exit']))
    assert rooms['d3-approach', 'clips'] == [(0, 0x67, 'cave', 'S'), (0, 0x77, 'N', 'W'), (0, 0x76, 'E', 'W'),
                                             (0, 0x75, 'E', 'W'), (0, 0x74, 'E', 'cave')]
    assert rooms['d3-approach', 'plain'] == [(0, 0x67, 'cave', 'W'), (0, 0x66, 'E', 'W'), (0, 0x65, 'E', 'W'),
                                             (0, 0x64, 'E', 'W'), (0, 0x63, 'E', 'S'), (0, 0x73, 'N', 'E'),
                                             (0, 0x74, 'W', 'cave')]
    assert rooms['d4-island', 'clips'] == [(4, 0x31, 'E', 'N'), (4, 0x21, 'S', 'N')]
    assert rooms['d4-island', 'plain'] == [(4, 0x31, 'E', 'W'), (4, 0x30, 'E', 'N'), (4, 0x20, 'S', 'E'),
                                           (4, 0x21, 'W', 'N')]


def write_chain(directory, name, runs, start_from, team, tag='tag', **provenance):
    path = directory / f'{tag}-{name}.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in runs))
    models = [{'name': specialist, 'checkpoint': f'{specialist}/{run_id}/ckpt.pt', 'act': act}
              for specialist, (run_id, act) in team.items()]
    meta = {'provenance': {'route': 'test', 'start_from': start_from, 'models': models, 'code': 'c', 'data': 'd',
                           'version': '0.0.0.0', **provenance}}
    path.with_name(path.name + '.meta.json').write_text(json.dumps(meta))


def write_eval(models, specialist, run_id, act, visited, **extra):
    directory = models / specialist / run_id
    directory.mkdir(parents=True)
    (directory / 'ckpt.pt').write_bytes(specialist.encode())
    registry.set_act_mode(directory, act)
    (directory / 'eval.json').write_text(json.dumps({'format': 2, 'act': act, 'visited_rooms': visited, **extra}))


def eval_room(level, room, visits, deaths, timeouts=0):
    return {'level': level, 'room': room, 'in_cave': False, 'visits': visits, 'episodes': visits, 'deaths': deaths,
            'death_rate': deaths / visits, 'steps_per_visit': 50.0, 'health_lost_per_visit': 1.0,
            'health_gained_per_visit': 0.5, 'health_change_per_visit': -0.5, 'hits_per_visit': 2.0,
            'clip_done_per_visit': 0.25, 'bombs_used_per_visit': 0.1,
            'exits': {'e': visits - deaths - timeouts, 'death': deaths, 'timeout': timeouts}}


def test_build_writes_every_scope_with_its_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(_native, 'route_steps', lambda name: PLAN)
    monkeypatch.setattr(room_stats, 'SCOPES', (FULL, FIRST, SECOND))
    chains, models = tmp_path / 'chains', tmp_path / 'models'
    chains.mkdir()
    team = {'ow-octorok': ('run-a', 'argmax'), 'goriya': ('run-b', 'sample')}
    write_chain(chains, 'full', FULL_RUNS, None, team)
    write_eval(models, 'ow-octorok', 'run-a', 'argmax', {'0:0x11': eval_room(0, 0x11, 100, 2, timeouts=1)})
    write_eval(models, 'goriya', 'run-b', 'sample', {'1:0x20': eval_room(1, 0x20, 50, 5)})
    (models / 'goriya' / 'promoted').write_text('run-b\n')

    assert room_stats.main(['tag', '--chains-dir', str(chains), '--models-dir', str(models)]) == 0
    path = room_stats.output_path('tag', chains)
    assert room_stats.newest(chains) == path
    data = room_stats.read(path)
    assert (data['format'], data['tag'], data['route']) == (room_stats.FORMAT, 'tag', 'test')
    assert set(data['scopes']) == {'full', 'seg-a', 'room'}         # no post file: no seg-b
    assert data['scopes']['full']['source'] == {'file': 'tag-full.jsonl', 'runs': 4, 'start_from': None,
                                                'route': 'test', 'act': {'argmax': 1, 'sample': 1}, 'clips': None,
                                                'clips_off_from': None, 'code': 'c', 'data': 'd',
                                                'version': '0.0.0.0'}
    assert data['scopes']['full']['files'] == [data['scopes']['full']['source']]
    assert (data['tags'], data['noclips']) == (['tag'], None)
    assert data['team']['goriya']['run'] == 'run-b' and data['promoted'] == {'goriya': 'run-b'}
    assert [s['name'] for s in data['segments']] == ['full', 'seg-a', 'seg-b']
    room = data['scopes']['room']
    assert [(s['specialist'], s['status'], s['mode']) for s in room['sources']] == [
        ('goriya', 'detailed', 'sample'), ('ow-octorok', 'detailed', 'argmax')]
    entry = room['rooms']['0:0x11']
    assert (entry['runs'], entry['deaths'], entry['death_rate'], entry['stuck'], entry['stuck_rate']) == (
        100, 2, 0.02, 1, 0.01)
    assert (entry['hits_per_visit'], entry['clips_per_visit'], entry['hearts_change_per_visit']) == (2.0, 0.25, -0.5)
    assert entry['reach_rate'] is None and entry['steps'] is None
    # Every scope's room entries carry the same fields.
    chain_entry = dict(data['scopes']['seg-a']['rooms']['0:0x11'])
    assert set(chain_entry) == set(entry)
    # Files and evaluations from before the clips switch: every visit played on.
    assert list(entry['by_clips']) == ['on'] and entry['by_clips']['on']['deaths'] == 2
    assert list(chain_entry['by_clips']) == ['on'] and list(data['scopes']['full']['by_clips']) == ['on']
    assert chain_entry['by_clips']['on'] == {k: v for k, v in chain_entry.items()
                                             if k not in ('level', 'room', 'in_cave', 'by_clips')}


def test_build_refuses_a_file_that_starts_elsewhere(tmp_path, monkeypatch):
    monkeypatch.setattr(_native, 'route_steps', lambda name: PLAN)
    monkeypatch.setattr(room_stats, 'SCOPES', (FULL,))
    write_chain(tmp_path, 'full', FULL_RUNS, '0_12c_0', {})
    with pytest.raises(room_stats.RoomStatsError, match='starts from 0_12c_0'):
        room_stats.build('tag', tmp_path, tmp_path / 'models')
    with pytest.raises(room_stats.RoomStatsError, match='no chain files'):
        room_stats.build('other', tmp_path, tmp_path / 'models')


# Round 22: the clips switch.  One file with runs on, off, a run that turns the switch off on step 2 (eval-chain
# --clips-off-from step:2) and a run from before the switch (no fields: on, steps inferred).
def on(level, room, outcome='exit-correct', **kw):
    return visit(level, room, outcome, clips_allowed=True, **kw)


def off(level, room, outcome='exit-correct', **kw):
    return visit(level, room, outcome, clips_allowed=False, **kw)


def clips_run(seed, clips, rooms, result='failure'):
    return {**run(seed, rooms, result), 'clips': clips}


MIXED_RUNS = [
    clips_run(0, True, [on(0, 0x10, step=0, hits=1), on(0, 0x11, step=1, hits=1), on(0, 0x12, step=2),
                        on(1, 0x20, step=3), on(1, 0x10, step=4), on(0, 0x12, step=5), on(0, 0x11, step=6, hits=1),
                        on(0, 0x01, step=7), visit(0, 0x01, 'milestone')], 'success'),
    clips_run(1, False, [off(0, 0x10, step=0), off(0, 0x11, 'death', step=1, hits=2, health_out=0.0)]),
    clips_run(2, True, [on(0, 0x10, step=0), on(0, 0x11, step=1, hits=0),
                        off(0, 0x12, 'death', step=2, health_out=0.0, clips_flips=[{'step': 0, 'clips': False}])]),
    run(3, [ok(0, 0x10), visit(0, 0x11, 'death', health_out=0.0)]),
]


def test_one_file_with_mixed_runs_splits_every_room_by_the_switch():
    scope = room_stats.chain_scope(MIXED_RUNS, PLAN, FULL)
    assert (scope['runs'], scope['arrived'], scope['endings']) == (4, 1, {'death': 3})
    # Runs that played a visit with each switch: the flipping run 2 in both.
    assert scope['by_clips'] == {
        'on': {'runs': 3, 'arrived': 1, 'arrival_rate': 0.33333, 'endings': {'death': 1}, 'off_route_visits': 0},
        'off': {'runs': 2, 'arrived': 0, 'arrival_rate': 0.0, 'endings': {'death': 2}, 'off_route_visits': 0}}
    room = scope['rooms']['0:0x11']                     # the combined numbers stay what they were
    assert (room['runs'], room['entered'], room['visits'], room['deaths'], room['death_rate']) == (4, 4, 5, 2, 0.5)
    on_, off_ = room['by_clips']['on'], room['by_clips']['off']
    assert (on_['runs'], on_['entered'], on_['visits'], on_['deaths'], on_['death_rate']) == (3, 3, 4, 1, 0.33333)
    assert (off_['runs'], off_['entered'], off_['visits'], off_['deaths'], off_['death_rate']) == (2, 1, 1, 1, 0.5)
    assert list(on_['steps']) == ['1', '6'] and list(off_['steps']) == ['1']
    assert on_['steps']['1']['visits'] == 3 and 'level' not in on_
    # Hits: the records' counts per visit that counts them (run 3, from before the field, doesn't).
    assert (room['hits_per_visit'], on_['hits_per_visit'], off_['hits_per_visit']) == (1.0, 0.66667, 2.0)
    assert scope['rooms']['1:0x20']['hits_per_visit'] is None
    # 0:0x12: run 0 crossed it twice with the switch on, run 2 died there after turning it off.
    room = scope['rooms']['0:0x12']
    assert {switch: (e['visits'], e['deaths'], e['death_rate']) for switch, e in room['by_clips'].items()} == {
        'on': (2, 0, 0.0), 'off': (1, 1, 0.5)}
    assert list(scope['rooms']['1:0x20']['by_clips']) == ['on']        # only on visits: no off entry
    # A segment splits the same way within its cut.
    segment = room_stats.chain_scope(MIXED_RUNS, PLAN, FIRST)
    assert segment['by_clips']['on']['arrived'] == 1 and segment['rooms']['0:0x11']['by_clips']['on']['visits'] == 3


def test_assign_steps_uses_the_recorded_step_and_infers_without_it():
    # Recorded: kept even where the cursor would pick another step of the room (a stretch's other way); a recorded
    # step of another room (a changed route) or None is off the route.
    records = [ok(0, 0x10, step=0), ok(0, 0x11, step=6), visit(0, 0x11, 'exit-wrong', step=3),
               visit(1, 0x21, 'exit-wrong', step=None), visit(0, 0x01, 'milestone', step=7)]
    assert room_stats.assign_steps(PLAN, records, 0) == [0, 6, None, None, None]
    # Without the field (files from before round 22): inferred along the route.
    records = [ok(0, 0x10), ok(0, 0x11), ok(0, 0x12), ok(1, 0x20)]
    assert room_stats.assign_steps(PLAN, records, 0) == [0, 1, 2, 3]
    # A recorded correct exit moves the cursor for a later record without one.
    assert room_stats.assign_steps(PLAN, [ok(0, 0x12, step=5), ok(0, 0x11)], 0) == [5, 6]


NOCLIPS_RUNS = [   # a switch-off tag: neither the runs nor the records name the switch, the file's meta does
    run(0, [*THROUGH, ok(0, 0x12), ok(0, 0x11), ok(0, 0x01), visit(0, 0x01, 'milestone')], 'success'),
    run(1, [ok(0, 0x10), visit(0, 0x11, 'death', health_out=0.0)]),
]


def test_two_tags_read_together_split_by_the_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(_native, 'route_steps', lambda name: PLAN)
    monkeypatch.setattr(room_stats, 'SCOPES', (FULL, FIRST, SECOND))
    chains, models = tmp_path / 'chains', tmp_path / 'models'
    chains.mkdir()
    post = [run(0, [ok(0, 0x12), visit(0, 0x11, 'death')])]
    write_chain(chains, 'full', FULL_RUNS, None, {}, clips=True)
    write_chain(chains, 'post', post, '0_12c_0', {}, clips=True)
    write_chain(chains, 'full', NOCLIPS_RUNS, None, {}, tag='tag-noclips', clips=False)
    # A segment the switch-off tag didn't measure, linked from the base tag: read once.
    (chains / 'tag-noclips-post.jsonl').symlink_to('tag-post.jsonl')

    assert room_stats.main(['tag', '--noclips', '--chains-dir', str(chains), '--models-dir', str(models)]) == 0
    path = chains / 'tag-both-room-stats.json'
    assert room_stats.output_path('tag', chains, 'tag-noclips') == path and room_stats.newest(chains) == path
    data = room_stats.read(path)
    assert (data['tag'], data['tags'], data['noclips']) == ('tag', ['tag', 'tag-noclips'], 'tag-noclips')
    full = data['scopes']['full']
    assert [(f['file'], f['runs'], f['clips']) for f in full['files']] == [
        ('tag-full.jsonl', 4, True), ('tag-noclips-full.jsonl', 2, False)]
    assert full['runs'] == 6 and {s: t['runs'] for s, t in full['by_clips'].items()} == {'on': 4, 'off': 2}
    assert (full['by_clips']['on']['arrived'], full['by_clips']['off']['arrived']) == (1, 1)
    room = full['rooms']['0:0x11']
    assert (room['runs'], room['deaths'], room['death_rate']) == (6, 3, 0.5)
    assert {s: (e['runs'], e['deaths'], e['death_rate']) for s, e in room['by_clips'].items()} == {
        'on': (4, 2, 0.5), 'off': (2, 1, 0.5)}
    assert data['scopes']['seg-b']['runs'] == 1 and len(data['scopes']['seg-b']['files']) == 1
    # One tag alone writes the plain name; another switch-off tag by name; a missing one is an error.
    assert room_stats.main(['tag', '--chains-dir', str(chains), '--models-dir', str(models)]) == 0
    assert room_stats.read(chains / 'tag-room-stats.json')['scopes']['full']['by_clips'].keys() == {'on'}
    assert room_stats.main(['tag', '--noclips', 'tag-noclips', '--chains-dir', str(chains),
                            '--models-dir', str(models), '--output', str(tmp_path / 'x.json')]) == 0
    assert room_stats.main(['tag', '--noclips', 'nope', '--chains-dir', str(chains),
                            '--models-dir', str(models)]) == 2


def test_the_room_scope_splits_by_the_evaluations_by_clips(tmp_path):
    models = tmp_path / 'models'
    both = {'1:0x20': eval_room(1, 0x20, 100, 10)}
    write_eval(models, 'goriya', 'run-b', 'sample', both, clips='both', by_clips={
        'on': {'episodes': 60, 'visited_rooms': {'1:0x20': eval_room(1, 0x20, 60, 3)}},
        'off': {'episodes': 40, 'visited_rooms': {'1:0x20': eval_room(1, 0x20, 40, 7)}}})
    write_eval(models, 'ow-octorok', 'run-a', 'argmax', {'0:0x11': eval_room(0, 0x11, 100, 2)})   # before the switch
    team = {'goriya': {'run': 'run-b', 'act': 'sample'}, 'ow-octorok': {'run': 'run-a', 'act': 'argmax'}}
    scope = room_stats.room_scope(models, team)
    assert [(s['specialist'], s['clips'], s['visits']) for s in scope['sources']] == [
        ('goriya', 'both', 100), ('ow-octorok', None, 100)]
    room = scope['rooms']['1:0x20']
    assert (room['runs'], room['deaths'], room['death_rate']) == (100, 10, 0.1)
    assert {s: (e['runs'], e['deaths'], e['death_rate']) for s, e in room['by_clips'].items()} == {
        'on': (60, 3, 0.05), 'off': (40, 7, 0.175)}
    assert 'level' not in room['by_clips']['on'] and room['by_clips']['off']['steps'] is None
    assert list(scope['rooms']['0:0x11']['by_clips']) == ['on']
    merged, _ = room_stats.load_evals_by_clips(models, team=team)
    assert room_stats.load_evals(models, team=team)[0] == merged['both']


def test_old_room_stats_files_still_read(tmp_path):
    path = tmp_path / 'old-room-stats.json'
    path.write_text(json.dumps({'format': 1, 'tag': 'old', 'route': 'test', 'scopes': {}}))
    assert room_stats.read(path)['tag'] == 'old'
    path.write_text(json.dumps({'format': 0, 'scopes': {}}))
    with pytest.raises(ValueError, match='format 1 or 2'):
        room_stats.read(path)
