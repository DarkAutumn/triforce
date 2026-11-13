"""zelda-map.py: savestate names, merging evals' per-room stats, room-stats scopes, the window (Qt offscreen),
screenshot rendering."""

import importlib.util
import json
import os
import re
import shutil
from pathlib import Path

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QPointF  # noqa: E402  pylint: disable=wrong-import-position
from PySide6.QtWidgets import QApplication  # noqa: E402  pylint: disable=wrong-import-position

from triforce.room_stats import FORMAT as ROOM_STATS_FORMAT  # noqa: E402  pylint: disable=wrong-import-position

from .conftest import require_state  # noqa: E402  pylint: disable=wrong-import-position

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('zelda_map', ROOT / 'zelda-map.py')
zelda_map = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zelda_map)


@pytest.fixture(scope='module')
def qapp():
    return QApplication.instance() or QApplication([])


def room_stats(specialist, level, room, visits, deaths=0, change=-0.5, hits=1.0, drops=None, exits=None):
    """One visited_rooms entry (the eval.json format 2 contract)."""
    return {'level': level, 'room': room, 'in_cave': False, 'visits': visits, 'episodes': visits, 'deaths': deaths,
            'death_rate': deaths / visits, 'steps_per_visit': 100.0, 'health_lost_per_visit': -min(change, 0),
            'health_gained_per_visit': max(change, 0), 'health_change_per_visit': change, 'hits_per_visit': hits,
            'no_hit_rate': 0.5, 'hits_dealt_per_visit': 3.0, 'keys_per_visit': 0.0,
            'drops_per_visit': drops or {}, 'pickups_per_visit': {}, 'exits': exits or {'n': visits - deaths},
            'specialist': specialist}


def write_eval(models, specialist, data, act=None):
    run = models / specialist / 'run1'
    run.mkdir(parents=True)
    (run / 'ckpt.pt').write_bytes(specialist.encode())
    (models / specialist / 'promoted').write_text('run1\n')
    if act is not None:
        zelda_map.registry.set_act_mode(run, act)
    if data is not None:
        (run / 'eval.json').write_text(json.dumps(data))


@pytest.fixture
def models(tmp_path):
    """goriya (samples) and d2-rope (argmax) both visited 1:0x44; ow-zora has an old-format eval; ow-start has no
    eval.json; ow-octorok's eval.json was measured by argmax but the model samples; d1-melee is a retired family
    (not in specialists/)."""
    models = tmp_path / 'models'
    write_eval(models, 'goriya', {'format': 2, 'act': 'sample', 'visited_rooms': {
        '1:0x44': room_stats('goriya', 1, 0x44, 30, deaths=3, change=-1.0, hits=2.0, drops={'heart': 0.5},
                             exits={'n': 20, 'w': 7, 'death': 3}),
        '1:0x35': room_stats('goriya', 1, 0x35, 10)}})
    write_eval(models, 'd2-rope', {'format': 2, 'act': 'argmax', 'visited_rooms': {
        '1:0x44': room_stats('d2-rope', 1, 0x44, 10, change=1.0, hits=0.0, drops={'rupee': 1.0},
                             exits={'n': 10})}}, act='argmax')
    write_eval(models, 'ow-zora', {'specialist': 'ow-zora', 'success_rate': 0.9, 'deterministic': False})
    write_eval(models, 'ow-start', None)
    write_eval(models, 'ow-octorok', {'format': 2, 'deterministic': True, 'visited_rooms': {
        '0:0x37': room_stats('ow-octorok', 0, 0x37, 10)}})
    write_eval(models, 'd1-melee', {'format': 2, 'act': 'sample', 'visited_rooms': {}})   # retired: not a source
    (models / 'stray').mkdir()                  # no promoted model: not a source
    return models


def test_state_names_and_picture_choice(tmp_path):
    assert zelda_map.parse_state_name('0_4ce_1') == (0, 0x4c, 'e', 1)
    assert zelda_map.parse_state_name('0_77cave_0') == (0, 0x77, 'cave', 0)
    assert zelda_map.parse_state_name('0_37c_2') == (0, 0x37, 'c', 2)
    assert zelda_map.parse_state_name('start') is None
    for name in ('0_37c_0', '0_37e_0', '0_37e_1', '0_77cave_0', 'start'):
        (tmp_path / f'{name}.state').write_bytes(b'')
    states = zelda_map.scan_states(tmp_path)
    assert states == {(0, 0x37, False): ['0_37c_0', '0_37e_0', '0_37e_1'], (0, 0x77, True): ['0_77cave_0']}
    assert zelda_map.pick_state(states[(0, 0x37, False)]) == '0_37e_0'
    # A subfolder's states (the Dungeon 4 scouting states) are found too; the folder's own picture a shared room.
    (tmp_path / 'd4').mkdir()
    for name in ('0_37e_0', '4_32w_0', 'idle_12'):
        (tmp_path / 'd4' / f'{name}.state').write_bytes(b'')
    states = zelda_map.scan_states(tmp_path)
    assert states[(0, 0x37, False)] == ['0_37c_0', '0_37e_0', '0_37e_1', 'd4/0_37e_0']
    assert states[(4, 0x32, False)] == ['d4/4_32w_0'] and len(states) == 3
    assert zelda_map.pick_state(states[(0, 0x37, False)]) == '0_37e_0'
    assert zelda_map.parse_state_name('d4/4_60cave_2') == (4, 0x60, 'cave', 2)


def test_load_evals_merges_rooms_by_visits(models):
    merged, sources = zelda_map.load_evals_by_clips(models)
    stats = merged['both']
    assert {(s['specialist'], s['status'], s['mode']) for s in sources} == {
        ('d2-rope', 'detailed', 'argmax'), ('goriya', 'detailed', 'sample'), ('ow-zora', 'old format', 'sample'),
        ('ow-start', 'no eval', None), ('ow-octorok', 'no eval', None)}
    octorok = next(s for s in sources if s['specialist'] == 'ow-octorok')
    assert 'eval.json was measured by argmax' in octorok['file']      # the model samples: not its evaluation
    room = stats[(1, 0x44, False)]
    assert (room['visits'], room['episodes'], room['deaths']) == (40, 40, 3)
    assert room['death_rate'] == pytest.approx(3 / 40)
    assert room['health_change_per_visit'] == pytest.approx((-1.0 * 30 + 1.0 * 10) / 40)
    assert room['hits_per_visit'] == pytest.approx(60 / 40)
    assert room['drops_per_visit'] == pytest.approx({'heart': 15 / 40, 'rupee': 10 / 40})
    assert room['exits'] == {'n': 30, 'w': 7, 'death': 3}
    assert sorted(room['specialists']) == [('d2-rope', 10), ('goriya', 30)]
    assert stats[(1, 0x35, False)]['visits'] == 10
    # 2 detailed specialists, 50 room visits, 1 argmax + 1 sample, 3 without detailed evals
    assert re.findall(r'\d+', zelda_map.eval_status(sources)) == ['2', '50', '1', '1', '3']


def test_load_evals_run_id_needs_an_evaluation_in_the_act_mode(models):
    run2 = models / 'goriya' / 'run2'
    run2.mkdir()
    (run2 / 'ckpt.pt').write_bytes(b'other weights')
    (run2 / 'eval.json').write_text(json.dumps({'format': 2, 'act': 'argmax', 'visited_rooms': {}}))
    _, sources = zelda_map.load_evals_by_clips(models, 'run2')
    assert next(s for s in sources if s['specialist'] == 'goriya')['run'] == 'run1'    # run2 samples: no eval
    (run2 / 'eval-sample.json').write_text(json.dumps({'format': 2, 'act': 'sample', 'visited_rooms': {}}))
    _, sources = zelda_map.load_evals_by_clips(models, 'run2')
    goriya = next(s for s in sources if s['specialist'] == 'goriya')
    assert (goriya['run'], goriya['mode'], goriya['file']) == ('run2', 'sample', 'eval-sample.json')


def test_level_bounds_and_badges():
    assert zelda_map.level_bounds(0, [0x77]) == (0, 0, 16, 8)
    assert zelda_map.level_bounds(4, []) == (0, 0, 16, 8)
    assert zelda_map.level_bounds(1, [0x22, 0x74, 0x41]) == (1, 2, 4, 6)
    stats = {'hits_per_visit': 0.0, 'hearts_change_per_visit': -0.25, 'death_rate': 0.5, 'reach_rate': 0.25}
    assert zelda_map.overlay_badge('location', 0x7, None)[0] == '07'
    assert zelda_map.overlay_badge('none', 0x7, stats) is None
    assert zelda_map.overlay_badge('hits', 0x7, None) is None
    assert zelda_map.overlay_badge('hits', 0x7, stats) == ('0.00', zelda_map.C.ok)
    assert zelda_map.overlay_badge('hearts', 0x7, stats)[0] == '-0.25'
    assert zelda_map.overlay_badge('deaths', 0x7, stats) == ('50%', zelda_map.C.bad)
    assert zelda_map.overlay_badge('reach', 0x7, stats)[0] == '25%'
    assert zelda_map.overlay_badge('hits', 0x7, {**stats, 'hits_per_visit': None}) is None    # chains count none
    # The death rate's scale depends on the scope: 1% is half way to red on a full run, nearly blue in one room.
    one = {'death_rate': 0.01}
    assert zelda_map.overlay_badge('deaths', 0x7, one, 'full') == ('1.0%', zelda_map.C.warn)
    assert zelda_map.overlay_badge('deaths', 0x7, one, 'room')[1] != zelda_map.C.warn
    title, parts = zelda_map.scale_parts('deaths', 'full')
    assert 'full runs' in title and [text for text, _ in parts] == ['0', '1.0%', '≥ 2.0%']
    assert [color for _, color in parts] == [zelda_map.C.ok, zelda_map.C.warn, zelda_map.C.bad]
    assert 'isolated episodes' in zelda_map.scale_html('deaths', 'room') and zelda_map.scale_html('none', 'room') == ''


def test_window_overlays_selection_and_zoom(qapp, models, tmp_path):
    states = {(1, 0x44, False): ['1_44w_0'], (0, 0x77, True): ['0_77cave_0']}
    window = zelda_map.MapWindow(models, {}, states)
    window.resize(1200, 800)
    window.show()
    qapp.processEvents()
    assert window.status.text().startswith('No room-stats file')            # room only, from the evaluations
    assert 'Eval data: 2 specialists' in window.status.text()
    assert window.scope == 'room' and not window.scope_group.button(0).isEnabled()
    assert not any(button.isEnabled() for button in window.segment_group.buttons())

    window.tabs.setCurrentIndex(1)
    window.set_overlay('deaths')
    assert window.view.badges[0x44][0] == '7.5%'
    assert 0x35 in window.view.badges and 0x22 not in window.view.badges
    window.select_room(0x44)
    html = window.details.text()
    for text in ('1_44w_0 (pictured)', 'goriya', 'RedGoriya', 'Visits', '40', 'heart 0.38', 'n 30 · w 7 · death 3',
                 '-0.50 ♥'):
        assert text in html
    window.select_room(0x23)
    assert 'No detailed eval data for this room.' in window.details.text()

    view = window.view
    point = QPointF(300, 200)
    before = (point - view.origin) / view.scale
    view.zoom_at(point, 2.0)
    assert not view.fitted
    assert (point - view.origin) / view.scale == before          # the map point under the cursor stays put
    window.tabs.setCurrentIndex(0)
    assert window.view.fitted and 'Cave' in window.room_html(0, 0x77)
    window.tabs.setCurrentIndex(1)
    assert not window.view.fitted                                 # each level keeps its own zoom
    window.view.fit()
    assert window.view.room_at(window.view.tile_rect(0x44).center()) == 0x44
    window.close()


def scope_entry(runs, visits, death, reach=None, change=-0.5, steps=None, hits=None):
    """One room's entry in a room-stats scope (triforce.room_stats, format 1; the fields the map reads)."""
    return {'runs': runs, 'entered': visits, 'visits': visits, 'reach_rate': reach, 'deaths': round(death * runs),
            'death_rate': death, 'stuck': 0, 'stuck_rate': 0.0, 'hearts_change_per_visit': change,
            'hits_per_visit': hits, 'clips_per_visit': None, 'steps': steps}


def write_room_stats(path):
    """1:0x44 on the full run (two route steps), on To D1 and alone; 1:0x35 on the full run only."""
    full = {'1:0x44': scope_entry(100, 80, 0.03, reach=0.5, steps={
                '57': scope_entry(100, 50, 0.01, reach=0.5), '61': scope_entry(100, 30, 0.02, reach=0.3)}),
            '1:0x35': scope_entry(100, 20, 0.0, reach=0.2)}
    data = {'format': 1, 'tag': 'r99', 'created': '2026-11-13T12:00:00',      # from before the clips switch
            'scopes': {
                'full': {'runs': 100, 'arrival_rate': 0.1, 'start': 'start', 'cut': None, 'rooms': full},
                'to-d1': {'runs': 200, 'arrival_rate': 0.5, 'start': '0_3cc_0', 'cut': '0_37c_0',
                          'rooms': {'1:0x44': scope_entry(200, 150, 0.1, reach=0.75)}},
                'room': {'runs': None, 'team': 'chains', 'sources': [{'status': 'detailed'}],
                         'rooms': {'1:0x44': scope_entry(300, 300, 0.05, hits=1.5)}}}}
    path.write_text(json.dumps(data))
    return path


def test_window_scopes_from_a_room_stats_file(qapp, models, tmp_path):
    path = write_room_stats(tmp_path / 'r99-room-stats.json')
    window = zelda_map.MapWindow(models, {}, {}, stats_path=path)
    window.resize(1200, 800)
    window.show()
    qapp.processEvents()
    assert window.status.text().startswith('Room stats r99 (r99-room-stats.json')
    assert window.scope == 'full' and window.segment == 'to-d1'      # the first segment with data
    enabled = {name for name, button in zip(zelda_map.SEGMENTS, window.segment_group.buttons()) if button.isEnabled()}
    assert enabled == {'to-d1'}

    window.tabs.setCurrentIndex(1)
    window.set_overlay('deaths')
    assert window.view.badges[0x44] == ('3.0%', zelda_map.C.bad)     # 3% of full runs: past the full-run scale
    assert 'full runs' in window.legend.text() and '100 runs from start' in window.scope_info.text()
    window.set_segment('to-d1')
    assert window.scope == 'segment' and window.view.badges[0x44][0] == '10%'
    assert 0x35 not in window.view.badges and 0x35 in window.view.dimmed and 0x44 not in window.view.dimmed
    assert 'To D1 · 200 runs from 0_3cc_0, 50% reach 0_37c_0' == window.scope_info.text()
    window.set_scope('room')
    assert window.view.badges[0x44] == ('5.0%', zelda_map.badness_color(0.05 / zelda_map.DEATHS_BAD['room']))
    assert not window.view.dimmed and 'isolated episodes' in window.legend.text()
    window.set_overlay('hits')
    assert window.view.badges[0x44][0] == '1.50'
    window.set_overlay('reach')
    assert 0x44 not in window.view.badges                          # isolated episodes have no reach
    window.set_scope('full')
    assert window.view.badges[0x44][0] == '50%' and window.view.badges[0x35][0] == '20%'

    # The sidebar: one row per scope with data, a row per route step where a scope visits the room on several.
    rows = zelda_map.scope_rows(window.scopes, '1:0x44', window.current_scope())
    assert [(label, current) for label, _, _, current in rows] == [
        ('Full run', True), ('· #57', False), ('· #61', False), ('To D1', False), ('Room only', False)]
    window.select_room(0x44)
    html = window.details.text()
    for text in ('Room · by scope', 'Full run', '· #61', 'To D1', 'Room only', '3.0%', '75%', '-0.50',
                 'Room · eval of the promoted models'):
        assert text in html
    # A file from before the clips switch: its visits played on, there is nothing to show with it off.
    assert [button.isEnabled() for button in window.clips_group.buttons()] == [True, True, False]
    window.set_clips('on')
    assert window.view.badges[0x44][0] == '50%' and '100 runs with clips on from start' in window.scope_info.text()
    rows = zelda_map.scope_rows(window.scopes, '1:0x44', window.current_scope(), 'on')
    assert [(label, current) for label, _, _, current in rows][:3] == [
        ('Full run', True), ('· on #57', False), ('· on #61', False)]
    window.close()
    window = zelda_map.MapWindow(models, {}, {}, stats_path=path, clips='off')
    assert window.clips == 'both'
    window.close()


D4 = next(scope for scope in zelda_map.STAT_SCOPES if scope.name == 'd4')


def d4_run(plan, steps, clips, death=None):
    """A chain run of Dungeon 4 along the route's ``steps`` (round 22 records: step, switch, hits); ``death``: it
    dies on its last step."""
    records = [{'level': plan[i]['level'], 'room': f"0x{plan[i]['room']:02x}", 'in_cave': plan[i]['in_cave'],
                'specialist': plan[i]['specialist'], 'steps': 20, 'frames': 200, 'health_in': 5.0,
                'health_out': 4.5, 'outcome': 'exit-correct', 'hits': 1, 'step': i, 'clips_allowed': clips}
               for i in steps]
    if death:
        records[-1].update(outcome='death', health_out=0.0, hits=4)
    return {'route': 'd2first', 'result': 'failure' if death else 'success', 'clips': clips, 'rooms': records}


def write_clips_room_stats(path):
    """The d4 scope of 8 runs, from the room_stats code: 4 with the switch on (the ladder clip 4:0x31 -> 4:0x21; one
    dies in 4:0x21), 4 with it off (the plain way through 4:0x30 and 4:0x20; two die in 4:0x20)."""
    from triforce import room_stats  # pylint: disable=import-outside-toplevel
    plan = list(zelda_map.native_route_steps('d2first'))
    common = list(range(105, 115))
    clip_way, plain_way, rest = [115, 116], [117, 118, 119, 120], list(range(121, 125))
    runs = ([d4_run(plan, common + clip_way + rest, True) for _ in range(3)]
            + [d4_run(plan, common + clip_way, True, death=True)]
            + [d4_run(plan, common + plain_way + rest, False) for _ in range(2)]
            + [d4_run(plan, common + plain_way[:3], False, death=True) for _ in range(2)])
    data = {'format': ROOM_STATS_FORMAT, 'tag': 'r99', 'tags': ['r99', 'r99-noclips'], 'created': '2026-11-13',
            'scopes': {'d4': room_stats.chain_scope(runs, plan, D4)}}
    path.write_text(json.dumps(data))
    return path


def test_the_clips_selector_splits_rooms_and_the_route(qapp, models, tmp_path):
    path = write_clips_room_stats(tmp_path / 'r99-both-room-stats.json')
    window = zelda_map.MapWindow(models, {}, {}, stats_path=path, route='d2first')
    window.resize(1200, 800)
    window.show()
    qapp.processEvents()
    assert window.status.text().startswith('Room stats r99 + r99-noclips')
    assert (window.scope, window.segment, window.clips) == ('segment', 'd4', 'both')
    assert all(button.isEnabled() for button in window.clips_group.buttons())
    window.tabs.setCurrentIndex(4)
    window.set_overlay('deaths')
    assert window.view.badges[0x20][0] == '25%' and window.view.badges[0x21][0] == '12%'      # of 8 runs
    assert 'D4 alone · 8 runs from 4_71s_0, 62% reach the D4 triforce' == window.scope_info.text()
    # Both ways drawn, the clip way's step numbers marked.
    assert window.view.route_rooms[0x31] == [111, 115, 117] and 0x20 in window.view.route_rooms
    assert window.view.route_marks and window.view.route_ways[115] == 'clips'
    assert {(0x31, 0x21, 'clips'), (0x31, 0x30, 'plain'), (0x21, 0x11, None)} <= set(window.view.route_segments)
    assert 'clip way dashed' in window.legend.text()
    window.view.repaint()

    window.set_clips('off')
    assert window.clips_group.checkedId() == 2
    assert window.view.badges[0x20][0] == '50%' and 0x21 in window.view.badges and window.view.badges[0x21][0] == '0%'
    assert 'D4 alone · 4 runs with clips off from 4_71s_0, 50% reach' in window.scope_info.text()
    assert window.view.route_rooms[0x31] == [111, 117] and not window.view.route_marks
    assert (0x31, 0x21, 'clips') not in window.view.route_segments
    assert (0x31, 0x30, 'plain') in window.view.route_segments and window.legend.text().endswith('the plain way')
    window.set_overlay('hits')                                     # chain records count hits now
    assert window.view.badges[0x20][0] == '2.50'
    window.select_room(0x21)
    rows = zelda_map.scope_rows(window.scopes, '4:0x21', window.current_scope(), 'off')
    assert [(label, current) for label, _, _, current in rows] == [
        ('D4 alone', False), ('· clips on', False), ('· clips off', True)]
    html = window.details.text()
    assert '· clips off' in html and 'plain way of d4-island' in html and 'clips way of' not in html

    window.set_clips('on')
    window.set_overlay('deaths')
    assert 0x20 not in window.view.badges and 0x20 in window.view.dimmed and window.view.badges[0x21][0] == '25%'
    assert 0x30 in window.view.route_rooms and 0x20 not in window.view.route_rooms
    assert (0x31, 0x21, 'clips') in window.view.route_segments
    assert not any(way == 'plain' for _, _, way in window.view.route_segments)
    window.close()


def test_screenshot_with_the_clips_selector(qapp, models, tmp_path, monkeypatch):
    path = write_clips_room_stats(tmp_path / 'r99-both-room-stats.json')
    monkeypatch.setattr(zelda_map, 'render_screens', lambda *args: ({}, zelda_map.Counter()))
    for clips in ('off', 'both'):
        out = tmp_path / f'map-{clips}.png'
        assert zelda_map.main(['--models-dir', str(models), '--stats', str(path), '--route', 'd2first', '--clips',
                               clips, '--scope', 'd4', '--level', '4', '--overlay', 'deaths', '--select', '0x21',
                               '--screenshot', str(out)]) == 0
        assert out.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n' and out.stat().st_size > 10_000
    args = zelda_map.build_parser().parse_args(['--select', '0x31', '--clips', 'off'])
    assert (args.select, args.clips, args.level, args.overlay) == (0x31, 'off', 0, 'none')


def test_window_finds_the_newest_room_stats_file(qapp, models, tmp_path):
    window = zelda_map.MapWindow(models, {}, {}, stats_dir=tmp_path)
    assert window.status.text().startswith(f'No room-stats file (none in {tmp_path})') and window.scope == 'room'
    write_room_stats(tmp_path / 'r99-room-stats.json')
    window.reload()
    assert window.stats_file == tmp_path / 'r99-room-stats.json' and window.scope == 'room'
    window.set_scope('full')
    assert window.current_scope() == 'full'
    (tmp_path / 'broken-room-stats.json').write_text('{"format": 0}')
    window.reload()                                                   # the newest is unreadable: say why
    assert 'not a room-stats file' in window.status.text() and window.scope == 'room'
    window.close()
    args = zelda_map.build_parser().parse_args(['--stats', 'x.json'])
    assert args.stats == Path('x.json')


def test_route_plan_and_overlay(qapp, models):
    plan = zelda_map.route_plan('d2first')
    first = next(s for s in plan if (s['level'], s['room']) == (2, 0x7d))
    assert (first['hearts'], first['sword'], first['triforce']) == (3, 'wood', [])
    heart = next(s for s in plan if s['in_cave'] and s['room'] == 0x2c)
    assert (heart['hearts'], heart['triforce'], heart['waypoint']) == (4, [2], 'heart')
    aquamentus = next(s for s in plan if (s['level'], s['room']) == (1, 0x35))
    assert (aquamentus['hearts'], aquamentus['sword'], aquamentus['items']) == (5, 'white',
                                                                                ['bombs', 'bow', 'boomerang'])
    manhandla = next(s for s in plan if (s['level'], s['room']) == (3, 0x4d))
    assert (manhandla['hearts'], manhandla['triforce'], 'raft' in manhandla['items']) == (6, [2, 1], True)
    gleeok = next(s for s in plan if (s['level'], s['room']) == (4, 0x13))
    assert (gleeok['hearts'], gleeok['triforce'], 'ladder' in gleeok['items']) == (7, [2, 1, 3], True)
    assert plan[-1]['exit'] == 'triforce' and plan[-1]['hearts'] == 8
    skip = next(s for s in plan if (s['level'], s['room'], s['entry']) == (3, 0x69, 'N'))
    assert 'treasure skip' in zelda_map.step_text(skip)
    walks = zelda_map.route_segments(plan, 0)
    assert (0x1a, 0x0a, None) in walks and (0x0a, 0x1a, None) in walks
    assert (0x3c, 0x2c) not in {walk[:2] for walk in walks}               # walls between them: the route goes round
    assert zelda_map.route_caves(plan, 0) == {0x77, 0x3c, 0x2c, 0x0a, 0x37, 0x74, 0x45}
    rooms = zelda_map.route_rooms(plan, 0)
    assert len(rooms[0x2c]) == 3 and len(rooms[0x1a]) == 2
    assert '#' in zelda_map.step_text(heart) and 'waypoint heart' in zelda_map.step_text(heart)

    window = zelda_map.MapWindow(models, {}, {}, route='d2first')
    window.resize(1200, 800)
    window.show()
    qapp.processEvents()
    assert 'route d2first' in window.windowTitle()
    assert window.view.route_rooms[0x0a] and (0x0a, 0x1a, None) in window.view.route_segments
    window.view.repaint()
    html = window.room_html(0, 0x2c)
    assert 'Route d2first' in html and 'waypoint heart' in html
    window.tabs.setCurrentIndex(2)
    assert set(window.view.route_rooms) == {0x7d, 0x7e, 0x6e, 0x5e, 0x4e, 0x3e, 0x2e, 0x1e, 0x0e, 0x0d}
    window.close()
    assert zelda_map.build_parser().parse_args(['--route', 'd2first']).route == 'd2first'
    assert zelda_map.build_parser().parse_args([]).route is None


def test_dungeon_cellars_are_drawn_with_their_entrance(qapp, models):
    rooms = zelda_map.game_rooms()
    assert zelda_map.cellar_entrances(rooms) == {(1, 0x7f, True): 0x22, (3, 0x0f, True): 0x69, (4, 0x60, True): 0x32}
    states = {(1, 0x22, False): ['1_22e_0'], (1, 0x7f, True): ['1_7fcave_0']}
    window = zelda_map.MapWindow(models, {}, states)
    assert 0x7f not in window.level_rooms(1) and 0x0f not in window.level_rooms(3)
    assert zelda_map.level_bounds(1, window.level_rooms(1))[2] <= 6
    assert zelda_map.level_bounds(3, window.level_rooms(3))[2:] == (5, 6)
    html = window.room_html(1, 0x22)
    assert '1_7fcave_0' in html and 'cellar 0x7f' in html
    assert zelda_map.specialist_for((1, 0x7f, True)) == 'equipment-room' and 'equipment-room' in html
    # A room people have a name for says it (game.yaml `name:`), and only that room.
    assert '>Name<' not in html and 'Breakfast Room' in window.room_html(4, 0x32)
    plan = zelda_map.route_plan('d2first')
    if any((s['level'], s['room'], s['in_cave']) == (1, 0x7f, True) for s in plan):
        assert 0x7f not in zelda_map.route_rooms(plan, 1, window.cellars)
        assert 0x22 in zelda_map.route_rooms(plan, 1, window.cellars)
    window.close()


def test_render_screens_caches_playfields(tmp_path):
    states_dir = tmp_path / 'states'
    states_dir.mkdir()
    shutil.copy(require_state('1_73s_0'), states_dir)
    states = zelda_map.scan_states(states_dir)
    screens, rendered = zelda_map.render_screens(states, tmp_path / 'screens', states_dir)
    assert rendered == {1: 1}
    path = screens[(1, 0x73, False)]
    from PIL import Image  # pylint: disable=import-outside-toplevel
    with Image.open(path) as image:
        assert image.size == (zelda_map.TILE_W, zelda_map.TILE_H)
    _, rendered = zelda_map.render_screens(states, tmp_path / 'screens', states_dir)
    assert not rendered
    _, rendered = zelda_map.render_screens(states, tmp_path / 'screens', states_dir, rerender=True)
    assert rendered == {1: 1}


def test_overworld_rooms_without_a_savestate_are_walked_into(tmp_path, monkeypatch):
    states_dir = tmp_path / 'states'
    states_dir.mkdir()
    for name in ('start', '0_77c_0', '0_1cs_0'):
        shutil.copy(require_state(name), states_dir)
    states = zelda_map.scan_states(states_dir)
    assert len(zelda_map.walk_in_rooms(states)) == 126
    # Two of them (0x0b lies above the Lost Hills, 0x1b, its only opening: entered with the maze's fourth step).
    walked = ((0, 0x0b, False), (0, 0x1f, False))
    monkeypatch.setattr(zelda_map, 'walk_in_rooms', lambda states: [key for key in walked if key not in states])
    screens, rendered = zelda_map.render_screens(states, tmp_path / 'screens', states_dir)
    assert rendered == {0: 4}
    from PIL import Image  # pylint: disable=import-outside-toplevel
    for key in ((0, 0x0b, False), (0, 0x1f, False)):
        with Image.open(screens[key]) as image:
            assert image.size == (zelda_map.TILE_W, zelda_map.TILE_H)
        assert zelda_map.screen_source(screens[key]) == 'start walk-in'
    assert zelda_map.screen_source(screens[(0, 0x77, False)]) == '0_77c_0'
    game = zelda_map.Game(render=True, states_dir=str(states_dir))
    zelda_map.walk_in_playfield(game, states_dir / 'start.state', 0x0b)
    link_y = game.read(0x84)                    # he came in by the S edge, where the scroll leaves him at y $DD
    assert (game.read(zelda_map.ROOM_ID), game.read(zelda_map.MODE), link_y) == (0x0b, 5, 0xDD)
    _, rendered = zelda_map.render_screens(states, tmp_path / 'screens', states_dir)
    assert not rendered
    # A savestate for a walked-in room replaces its picture, even when the file is older than the PNG (0x1c's state
    # file stands in for one of 0x1f: only the name and the date matter here).
    shutil.copy(require_state('0_1cs_0'), states_dir / '0_1fs_0.state')
    os.utime(states_dir / '0_1fs_0.state', (0, 0))
    _, rendered = zelda_map.render_screens(zelda_map.scan_states(states_dir), tmp_path / 'screens', states_dir)
    assert rendered == {0: 1}
    assert zelda_map.screen_source(screens[(0, 0x1f, False)]) == '0_1fs_0'


def test_a_failed_rerender_keeps_the_cache(tmp_path, monkeypatch):
    states_dir = tmp_path / 'states'
    states_dir.mkdir()
    shutil.copy(require_state('1_73s_0'), states_dir)
    states = zelda_map.scan_states(states_dir)
    screens, _ = zelda_map.render_screens(states, tmp_path / 'screens', states_dir)
    path = screens[(1, 0x73, False)]
    stray = tmp_path / 'screens' / zelda_map.screen_name((1, 0x22, False))      # a room whose savestate went
    shutil.copy(path, stray)

    def no_rom(*_args, **_kwargs):
        raise FileNotFoundError('ROM not found')
    monkeypatch.setattr(zelda_map, 'Game', no_rom)
    with pytest.raises(FileNotFoundError):
        zelda_map.render_screens(states, tmp_path / 'screens', states_dir, rerender=True)
    assert path.is_file() and stray.is_file()
    monkeypatch.undo()
    zelda_map.render_screens(states, tmp_path / 'screens', states_dir, rerender=True)
    assert path.is_file() and not stray.exists()
