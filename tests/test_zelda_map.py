"""zelda-map.py: savestate names, merging evals' per-room stats, the window (Qt offscreen), screenshot rendering."""

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
    stats, sources = zelda_map.load_evals(models)
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
    _, sources = zelda_map.load_evals(models, 'run2')
    assert next(s for s in sources if s['specialist'] == 'goriya')['run'] == 'run1'    # run2 samples: no eval
    (run2 / 'eval-sample.json').write_text(json.dumps({'format': 2, 'act': 'sample', 'visited_rooms': {}}))
    _, sources = zelda_map.load_evals(models, 'run2')
    goriya = next(s for s in sources if s['specialist'] == 'goriya')
    assert (goriya['run'], goriya['mode'], goriya['file']) == ('run2', 'sample', 'eval-sample.json')


def test_level_bounds_and_badges():
    assert zelda_map.level_bounds(0, [0x77]) == (0, 0, 16, 8)
    assert zelda_map.level_bounds(4, []) == (0, 0, 16, 8)
    assert zelda_map.level_bounds(1, [0x22, 0x74, 0x41]) == (1, 2, 4, 6)
    stats = {'hits_per_visit': 0.0, 'health_change_per_visit': -0.25, 'death_rate': 0.5}
    assert zelda_map.overlay_badge('location', 0x7, None)[0] == '07'
    assert zelda_map.overlay_badge('none', 0x7, stats) is None
    assert zelda_map.overlay_badge('hits', 0x7, None) is None
    assert zelda_map.overlay_badge('hits', 0x7, stats) == ('0.00', zelda_map.C.ok)
    assert zelda_map.overlay_badge('hearts', 0x7, stats)[0] == '-0.25'
    assert zelda_map.overlay_badge('deaths', 0x7, stats) == ('50%', zelda_map.C.bad)


def test_window_overlays_selection_and_zoom(qapp, models, tmp_path):
    states = {(1, 0x44, False): ['1_44w_0'], (0, 0x77, True): ['0_77cave_0']}
    window = zelda_map.MapWindow(models, {}, states)
    window.resize(1200, 800)
    window.show()
    qapp.processEvents()
    assert window.status.text().startswith('Eval data: 2 specialists')

    window.tabs.setCurrentIndex(1)
    window.set_overlay('deaths')
    assert window.view.badges[0x44][0] == '8%'
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
    assert plan[-1]['exit'] == 'triforce' and plan[-1]['hearts'] == 7
    skip = next(s for s in plan if (s['level'], s['room'], s['entry']) == (3, 0x69, 'N'))
    assert 'treasure skip' in zelda_map.step_text(skip)
    assert (0x1a, 0x0a) in zelda_map.route_segments(plan, 0) and (0x0a, 0x1a) in zelda_map.route_segments(plan, 0)
    assert (0x3c, 0x2c) not in zelda_map.route_segments(plan, 0)          # walls between them: the route goes round
    assert zelda_map.route_caves(plan, 0) == {0x77, 0x3c, 0x2c, 0x0a, 0x37, 0x74}
    rooms = zelda_map.route_rooms(plan, 0)
    assert len(rooms[0x2c]) == 3 and len(rooms[0x1a]) == 2
    assert '#' in zelda_map.step_text(heart) and 'waypoint heart' in zelda_map.step_text(heart)

    window = zelda_map.MapWindow(models, {}, {}, route='d2first')
    window.resize(1200, 800)
    window.show()
    qapp.processEvents()
    assert 'route d2first' in window.windowTitle()
    assert window.view.route_rooms[0x0a] and (0x0a, 0x1a) in window.view.route_segments
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
