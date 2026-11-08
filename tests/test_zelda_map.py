"""zelda-map.py: savestate names, merging evals' per-room stats, the window (Qt offscreen), screenshot rendering."""

import importlib.util
import json
import os
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


def write_eval(models, specialist, data):
    run = models / specialist / 'run1'
    run.mkdir(parents=True)
    (models / specialist / 'promoted').write_text('run1\n')
    if data is not None:
        (run / 'eval.json').write_text(json.dumps(data))


@pytest.fixture
def models(tmp_path):
    """goriya and d2-rope both visited 1:0x44; old has an old-format eval; none has no eval.json."""
    models = tmp_path / 'models'
    write_eval(models, 'goriya', {'format': 2, 'visited_rooms': {
        '1:0x44': room_stats('goriya', 1, 0x44, 30, deaths=3, change=-1.0, hits=2.0, drops={'heart': 0.5},
                             exits={'n': 20, 'w': 7, 'death': 3}),
        '1:0x35': room_stats('goriya', 1, 0x35, 10)}})
    write_eval(models, 'd2-rope', {'format': 2, 'visited_rooms': {
        '1:0x44': room_stats('d2-rope', 1, 0x44, 10, change=1.0, hits=0.0, drops={'rupee': 1.0},
                             exits={'n': 10})}})
    write_eval(models, 'old', {'specialist': 'old', 'success_rate': 0.9})
    write_eval(models, 'none', None)
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


def test_load_evals_merges_rooms_by_visits(models):
    stats, sources = zelda_map.load_evals(models)
    assert {(s['specialist'], s['status']) for s in sources} == {
        ('d2-rope', 'detailed'), ('goriya', 'detailed'), ('old', 'old format'), ('none', 'no eval.json')}
    room = stats[(1, 0x44, False)]
    assert (room['visits'], room['episodes'], room['deaths']) == (40, 40, 3)
    assert room['death_rate'] == pytest.approx(3 / 40)
    assert room['health_change_per_visit'] == pytest.approx((-1.0 * 30 + 1.0 * 10) / 40)
    assert room['hits_per_visit'] == pytest.approx(60 / 40)
    assert room['drops_per_visit'] == pytest.approx({'heart': 15 / 40, 'rupee': 10 / 40})
    assert room['exits'] == {'n': 30, 'w': 7, 'death': 3}
    assert sorted(room['specialists']) == [('d2-rope', 10), ('goriya', 30)]
    assert stats[(1, 0x35, False)]['visits'] == 10
    assert zelda_map.eval_status(sources) == 'Eval data: 2 specialists (50 room visits) · 2 without detailed evals'


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
