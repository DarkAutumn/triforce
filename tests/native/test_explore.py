"""`Explorer`, `AcceptSpec` and `explore_states` (small budgets: the search itself is tested in Rust)."""

import json

import pytest

from triforce import _native as native
from triforce.explore import track_dir

from ..conftest import REPO_STATES_DIR, require_rom


def test_explorer_accepts_entering_a_room(states_dir, tmp_path):
    explorer = native.Explorer('ow-start', seed=3, states_dir=str(states_dir))
    accept = native.AcceptSpec((0, 0x67, False))
    assert accept.target == (0, 0x67, False) and accept.min_keys == 0 and accept.item_taken_room is None
    result = explorer.explore('start', iterations=60, explore_steps=25, patience=5, accept=accept)
    assert result.start_state == 'start' and result.iterations >= 1 and result.cells >= 1
    assert result.success, result
    assert all(len(a) == 3 for a in result.actions)
    info, pre_entry = explorer.replay(result, capture_pre_entry=True)
    assert accept.accepts(explorer.state)
    assert info['room'] == [0, 0x67, False]
    assert pre_entry is not None
    state = native.Game().load(pre_entry)
    assert state.full_location == (0, 0x67, False)

    written = explorer.write(result, str(tmp_path / 'track'))
    assert not written                       # the specialist's own outcome (exit-correct) differs from accept
    meta = json.loads((tmp_path / 'track' / 'trajectory.json').read_text())
    assert meta['start_state'] == 'start' and meta['steps'] == len(result.actions)
    assert (tmp_path / 'track' / '0000.state').read_bytes()[:8] == b'TFZSTATE'


def test_explore_states_is_independent_of_parallelism(states_dir, tmp_path):
    def run(parallel):
        out = tmp_path / f'p{parallel}'
        results = native.explore_states('ow-start', str(out), states=['start', 'start'], iterations=3,
                                        explore_steps=5, patience=1, parallel=parallel, seed=2,
                                        states_dir=str(states_dir))
        return [(r.start_state, r.success, r.actions, r.cells, r.env_steps, replayed) for r, replayed in results]
    assert run(1) == run(2)


def test_an_unsolved_search_says_what_it_reached():
    """Round 40 (Tracks41): the failed episodes by ending, the living enemies at the reset and the fewest reached,
    the longest path and the best closeness, for the explorer's `reached:` line."""
    require_rom()
    explorer = native.Explorer('zol', seed=0, states_dir=str(REPO_STATES_DIR))
    result = explorer.explore('d3first_3_4bs_0', iterations=8, explore_steps=10, patience=1)
    assert not result.success
    assert all(isinstance(e, str) and n > 0 for e, n in result.failures)
    assert sum(n for _, n in result.failures) <= result.iterations
    assert result.enemies[0] == 3 and 0 <= result.enemies[1] <= 3          # 3:0x4b's three Zols
    assert 1 <= result.longest <= result.iterations * 10
    assert 0.0 <= result.closest <= 1.0


def test_a_track_says_the_layout_it_was_explored_in(tmp_path):
    """Round 45 (Tracks45): an explorer's resets serve the layout it is given under the reset rule (3:0x7c from S:
    the route's spawn cycle 7, the door's one other reachable value 8), and its track's `trajectory.json` says what the
    room was entered with; a cycle the door cannot hold is refused, and its track would be named `~c<N>`."""
    require_rom()
    for layout, asked, cycle, planned in (('route', 'route', 7, True), (8, 'cycle', 8, False)):
        explorer = native.Explorer('zol', seed=0, states_dir=str(REPO_STATES_DIR), layout=layout)
        result = explorer.explore('d3first_3_7cs_0', iterations=1, explore_steps=5, patience=1)
        explorer.write(result, str(tmp_path / str(layout)))
        meta = json.loads((tmp_path / str(layout) / 'trajectory.json').read_text())
        entry = meta['layout']['entry']
        assert meta['layout']['asked'] == asked and isinstance(meta['layout']['rng_phase'], int)
        assert (entry['spawn_cycle'], entry['planned'], entry['reachable'], entry['source']) == \
            (cycle, planned, True, 'rule')
        assert entry['door'] == [3, 0x7c, 8, False] and entry['chase_flag'] in (0, 1)
    refused = native.Explorer('zol', seed=0, states_dir=str(REPO_STATES_DIR), layout=3)
    with pytest.raises(Exception, match="not a layout its door's reset rule holds"):
        refused.explore('d3first_3_7cs_0', iterations=1, explore_steps=5, patience=1)
    assert track_dir(tmp_path, 'zol', 'd3first_3_7cs_0', 0, False, cycle=8).name == 'd3first_3_7cs_0~c8'
