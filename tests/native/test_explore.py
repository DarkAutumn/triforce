"""`Explorer`, `AcceptSpec` and `explore_states` (small budgets: the search itself is tested in Rust)."""

import json

from triforce import _native as native


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
