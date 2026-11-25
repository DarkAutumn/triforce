"""Round 42: harvested states the route can be in. A handoff harvested on another order of the route (2.0.0.0's
Dungeon 5 before Dungeon 2: no Dungeon 2 piece) is found by the audit, refused by a training reset and never written by
a chain's harvest; the training pool draws a harvested state for the seed draw's room entry."""

from pathlib import Path

import pytest

from triforce import _native as native, harvest_audit
from triforce.lab.collect import task_yaml
from triforce.specialist import load_specialist

from .conftest import require_state

TRIFORCE = 0x671        # InvTriforce: bit (level - 1)


def patched(src: Path, out: Path, writes: dict) -> Path:
    """``src`` with RAM bytes written (a raw machine restore: no load-time changes)."""
    game = native.Game(seed=0)
    game.restore(native.Snapshot.from_state_file(src.read_bytes()))
    for addr, value in writes.items():
        game.write(addr, value)
    out.write_bytes(game.snapshot().state_file())
    return out


@pytest.fixture
def snapshot(tmp_path):
    """A harvested snapshot with darknut's 5:0x64 from E twice: as the route brings it, and without Dungeon 2's piece."""
    seed = require_state('d3first_5_64e_0')
    folder = tmp_path / 'harvested' / 'darknut'
    folder.mkdir(parents=True)
    (folder / 'd3first_5_64e_h1_3.state').write_bytes(seed.read_bytes())
    game = native.Game(seed=0)
    state = game.restore(native.Snapshot.from_state_file(seed.read_bytes()))
    assert state.var('triforce') == 0b1111
    patched(seed, folder / 'd3first_5_64e_h2_3.state', {TRIFORCE: 0b1101})
    return tmp_path / 'harvested'


def test_start_on_route_reads_links_progress(snapshot):
    game = native.Game(seed=0)
    for name, on in (('d3first_5_64e_h1_3', True), ('d3first_5_64e_h2_3', False)):
        path = snapshot / 'darknut' / f'{name}.state'
        state = game.restore(native.Snapshot.from_state_file(path.read_bytes()))
        assert native.start_on_route('d3first', name, state) is on
    assert native.start_on_route('d3first', 'not-a-state-name', state) is False


def test_the_audit_finds_the_other_orders_state(snapshot, tmp_path):
    rows = harvest_audit.audit_family(('walkers', 'darknut@d3first', str(snapshot), 'd3first'))
    by_file = {r['file']: r for r in rows}
    assert harvest_audit.invalid(by_file['d3first_5_64e_h1_3.state']) == []
    assert harvest_audit.invalid(by_file['d3first_5_64e_h2_3.state']) == ['PROGRESS']
    out = tmp_path / 'invalid.txt'
    assert harvest_audit.write_invalid(rows, out) == 1
    assert out.read_text() == 'darknut/d3first_5_64e_h2_3.state PROGRESS\n'


def test_a_training_reset_refuses_a_harvested_state_off_the_route(snapshot):
    """The seed draw (5:0x64 from E) brings its entry's harvested states at fraction 1.0: the route's is played, the
    other order's refused with the audit to run."""
    config = load_specialist(task_yaml('darknut@d3first')).replace(
        harvested_states=str(snapshot / 'darknut'), harvested_fraction=1.0, start_states=['5_64e_[0-9]'])
    stale_path = snapshot / 'darknut' / 'd3first_5_64e_h2_3.state'
    stale = stale_path.read_bytes()
    stale_path.unlink()
    env = native.Env(config, seed=0)
    for _ in range(4):
        _, info = env.reset()
        assert info['start_state'].endswith('d3first_5_64e_h1_3.state'), info['start_state']
    (snapshot / 'darknut' / 'd3first_5_64e_h1_3.state').write_bytes(stale)
    with pytest.raises(native.StartStateError, match='harvest_audit'):
        env.reset()
