"""``scripts/experiments/r40_hearts.py``: a stored state holding Link's hearts as the pre-2.0.0.16 writer wrote them
(a full heart beside an empty partial, no refill running) is written again with ``hearts_encoding`` at the same half
hearts; every other state of the copy is the source's own file, and ``verify`` proves both."""

import json
import os
import sys
from pathlib import Path

from triforce import _native

from .conftest import REPO_STATES_DIR, require_state

ROOT = Path(__file__).resolve().parents[1]
FILLING = 0x63


def script():
    sys.path.insert(0, str(ROOT / 'scripts' / 'experiments'))
    import r40_hearts  # pylint: disable=import-outside-toplevel,import-error
    return r40_hearts


def write(game, data: bytes, path: Path, hearts: int, partial: int, filling: int = 0):
    game.restore(_native.Snapshot.from_state_file(data))
    game.set_value('hearts_and_containers', hearts)
    game.set_value('partial_hearts', partial)
    if filling:
        game.write(FILLING, filling)
    path.parent.mkdir(parents=True, exist_ok=True)
    game.save_state_file(str(path))


def test_the_old_bytes_are_rewritten_at_the_same_hearts_and_nothing_else(tmp_path, capsys):
    mod = script()
    data = require_state('d3first_3_4ae_0').read_bytes()       # 3 containers
    game = _native.Game(states_dir=str(REPO_STATES_DIR))
    src, dst, audit = tmp_path / 'src', tmp_path / 'dst', tmp_path / 'audit.jsonl'
    write(game, data, src / 'zol' / 'old2.state', 0x22, 0x00)            # the old 2 hearts of 3
    write(game, data, src / 'zol' / 'old1.state', 0x21, 0x00)            # the old 1 heart
    write(game, data, src / 'zol' / 'played.state', 0x21, 0xFE)          # a borrow's partial
    write(game, data, src / 'zol' / 'half.state', 0x21, 0x7F)            # 1.5 hearts, both writers'
    write(game, data, src / 'zol' / 'filling.state', 0x21, 0x00, 1)      # a refill's completed heart
    write(game, data, src / 'zol' / 'dead.state', 0x20, 0x00)            # no full heart: not the old writer's

    assert mod.audit([src], str(audit), set()) == 0
    recs = {Path(r['path']).stem: r for r in map(json.loads, audit.read_text().splitlines())}
    assert {name for name, rec in recs.items() if mod.old_bytes(rec)} == {'old2', 'old1'}
    assert mod.repair(src, dst, [str(audit)]) == 0
    assert mod.verify(src, dst) == 0
    out = capsys.readouterr().out
    assert '6 files, 4 byte-identical' in out and '2 differ' in out

    after = {}
    for name in recs:
        state = game.restore(_native.Snapshot.from_state_file((dst / 'zol' / f'{name}.state').read_bytes()))
        after[name] = (state.read(0x66F), state.read(0x670), round(2 * state.health))
    assert after['old2'] == (0x21, 0xFF, 4) and recs['old2']['half_hearts'] == 4
    assert after['old1'] == (0x20, 0xFF, 2) and recs['old1']['half_hearts'] == 2
    for name in ('played', 'half', 'filling', 'dead'):          # the source's own files, linked
        assert os.stat(dst / 'zol' / f'{name}.state').st_ino == os.stat(src / 'zol' / f'{name}.state').st_ino
    for name in ('old2', 'old1'):                                # new files: the source is never written through
        assert os.stat(dst / 'zol' / f'{name}.state').st_ino != os.stat(src / 'zol' / f'{name}.state').st_ino
        assert (src / 'zol' / f'{name}.state').read_bytes() != (dst / 'zol' / f'{name}.state').read_bytes()
