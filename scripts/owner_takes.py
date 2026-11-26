"""The owner's takes: the replays he plays on ``emu-zelda.py`` and names for the project, kept in git and in the
archive, with an index (the owner, Nov 25: "Place these somewhere durable (along with the other replays I make for
you, this may be an ongoing thing)"; and the same day: "not all save states I play need saving, only the ones I
tell you about. I might save a few by accident, I might save some but make better ones and not use the others").

    triforce job run --agent A --task T --subtask S --desc '...' --express --timeout 10m -- \\
        .venv/bin/python scripts/owner_takes.py add PATH|NAME... [--from DIR] [--no-archive] [--entry-rule PATH]
    ... scripts/owner_takes.py sync [--no-archive] [--entry-rule PATH]
    ... scripts/owner_takes.py index [--entry-rule PATH]

``add`` files only the takes it is given: a path, or a file name (with or without ``.tfreplay``) matched under
``--from`` (default ``~/triforce-review``, read only: nothing there is moved, renamed or deleted).  Each must be one
of his (META ``kind`` ``human``, what ``emu-zelda.py`` records), name one take only (copies of one file by SHA-1
count once) and verify (``tf replay verify``: ``_native.replay_verify``); if any of them does not, nothing is filed
and the command exits 1.  A take not yet kept (by SHA-1, against every file under ``replays/owner/``) is copied under
its own name into ``replays/owner/<room>/`` (a room take: ``5x47``, ``ox1f`` by META ``human.room``) or
``replays/owner/game/`` (a whole game from power on: ``start_<name>...``); then every take kept there is archived
(``scripts/archive_replays.py --kind owner --keep-names``: ``/backup/triforce/replays/<version>/owner/``; what the
archive holds is skipped) and the index is rewritten.  Nothing he did not name is filed.

``sync`` files nothing new: it archives what ``replays/owner/`` holds (a missed archive catches up) and rewrites the
index.  ``index`` (also the end of ``add`` and ``sync``) reads every take under ``replays/owner/`` and writes
``replays/owner/INDEX.md``, one row a take (room, entry, date, seed, frames, how it ended, ``$524`` and whether it is
the route's layout for that door, the hearts and items he came in with, the cheats and overrides, the replay
version) and, for a whole game, the rooms it passes with the frame Link is first in play in each.  What each take
holds is cached by SHA-1 in ``replays/owner/index.json``, so a whole game is read once.

The route's layout for a door is ``triforce/entry_rule.<namespace>.json``'s ``planned`` (round 43, on r40-int and
after it); a checkout without it takes ``--entry-rule PATH``, else the column says "layout unknown".
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from triforce import _native, jobs  # noqa: E402  pylint: disable=wrong-import-position
from triforce.solved_rooms import read_passes, room_text  # noqa: E402  pylint: disable=wrong-import-position

OWNER = ROOT / 'replays' / 'owner'
INDEX_MD = OWNER / 'INDEX.md'
INDEX_JSON = OWNER / 'index.json'
SOURCE = Path.home() / 'triforce-review'
ENTRY_RULE = ROOT / 'triforce' / 'entry_rule.d3first.json'
# The side Link comes in by -> his facing on the frame before the entry (zelda::room_entry: up 8, down 4, left 2,
# right 1); a stair or cave entry is 'inside'.
FACING = {'S': 8, 'N': 4, 'E': 2, 'W': 1}
ITEM_BYTES = {'sword': 0x657, 'bombs': 0x658, 'arrows': 0x659, 'bow': 0x65A, 'candle': 0x65B, 'recorder': 0x65C,
              'food': 0x65D, 'potion': 0x65E, 'wand': 0x65F, 'raft': 0x660, 'book': 0x661, 'ring': 0x662,
              'ladder': 0x663, 'magic key': 0x664, 'bracelet': 0x665, 'letter': 0x666, 'rupees': 0x66D,
              'keys': 0x66E, 'boomerang': 0x674, 'magic boomerang': 0x675, 'magic shield': 0x676}
COUNTS = ('bombs', 'rupees', 'keys', 'potion')
GRADES = {'sword': ('wood sword', 'white sword', 'magic sword'), 'ring': ('blue ring', 'red ring'),
          'candle': ('blue candle', 'red candle'), 'arrows': ('arrows', 'silver arrows')}
NAME_DATE = re.compile(r'_(\d{6})-(\d{6})(?:-\d+)?$')


def sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def is_take(data: bytes) -> bool:
    try:
        return (_native.Replay(data).meta or {}).get('kind') == 'human'
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def destination(data: bytes, name: str) -> Path:
    human = (_native.Replay(data).meta or {}).get('human') or {}
    if human.get('room'):
        level, room = (int(v, 0) for v in human['room'].split(':'))
        return OWNER / room_text(level, room) / name
    return OWNER / 'game' / name


def entry_rule(path) -> dict:
    """(level, room, facing, inside) -> the route's planned `$524`, from the entry rule file (empty without one)."""
    path = Path(path)
    if not path.is_file():
        return {}
    rule = json.loads(path.read_text())
    return {(d['level'], d['room'], d['facing'], d['inside']): d['planned'] for d in rule['doors']}


def planned(rule: dict, level: int, room: int, entry: str):
    if entry == 'cave':
        found = [v for (lv, rm, _f, inside), v in rule.items() if (lv, rm, inside) == (level, room, True)]
        return found[0] if len(found) == 1 else None
    return rule.get((level, room, FACING.get(entry), False))


def came_with(data: bytes) -> dict:
    """Hearts and items on the first frame the player has control (the CONTROL event; a whole game: its
    hand-over)."""
    replay = _native.Replay(data)
    control = next((e['frame'] for e in replay.events() if e.get('type') == 'control'), 0)
    player = _native.ReplayPlayer(data)
    player.seek(control)
    ram = player.ram()
    state = _native.Game().load(player.save_state(), randomize_rng=False, controllable=True)
    items = {k: int(ram[a]) for k, a in ITEM_BYTES.items() if int(ram[a])}
    return {'hearts': f"{state.health:g}/{state.max_health}", 'items': items}


def describe(data: bytes, name: str) -> dict:
    """What the index says of one take (cached by SHA-1)."""
    replay = _native.Replay(data)
    meta = replay.meta or {}
    human = meta.get('human') or {}
    ending = meta.get('ending') or {}
    found = NAME_DATE.search(Path(name).stem)
    out = {'name': name, 'version': replay.version, 'triforce': meta.get('triforce'), 'seed': meta.get('seed'),
           'frames': replay.frames, 'date': (f"20{found.group(1)[:2]}-{found.group(1)[2:4]}-{found.group(1)[4:]} "
                                             f"{found.group(2)[:2]}:{found.group(2)[2:4]}" if found else None),
           'ending': ' / '.join(str(v) for v in (ending.get('kind'), ending.get('result'), ending.get('reason'))
                                if v),
           'cheats': human.get('cheats'), 'overrides': human.get('overrides') or {},
           'came_with': came_with(data)}
    if human.get('room'):
        out.update({'kind': 'room', 'room': human['room'], 'entry': human.get('entry'), 'state': human.get('state')})
        level, room = (int(v, 0) for v in human['room'].split(':'))
        passes = [p for p in read_passes(data, name) if (p.level, p.room) == (level, room)]
        out['layout'] = passes[0].layout if passes else None
        out['chase'] = passes[0].chase if passes else None
    else:
        out.update({'kind': 'game', 'name_typed': human.get('name'), 'reset': human.get('reset'),
                    'hand_over': human.get('hand_over'),
                    'rooms': [{'room': room_text(p.level, p.room), 'entry': p.entry, 'frame': p.start,
                               'layout': p.layout, 'exit': p.exit} for p in read_passes(data, name)]})
    return out


def write_index(rule: dict, rule_name: str) -> None:
    cache = json.loads(INDEX_JSON.read_text()) if INDEX_JSON.is_file() else {}
    takes, fresh = [], {}
    for path in sorted(OWNER.rglob('*.tfreplay')):
        data = path.read_bytes()
        key = sha1(data)
        entry = cache.get(key) or describe(data, path.name)
        entry['path'] = str(path.relative_to(ROOT))
        fresh[key] = entry
        takes.append(entry)
    INDEX_JSON.write_text(json.dumps(fresh, indent=1, sort_keys=True) + '\n')
    lines = ['# The owner\'s takes', '',
             'The replays the owner played on `emu-zelda.py` and named for the project, kept as he saved them (his',
             'file names). Written by `scripts/owner_takes.py` (`add NAME...` files the takes he names; do not edit',
             'by hand). `$524`: the spawn cycle the room was entered with (where its monsters stand); "route" when it',
             f'is the layout the route brings to that door (`{rule_name}`).',
             'Solved rooms: `triforce/solved_rooms.yaml`.',
             '', '## One room', '',
             '| take | room | entry | date | seed | frames | ended | `$524` | chase | came with | cheats | v |',
             '|---|---|---|---|---:|---:|---|---|---:|---|---|---:|']
    for t in (t for t in takes if t['kind'] == 'room'):
        level, room = (int(v, 0) for v in t['room'].split(':'))
        want = planned(rule, level, room, t['entry']) if rule else None
        layout = ('-' if t.get('layout') is None else
                  f"{t['layout']} ({'route' if t['layout'] == want else f'route {want}'})" if want is not None else
                  f"{t['layout']} (layout unknown)")
        lines.append(f"| [{t['name']}]({Path(t['path']).relative_to('replays/owner')}) | "
                     f"{room_text(level, room)} | {t['entry']} | {t['date'] or '-'} | {t['seed']} | {t['frames']} | "
                     f"{t['ending']} | {layout} | {t.get('chase', '-')} | {_came_with(t)} | {_cheats(t)} | "
                     f"{t['version']} |")
    games = [t for t in takes if t['kind'] == 'game']
    lines += ['', '## Whole games (from power on)', '']
    if not games:
        lines.append('None yet.')
    for t in games:
        lines += [f"### [{t['name']}]({Path(t['path']).relative_to('replays/owner')})", '',
                  f"{t['date'] or '-'}, {t['frames']} frames, ended {t['ending']}; name {t.get('name_typed')}, "
                  f"reset {t.get('reset')}; cheats {_cheats(t)}; came with {_came_with(t)}; replay version "
                  f"{t['version']}.", '', '| room | entry | first frame in play | `$524` | left by |',
                  '|---|---|---:|---|---|']
        for r in t['rooms']:
            lines.append(f"| {r['room']} | {r['entry']} | {r['frame']} | {r['layout']} | {r['exit']} |")
        lines.append('')
    INDEX_MD.write_text('\n'.join(lines).rstrip() + '\n')


def _came_with(t: dict) -> str:
    c = t['came_with']
    items = ', '.join(GRADES[k][v - 1] if k in GRADES and 0 < v <= len(GRADES[k]) else
                      f"{k} {v}" if k in COUNTS or v > 1 else k for k, v in c['items'].items())
    return f"{c['hearts']} hearts; {items}"


def _cheats(t: dict) -> str:
    cheats = t.get('cheats') or {}
    parts = (['invincible'] if cheats.get('invincible') else []) + [f"infinite {i}" for i in cheats.get('infinite')
                                                                    or []]
    parts += [f"{k} {v}" for k, v in (t.get('overrides') or {}).items()]
    return ', '.join(parts) or 'none'


def find(names: list, source: Path) -> tuple[list, list]:
    """The takes ``names`` name: (path, bytes) each, and the problems (not found, more than one take, not his)."""
    found, problems = [], []
    files = None
    for name in names:
        path = Path(name).expanduser()
        if path.is_file():
            matches = [path]
        else:
            if files is None:
                files = sorted(source.rglob('*.tfreplay'))
            stem = name[:-len('.tfreplay')] if name.endswith('.tfreplay') else name
            matches = [p for p in files if p.stem == stem]
        takes = {}
        for match in matches:
            takes.setdefault(sha1(match.read_bytes()), match)
        if not takes:
            problems.append(f"{name}: no such take under {source}")
        elif len(takes) > 1:
            problems.append(f"{name}: names {len(takes)} different files: {', '.join(map(str, takes.values()))}")
        else:
            path = next(iter(takes.values()))
            data = path.read_bytes()
            if not is_take(data):
                problems.append(f"{name}: {path} is not one of his takes (META kind is not human)")
            else:
                found.append((path, data))
    return found, problems


def add(names: list, source: Path, archive: bool) -> int:
    """Files the takes ``names`` name (``find``), all or none: each verified, copied under its own name, then the
    archive."""
    found, problems = find(names, source)
    for path, data in found:
        try:
            _native.replay_verify(data)
        except Exception as error:  # pylint: disable=broad-exception-caught
            problems.append(f"{path}: does not verify: {error}")
    if problems:
        print('nothing filed:\n  ' + '\n  '.join(problems))
        return 1
    kept = {sha1(p.read_bytes()) for p in OWNER.rglob('*.tfreplay')}
    new = 0
    for path, data in found:
        key = sha1(data)
        if key in kept:
            print(f"already kept: {path}")
            continue
        target = destination(data, path.name)
        if target.exists():
            target = target.with_name(f"{target.stem}_{key[:8]}{target.suffix}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        kept.add(key)
        new += 1
        print(f"kept {target.relative_to(ROOT)}  <- {path}")
    print(f"{new} new take(s); {len(kept)} kept in {OWNER.relative_to(ROOT)}")
    if archive:
        archive_all()
    return 0


def archive_all() -> None:
    """Every take kept in ``replays/owner/`` into the backup (what it holds is skipped, by SHA-1)."""
    sys.path.insert(0, str(ROOT / 'scripts'))
    import archive_replays  # pylint: disable=import-outside-toplevel
    archive_replays.main([str(OWNER), '--kind', 'owner', '--keep-names'])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('add', help="file the takes he named (paths or file names under --from), archive, index")
    a.add_argument('names', nargs='+', metavar='PATH|NAME')
    a.add_argument('--from', dest='source', default=str(SOURCE))
    s = sub.add_parser('sync', help="file nothing new: archive what replays/owner holds, rewrite the index")
    for p in (a, s):
        p.add_argument('--no-archive', action='store_true', help="leave /backup alone")
    i = sub.add_parser('index', help="write replays/owner/INDEX.md")
    for p in (a, s, i):
        p.add_argument('--entry-rule', default=str(ENTRY_RULE))
    args = parser.parse_args(argv)
    jobs.require_job()
    status = 0
    if args.command == 'add':
        status = add(args.names, Path(args.source).expanduser(), not args.no_archive)
    elif args.command == 'sync' and not args.no_archive:
        archive_all()
    rule = entry_rule(args.entry_rule)
    write_index(rule, Path(args.entry_rule).name if rule else 'no entry rule file: layout unknown')
    return status


if __name__ == '__main__':
    sys.exit(main())
