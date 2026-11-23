"""Our wins against the human speedrun board: the Legend of Zelda (NES), Any% No Up+A on speedrun.com.

    .venv/bin/python scripts/speedrun.py fetch                  # snapshot the board (run only when asked)
    .venv/bin/python scripts/speedrun.py time REPLAY...         # time wins and place them on the snapshot
    .venv/bin/python scripts/speedrun.py list [PATH...]         # every run's frames and duration

``list`` and ``time`` of more than one replay are jobs of the machine's queue (``jobs.require_job``: every replay
is played to its start and end): ``triforce job run --agent A --task T --subtask S --desc '...' --timeout 20m --
.venv/bin/python scripts/speedrun.py list``; the replays timed of the total are the progress.  ``time`` of one
replay and ``fetch`` run anywhere.

``fetch`` reads the whole board from speedrun.com's API (stdlib only) into ``target/leaderboard/zelda-any-noupa.json``
(not committed; it goes out of date, which is fine: refresh only when the owner asks).

**The timing (the board's rule, the owner, Nov 23):** from the first frame Link can be controlled on the overworld in
0:0x77 to the frame Link stands beside Zelda, the first frame he can no longer be controlled. Frame-exact, read from
the game's RAM as the replay plays:

- start: the replay's first frame after which `GameMode` ($12) is 5 (play) with `CurLevel` 0 and `RoomId` $77;
  Link's input is read from the next frame on, the first controllable frame;
- end: the first frame after which Zelda (object slot 1) is in state 1 with Link halted (`ObjState` $AC = $40) in
  9:0x32: asm:Zelda_Update caught him in front of her during that frame, the last frame he moved; the next frame
  is the first he cannot be controlled (mode $13, the win, begins the frame after: asm:Zelda_State1WaitForEnding).

The time is (end - start) frames at the NES's 60.0988 frames a second (NTSC: 39375000 / 655171); a run that did
not end the game is timed from the start to its last frame.  ``time`` places each win on the snapshot (one more than
the runs strictly faster).  Times are cached by the replay's SHA-1 in ``target/leaderboard/times.json``.
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / 'target' / 'leaderboard' / 'zelda-any-noupa.json'
API = 'https://www.speedrun.com/api/v1'
USER_AGENT = 'triforce-leaderboard/1.0'
NES_FPS = 39375000 / 655171          # 60.0988...: the NTSC NES's frame rate


def _get(path: str, **params):
    url = f'{API}{path}' + (('?' + urllib.parse.urlencode(params)) if params else '')
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)['data']


def _norm(text: str) -> str:
    return ''.join(c for c in text.lower() if c.isalnum())


def clock(seconds: float) -> str:
    """``41:36.123`` or ``1:02:03.456``."""
    hours, rest = divmod(float(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    text = f'{int(hours)}:{int(minutes):02d}:{secs:06.3f}' if hours else f'{int(minutes)}:{secs:06.3f}'
    return text


def _player(entry: dict) -> str:
    if entry.get('rel') == 'guest':
        return entry.get('name', '?')
    return (entry.get('names') or {}).get('international') or entry.get('name') or entry.get('id', '?')


def _board_ids(game_id: str):
    """The category and variable filter of Any% No Up+A: a category of that name, or Any% with a sub-category
    variable whose value is No Up+A."""
    categories = [c for c in _get(f'/games/{game_id}/categories') if c['type'] == 'per-game']
    for category in categories:
        if _norm(category['name']) == _norm('Any% No Up+A'):
            return category, {}
    for category in categories:
        if _norm(category['name']) != _norm('Any%'):
            continue
        for variable in _get(f"/categories/{category['id']}/variables"):
            for value_id, value in variable['values']['values'].items():
                if 'noupa' in _norm(value['label']):
                    return category, {f"var-{variable['id']}": value_id}
    raise SystemExit('speedrun.py: no Any% No Up+A board found; categories: '
                     + ', '.join(f"{c['id']} {c['name']}" for c in categories))


def fetch(out: Path) -> dict:
    game = _get('/games', abbreviation='the_legend_of_zelda')[0]
    category, variables = _board_ids(game['id'])
    board = _get(f"/leaderboards/{game['id']}/category/{category['id']}", embed='players', **variables)
    players = {p.get('id') or p.get('name'): p for p in board['players']['data']}
    runs = []
    for entry in board['runs']:
        run = entry['run']
        names = [_player(players.get(p.get('id') or p.get('name'), p)) for p in run['players']]
        runs.append({'place': entry['place'], 'player': ' & '.join(names), 'seconds': run['times']['primary_t'],
                     'time': clock(run['times']['primary_t']), 'date': run.get('date') or '',
                     'link': run.get('weblink', '')})
    runs.sort(key=lambda r: (r['place'], r['seconds']))
    data = {'game': game['names']['international'], 'category': category['name'] + (' (No Up+A)' if variables else ''),
            'fetched': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'runs': runs}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + '\n')
    print(f"{data['game']}: {data['category']}: {len(runs)} runs -> {out}")
    return data


def place(seconds: float, board: dict) -> int:
    return 1 + sum(1 for run in board['runs'] if run['seconds'] < seconds)


TIMES = ROOT / 'target' / 'leaderboard' / 'times.json'
START_SCAN = 60_000          # frames searched for the start (power on to control takes about 5,000-35,000)
END_SCAN = 4_000             # frames before a win's last frame searched for the touch of Zelda


def _span(data: bytes, frames: int, finished: bool) -> tuple:
    """(start, end) frame counts, the board's rule (the module docstring): start = frames run when the game is first
    in play in 0:0x77 (None when the replay never is: a segment), end = frames run when Zelda first holds Link
    (None when the run did not end the game)."""
    from triforce import _native  # pylint: disable=import-outside-toplevel
    player = _native.ReplayPlayer(data)
    start = end = None
    player.seek(0)
    while player.frame < min(frames, START_SCAN) and player.step(1):
        ram = player.ram()
        if ram[0x12] == 5 and ram[0x10] == 0 and ram[0xEB] == 0x77:
            start = player.frame
            break
    if finished:
        player.seek(max(0, frames - END_SCAN))
        while player.step(1):
            ram = player.ram()
            if ram[0x10] == 9 and ram[0xEB] == 0x32 and ram[0xAD] == 1 and ram[0xAC] == 0x40:
                end = player.frame
                break
    return start, end


def describe(path: Path, cache: dict) -> dict:
    """Any replay: its ending, frames, the board's start and end frames and the time between (the module
    docstring); a run that did not end the game is timed to its last frame."""
    import bz2  # pylint: disable=import-outside-toplevel
    import hashlib  # pylint: disable=import-outside-toplevel
    from triforce import _native  # pylint: disable=import-outside-toplevel
    data = path.read_bytes()
    data = bz2.decompress(data) if path.suffix == '.bz2' else data
    sha = hashlib.sha1(data).hexdigest()
    if sha not in cache:
        replay = _native.Replay(data)
        meta = replay.meta
        finished = str(replay.ending) == 'win' and 'game ended' in str(meta.get('ending', ''))
        start, end = _span(data, replay.frames, finished)
        cache[sha] = {'version': str(meta.get('triforce', '')), 'route': str(meta.get('route', '')),
                      'golden': str(meta.get('golden', '')) == 'True',
                      'ending': 'end' if finished else str(replay.ending), 'frames': replay.frames,
                      'start_frame': start, 'end_frame': end}
    row = {'file': str(path), 'sha1': sha, **cache[sha]}
    first = row['start_frame'] or 0
    last = row['end_frame'] if row['end_frame'] is not None else row['frames']
    row['played_frames'] = last - first
    row['seconds'] = row['played_frames'] / NES_FPS
    return row


def _replays(paths: list) -> list:
    files = []
    for name in paths:
        p = Path(name)
        files += sorted(f for f in p.rglob('*') if f.name.endswith(('.tfreplay', '.tfreplay.bz2'))) if p.is_dir() else [p]
    return files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', maxsplit=1)[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p_fetch = sub.add_parser('fetch', help="snapshot the board (only when asked)")
    p_fetch.add_argument('--out', default=str(BOARD))
    p_time = sub.add_parser('time', help="time winning replays and place them on the snapshot")
    p_time.add_argument('replays', nargs='+')
    p_time.add_argument('--board', default=str(BOARD))
    p_list = sub.add_parser('list', help="frames and duration of every run (wins placed on the snapshot)")
    p_list.add_argument('paths', nargs='*', default=[str(ROOT / 'corpus' / 'replays'), str(ROOT / 'replays')],
                        help="replay files or folders (default: corpus/replays and replays/)")
    p_list.add_argument('--board', default=str(BOARD))
    p_list.add_argument('--json', default=None, help="also write the rows to this JSON file")
    args = parser.parse_args(argv)
    if args.command == 'fetch':
        fetch(Path(args.out))
        return 0
    names = _replays(args.paths) if args.command == 'list' else [Path(n) for n in args.replays]
    from triforce import jobs  # pylint: disable=import-outside-toplevel
    if args.command == 'list' or len(names) > 1:
        jobs.require_job()      # every replay played to its start and end: a job
    board_path = Path(args.board)
    board = json.loads(board_path.read_text()) if board_path.exists() else None
    if board is None:
        print(f"(no board snapshot at {board_path}: `speedrun.py fetch` first; times only)")
    else:
        print(f"board: {board['category']}, {len(board['runs'])} runs, fetched {board['fetched']}")
    cache = json.loads(TIMES.read_text()) if TIMES.exists() else {}
    rows = []
    for f in names:
        rows.append(describe(f, cache))
        jobs.progress(len(rows), len(names), 'replays')
    TIMES.parent.mkdir(parents=True, exist_ok=True)
    TIMES.write_text(json.dumps(cache, indent=1) + '\n')
    for row in rows:
        row['place'] = place(row['seconds'], board) if board and row['ending'] == 'end' else None
    if args.command == 'list':
        print(f"{'version':9} {'kind':7} {'ending':7} {'frames':>7} {'start':>7} {'end':>7} {'played':>7} "
              f"{'time':>10} {'place':>9}  file")
        for row in sorted(rows, key=lambda r: (r['version'], r['file'])):
            kind = 'golden' if row['golden'] else 'run'
            where = f"{row['place']}/{len(board['runs']) + 1}" if row['place'] else ''
            print(f"{row['version']:9} {kind:7} {row['ending']:7} {row['frames']:>7} {row['start_frame'] or '-':>7} "
                  f"{row['end_frame'] or '-':>7} {row['played_frames']:>7} {clock(row['seconds']):>10} {where:>9}  "
                  f"{Path(row['file']).name}")
        if args.json:
            Path(args.json).write_text(json.dumps(rows, indent=1) + '\n')
        return 0
    for row in rows:
        if row['ending'] != 'end':
            print(f"{row['file']}: did not end the game ({row['ending']}), skipped")
            continue
        where = f"place {row['place']} of {len(board['runs']) + 1}" if board else ''
        print(f"{clock(row['seconds']):>10}  {where:>18}  frames {row['start_frame']}-{row['end_frame']}  {row['file']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
