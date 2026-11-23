"""Our wins against the human speedrun board: the Legend of Zelda (NES), Any% No Up+A on speedrun.com.

    .venv/bin/python scripts/speedrun.py fetch                  # snapshot the board (run only when asked)
    .venv/bin/python scripts/speedrun.py time REPLAY...         # time wins and place them on the snapshot

``fetch`` reads the whole board from speedrun.com's API (stdlib only) into ``target/leaderboard/zelda-any-noupa.json``
(not committed; it goes out of date, which is fine: refresh only when the owner asks).

``time`` times each replay that ends in a win (golden paths and single runs alike): **from the frame Link can first
be controlled to the replay's last frame, the game's end**, at the NES's 60.0988 frames a second (NTSC: 39375000 /
655171), and gives the place that time would take on the snapshot (one more than the runs strictly faster).
Replays that are not wins are skipped with a note.
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


def describe(path: Path) -> dict:
    """Any replay: its ending, frames, the frame Link can first be controlled, and the duration from there to the
    last frame at the NES's rate (for a win, the speedrun time)."""
    import bz2  # pylint: disable=import-outside-toplevel
    from triforce import _native  # pylint: disable=import-outside-toplevel
    data = path.read_bytes()
    replay = _native.Replay(bz2.decompress(data) if path.suffix == '.bz2' else data)
    control = next((e['frame'] for e in replay.events() if e.get('type') == 'control'), 0)
    meta = replay.meta
    finished = str(replay.ending) == 'win' and 'game ended' in str(meta.get('ending', ''))
    return {'file': str(path), 'version': str(meta.get('triforce', '')), 'route': str(meta.get('route', '')),
            'golden': str(meta.get('golden', '')) == 'True', 'ending': 'end' if finished else str(replay.ending),
            'frames': replay.frames, 'control_frame': control, 'played_frames': replay.frames - control,
            'seconds': (replay.frames - control) / NES_FPS}


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
    board_path = Path(args.board)
    board = json.loads(board_path.read_text()) if board_path.exists() else None
    if board is None:
        print(f"(no board snapshot at {board_path}: `speedrun.py fetch` first; times only)")
    else:
        print(f"board: {board['category']}, {len(board['runs'])} runs, fetched {board['fetched']}")
    if args.command == 'list':
        rows = [describe(f) for f in _replays(args.paths)]
        print(f"{'version':9} {'kind':7} {'ending':7} {'frames':>7} {'control':>7} {'played':>7} {'time':>10} "
              f"{'place':>9}  file")
        for row in sorted(rows, key=lambda r: (r['version'], r['file'])):
            row['place'] = place(row['seconds'], board) if board and row['ending'] == 'end' else None
            kind = 'golden' if row['golden'] else 'run'
            where = f"{row['place']}/{len(board['runs']) + 1}" if row['place'] else ''
            print(f"{row['version']:9} {kind:7} {row['ending']:7} {row['frames']:>7} {row['control_frame']:>7} "
                  f"{row['played_frames']:>7} {clock(row['seconds']):>10} {where:>9}  {Path(row['file']).name}")
        if args.json:
            Path(args.json).write_text(json.dumps(rows, indent=1) + '\n')
        return 0
    for name in args.replays:
        row = describe(Path(name))
        if row['ending'] != 'end':
            print(f"{name}: did not end the game ({row['ending']}), skipped")
            continue
        where = f"place {place(row['seconds'], board)} of {len(board['runs']) + 1}" if board else ''
        print(f"{clock(row['seconds']):>10}  {where:>18}  frames {row['control_frame']}-{row['frames']}  {name}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
