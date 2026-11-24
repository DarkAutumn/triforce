"""The aggregated run statistics as SQL tables in the store's index (``store.db``), queryable after the raw files are
trimmed or moved to the archive.  A pure duplicate of what ``triforce.run_stats`` computed: the ``*.stats.json``
files stay the source, these tables are derived and rebuilt from them (``reindex``).

- ``evaluations``: one row per statistics file (a chain, an evaluation, a training, a golden race, a 250 test), its
  top-level fields one to one (nested names joined with ``_``: ``rate_successes``, ``all_hearts_lost_sum``), the
  store's own identities (the effective team, its overrides, the data snapshots) and every run's outcome as one
  compressed JSON blob (``runs``: seeds, result letters, furthest step, frames, frames to each step ...), enough to
  pair two evaluations by seed once their raw files are gone.
- ``evaluation_rooms``: one row per evaluation and route step (a chain's ``by_step``; an evaluation's ``by_room``),
  each per-visit figure as its count, sum and sum of squares, so any mean and its standard error is a query.

Column names are ``run_stats``' field names (Stats40's, agreed Nov 23); ``SCHEMA_VERSION`` numbers the tables.
"""

import json
import sqlite3
import zlib
from pathlib import Path
from typing import Optional

from .run_stats import VISIT_MEANS

SCHEMA_VERSION = 1
MOMENTS = ('n', 'sum', 'sq')
DISTRIBUTION = ('n', 'sum', 'sq', 'median', 'p90')

# evaluations: (column, dotted path in the statistics file, JSON?)
EVALUATION_FIELDS = [
    ('stats_id', 'id', False), ('name', 'name', False), ('kind', 'kind', False), ('source', 'source', False),
    ('written', 'written', False), ('description', 'description', False),
    *[(f'job_{k}', f'job.{k}', False) for k in ('id', 'agent', 'task', 'subtask', 'description', 'commit', 'branch')],
    *[(f'provenance_{k}', f'provenance.{k}', False)
      for k in ('route', 'start_from', 'stop_at', 'stop_step', 'segment', 'power_on', 'clips', 'version', 'code',
                'data')],
    ('provenance_overrides', 'provenance.overrides', True), ('provenance_team', 'provenance.team', True),
    *[(f'records_{k}', f'records.{k}', False) for k in ('counters', 'runs', 'asked', 'unfinished_batches')],
    *[(f'rate_{k}', f'rate.{k}', False) for k in ('n', 'successes', 'rate', 'se')],
    *[(f'endings_{k}', f'endings.{k}', False) for k in ('wins', 'deaths', 'stuck', 'errors')],
    ('endings_stuck_reasons', 'endings.stuck_reasons', True), ('endings_death_rooms', 'endings.death_rooms', True),
    ('triforces', 'triforces', True), ('game_ended', 'game_ended', False), ('furthest_step', 'furthest_step', False),
    ('flags', 'flags', True),
    *[(f'beams_{group}_{m}', f'beams.{group}.{m}', False)
      for group in ('rooms_with_beams', 'frames_with_beams', 'held_frames') for m in MOMENTS],
    ('beams_never_lost', 'beams.never_lost', False),
    *[(f'all_hearts_lost_{m}', f'all.hearts_lost.{m}', False) for m in MOMENTS],
    ('all_shots_hit_sum', 'all.shots_hit.sum', False), ('all_kills_far_sum', 'all.kills_far.sum', False),
    *[(f'all_frames_{m}', f'all.frames.{m}', False) for m in DISTRIBUTION],
    *[(f'time_end_{m}', f'time.end.{m}', False) for m in DISTRIBUTION],
    ('by_state', 'by_state', True),
]
STORE_FIELDS = ('store_team', 'store_team_sha256', 'store_overrides', 'store_run', 'store_data')
ROOM_FIELDS = [
    'step', 'room', 'level', 'room_id', 'cave', 'family', 'network', 'network_run', 'network_sha1',
    'visits', 'runs', 'exits', 'wrong_exits', 'deaths', 'stuck', 'other_endings',
    *[f'frames_{m}' for m in DISTRIBUTION], *[f'lost_frame_{m}' for m in DISTRIBUTION],
    *[f'enter_{m}' for m in DISTRIBUTION],
    *[f'{name}_{m}' for name in VISIT_MEANS for m in MOMENTS],
]

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS stats_schema (version INTEGER);"
    "CREATE TABLE IF NOT EXISTS evaluations (id INTEGER PRIMARY KEY AUTOINCREMENT, schema INTEGER, raw_path TEXT "
    "UNIQUE, output INTEGER, filed REAL, from_step INTEGER, to_step INTEGER, "
    + ', '.join(f'{name} {"TEXT" if as_json else "NUMERIC"}' for name, _, as_json in EVALUATION_FIELDS)
    + ', ' + ', '.join(f'{name} TEXT' for name in STORE_FIELDS) + ", runs BLOB);"
    "CREATE TABLE IF NOT EXISTS evaluation_rooms (evaluation INTEGER REFERENCES evaluations(id) ON DELETE CASCADE, "
    + ', '.join(f'{name} NUMERIC' for name in ROOM_FIELDS) + ");"
    "CREATE INDEX IF NOT EXISTS evaluation_rooms_room ON evaluation_rooms (level, room_id);"
    "CREATE INDEX IF NOT EXISTS evaluation_rooms_family ON evaluation_rooms (family);"
    "CREATE INDEX IF NOT EXISTS evaluation_rooms_evaluation ON evaluation_rooms (evaluation);"
    "CREATE INDEX IF NOT EXISTS evaluations_team ON evaluations (store_team);"
    "CREATE INDEX IF NOT EXISTS evaluations_written ON evaluations (written);"
)


def create(con: sqlite3.Connection):
    """The tables, inside the caller's transaction (``executescript`` would commit it)."""
    for statement in SCHEMA.split(';'):
        if statement.strip():
            con.execute(statement)
    if con.execute('SELECT 1 FROM stats_schema').fetchone() is None:
        con.execute('INSERT INTO stats_schema VALUES (?)', (SCHEMA_VERSION,))


def _get(data, dotted: str):
    for part in dotted.split('.'):
        if not isinstance(data, dict) or part not in data:
            return None
        data = data[part]
    return data


def _value(value, as_json: bool):
    if value is None:
        return None
    if as_json or isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _split_room(room: Optional[str]) -> tuple:
    """``3:0x5b[:cave]`` -> (3, 0x5b, 0/1); None parts for anything else."""
    try:
        parts = str(room).split(':')
        return int(parts[0]), int(parts[1], 16), int(len(parts) > 2 and parts[2] == 'cave')
    except (ValueError, IndexError):
        return None, None, None


def room_rows(stats: dict) -> list[dict]:
    """The ``evaluation_rooms`` rows of a statistics file: a chain's ``by_step`` (with each step's frames to enter it),
    else an evaluation's ``by_room``."""
    steps = stats.get('by_step')
    groups = [(key, group) for key, group in sorted((steps or {}).items(), key=lambda item: int(item[0]))] \
        if steps else [(None, group | {'room': group.get('room', room)})
                       for room, group in sorted((stats.get('by_room') or {}).items())]
    rows = []
    for step, group in groups:
        level, room_id, cave = _split_room(group.get('room'))
        row = {'step': None if step is None else int(step), 'room': group.get('room'), 'level': level,
               'room_id': room_id, 'cave': cave, 'family': group.get('family'), 'network': group.get('network'),
               'network_run': group.get('network_run'), 'network_sha1': group.get('network_sha1'),
               **{k: group.get(k) for k in ('visits', 'runs', 'exits', 'wrong_exits', 'deaths', 'stuck')},
               'other_endings': _value(group.get('other_endings'), True)}
        for name in ('frames', 'lost_frame', 'enter'):
            for m in DISTRIBUTION:
                row[f'{name}_{m}'] = _get(group, f'{name}.{m}')
        for name in VISIT_MEANS:
            for m in MOMENTS:
                row[f'{name}_{m}'] = _get(group, f'{name}.{m}')
        rows.append(row)
    return rows


def index(con: sqlite3.Connection, stats: dict, raw_path, output: Optional[int] = None,
          store_fields: Optional[dict] = None, filed: Optional[float] = None) -> int:
    """Writes the rows of one statistics file (replacing an earlier indexing of the same ``raw_path``); returns the
    ``evaluations`` id.  ``store_fields``: the store's identities (``STORE_FIELDS``)."""
    create(con)
    con.execute('DELETE FROM evaluation_rooms WHERE evaluation IN (SELECT id FROM evaluations WHERE raw_path = ?)',
                (str(raw_path),))
    con.execute('DELETE FROM evaluations WHERE raw_path = ?', (str(raw_path),))
    steps = [int(k) for k in (stats.get('by_step') or {})]
    fields = {name: _value(_get(stats, dotted), as_json) for name, dotted, as_json in EVALUATION_FIELDS}
    fields.update({name: _value((store_fields or {}).get(name), False) for name in STORE_FIELDS})
    fields.update({'schema': SCHEMA_VERSION, 'raw_path': str(raw_path), 'output': output, 'filed': filed,
                   'from_step': min(steps) if steps else None,
                   'to_step': _get(stats, 'provenance.stop_step') or stats.get('furthest_step'),
                   'runs': zlib.compress(json.dumps(stats.get('runs'), separators=(',', ':')).encode())
                   if stats.get('runs') is not None else None})
    cursor = con.execute(f"INSERT INTO evaluations ({', '.join(fields)}) VALUES ({', '.join('?' * len(fields))})",
                         list(fields.values()))
    evaluation = cursor.lastrowid
    # A store made before a per-visit measure was added (round 40: `aimless_frames`) lacks its columns: add them, so
    # a newer file is never refused by an older table (NUMERIC, as the table's own; older rows read NULL).
    have = {r[1] for r in con.execute('PRAGMA table_info(evaluation_rooms)')}
    for name in ROOM_FIELDS:
        if name not in have:
            con.execute(f'ALTER TABLE evaluation_rooms ADD COLUMN {name} NUMERIC')
    for row in room_rows(stats):
        con.execute(f"INSERT INTO evaluation_rooms (evaluation, {', '.join(row)}) "
                    f"VALUES (?, {', '.join('?' * len(row))})", [evaluation, *row.values()])
    return evaluation


def runs_of(blob: Optional[bytes]) -> Optional[dict]:
    """An ``evaluations.runs`` blob as the statistics file's ``runs`` dict."""
    return None if blob is None else json.loads(zlib.decompress(blob))


def reindex(con: sqlite3.Connection, root: Path) -> dict:
    """Rebuilds both tables from the statistics files the store holds: every filed ``stats`` output, and each sealed
    run's ``train.stats.json``; returns counts (and what could not be read)."""
    con.execute('DROP TABLE IF EXISTS evaluation_rooms')
    con.execute('DROP TABLE IF EXISTS evaluations')
    con.execute('DROP TABLE IF EXISTS stats_schema')
    create(con)
    done, missing = 0, []
    sources = [(row['path'], row['id'], row['team'], row['run'], row['data'], row['created'])
               for row in con.execute("SELECT id, path, team, run, data, created FROM outputs WHERE kind = 'stats' "
                                      "ORDER BY id")]
    sources += [(f"runs/{row['family']}/{row['name']}/train.stats.json", None, None, f"{row['family']}/{row['name']}",
                 row['data'], row['sealed'])
                for row in con.execute("SELECT family, name, data, sealed FROM runs WHERE status = 'sealed'")]
    for path, output, team, run, data, created in sources:
        full = Path(root) / path
        if not full.is_file():
            if output is not None:
                missing.append(path)
            continue
        try:
            stats = json.loads(full.read_text(encoding='utf-8'))
        except ValueError:
            missing.append(path)
            continue
        index(con, stats, full, output, {'store_team': team, 'store_run': run, 'store_data': data}, created)
        done += 1
    return {'evaluations': done, 'rooms': con.execute('SELECT COUNT(*) FROM evaluation_rooms').fetchone()[0],
            'unreadable': missing}


EXAMPLES = {
    'deaths-by-room-over-time': (
        "Deaths a 1000 visits in a room (default 3:0x5b), by team and day",
        "SELECT substr(e.written, 1, 10) AS day, e.store_team AS team, r.room, SUM(r.visits) AS visits, "
        "SUM(r.deaths) AS deaths, ROUND(1000.0 * SUM(r.deaths) / SUM(r.visits), 1) AS deaths_per_1000 "
        "FROM evaluation_rooms r JOIN evaluations e ON e.id = r.evaluation WHERE r.level = :level AND "
        "r.room_id = :room GROUP BY day, team, r.room ORDER BY day, team"),
    'bombs-by-family': (
        "Bombs used a visit by family (mean and its standard error from the sums)",
        "SELECT r.family, SUM(r.bombs_used_n) AS n, ROUND(1.0 * SUM(r.bombs_used_sum) / SUM(r.bombs_used_n), 3) AS "
        "mean, ROUND(SQRT(MAX(0, 1.0 * SUM(r.bombs_used_sq) / SUM(r.bombs_used_n) - (1.0 * SUM(r.bombs_used_sum) / "
        "SUM(r.bombs_used_n)) * (1.0 * SUM(r.bombs_used_sum) / SUM(r.bombs_used_n))) / SUM(r.bombs_used_n)), 4) "
        "AS se FROM evaluation_rooms r WHERE r.bombs_used_n > 0 GROUP BY r.family ORDER BY mean DESC"),
}
