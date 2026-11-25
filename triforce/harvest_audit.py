"""Audit a harvested snapshot against the route as it stands (round 42; docs/data-layout.md "Harvested states").

    python -m triforce.harvest_audit audit [--snapshot NAME|DIR] [--route d3first] [--families F,...] [--jobs 8]
        --out AUDIT.jsonl [--invalid INVALID.txt]
    python -m triforce.harvest_audit report AUDIT.jsonl [--chains DIR] --md OUT.md [--invalid INVALID.txt]

A harvested state is a chain's room entry, named for the route step Link plays there (``..._h<seed>_<n>``); its name
outlives the route that wrote it. 2.0.0.5 reordered ``d3first`` (Dungeon 2 before Dungeon 5) and kept the first
order's files: a fifth of the snapshot the generation's teachers drew held Link's pieces and items of no step of
today's route. Run it before a retrain on the snapshot the teachers will draw (``--snapshot``, default the store's
current one); it exits 1 when a state is invalid. A training reset refuses a harvested start that fails the route's
progress (``_native.start_on_route``), and the chains' harvest writes none; this audit is the full check.

``audit`` reads every ``<ns>_*.state`` of each family of ``triforce/groups.yaml`` (or ``--families``) and writes a row
per state (its name, what its machine holds read raw, before any load-time randomization, the step its name
resolves to, one reset in the family's task) with its problems:

- ``PROGRESS``: Link's triforce pieces, sword, bow, raft, ladder, recorder, red ring or silver arrow are the route's
  at no step of the room the name spells (``zelda::objectives::start_on_route``): a step leaving by stairs or a cave
  then has no target and no correct exit; one leaving by an edge is played with what the route never brings; an
  entry the route never makes gets an off-route objective (the first order's 0x0c from S, won in one step by stepping
  back out S).
- ``CONTAINERS``: heart containers other than the route's at that step (``entry_states.RouteIndex.hearts``); for an
  entry the route never makes, at every step of the room (``ITEMS`` alike).
- ``ITEMS``: an item the route does not hold at that step: a candle before 7:0x4a's cellar, arrows before the shop,
  food before 0x34's cave, a potion or the letter ever, more than 8 bombs or a bigger bag.
- ``CLIPS-OFF``: the start fixes the run-time clips switch off (a plain-way-only entry) in a task with clips enabled.
- ``REFUSED``: the env's reset fails on the state (the message is kept).
- ``NAME``: not an entry state's name.

Flagged, not invalid: ``OFF-ENTRY`` (the route never enters the room that way on that visit, but a step of the room
has Link's progress: a run back on the route's way; the training pool draws a harvested entry only where a seed
state has it) and ``NO-TARGET`` (the first observation names no target: a fight's objective may have none).

``report`` writes a table per family (states, invalid by problem, the hearts the states start with against the
baseline chains' ``health_in`` at the same room entry) and the list of invalid states, ``<family>/<file> <problems>``
a line: the input of a cleaned snapshot.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from . import _native, jobs, paths
from .entry_states import RouteIndex

INVALID = ('NAME', 'PROGRESS', 'CONTAINERS', 'ITEMS', 'CLIPS-OFF', 'REFUSED')
GROUPS = Path(__file__).resolve().parent / 'groups.yaml'
# RAM (zelda-asm/index/ram.md)
INV_BOMBS, INV_ARROW, INV_CANDLE, INV_FOOD, INV_POTION, INV_LETTER, INV_KEYS = (
    0x658, 0x659, 0x65B, 0x65D, 0x65E, 0x666, 0x66E)
HEART_VALUES, HEART_PARTIAL, INV_MAX_BOMBS = 0x66F, 0x670, 0x67C
# Items the route's cave steps give that StepProgress does not count: game.yaml treasure -> the item.
ITEM_TREASURES = {'red-candle': 'candle', 'wood-arrow': 'arrows', 'food': 'food'}


def route_items(route: str) -> list[set]:
    """Per step, the items of ``ITEM_TREASURES`` Link holds on entering it (its way: the clip way's and the shared
    steps', a plain-way step the plain way's)."""
    steps = _native.route_steps(route)
    rooms = _native.game_rooms(route)
    per = {}
    for way in ('clips', 'plain'):
        held = set()
        for s in steps:
            if s['way'] not in (None, way):
                continue
            per.setdefault(s['index'], set(held))
            treasure = rooms.get((s['level'], s['room'], s['in_cave']), {}).get('treasure') if s['in_cave'] else None
            if treasure in ITEM_TREASURES:
                held.add(ITEM_TREASURES[treasure])
    return [per[i] for i in range(len(steps))]


def hearts_of(read) -> tuple[int, float]:
    """(containers, hearts) as the game holds them (``asm:Link_BeHarmed``; ``start_states::hearts_encoding``)."""
    values, partial = read(HEART_VALUES), read(HEART_PARTIAL)
    return (values >> 4) + 1, (values & 0x0F) + (1.0 if partial >= 0x80 else 0.5 if partial > 0 else 0.0)


def teacher_families(only: Optional[list[str]] = None) -> list[tuple[str, str]]:
    """(teacher, policy key) of every member of ``triforce/groups.yaml``, or of the families named."""
    groups = yaml.safe_load(GROUPS.read_text(encoding='utf-8'))
    members = [(t, m) for t, v in groups['teachers'].items() for m in v['members']]
    return [(t, m) for t, m in members if not only or m.split('@')[0] in only or m in only]


def audit_family(args) -> list[dict]:
    """Every state of one family's folder in the snapshot (module doc)."""
    teacher, key, snapshot, route = args
    from .lab.collect import task_yaml  # pylint: disable=import-outside-toplevel
    from .specialist import load_specialist  # pylint: disable=import-outside-toplevel
    family = key.split('@')[0]
    config = load_specialist(task_yaml(key)).replace(state_routes=[route])
    ns = _native.route_namespace(route)
    folder = Path(snapshot) / family
    files = sorted(folder.glob(f'{ns}_*.state' if ns else '*.state')) if folder.is_dir() else []
    steps = _native.route_steps(route)
    items_at = route_items(route)
    index = RouteIndex(route)
    by_entry, by_room = defaultdict(list), defaultdict(list)
    for s in steps:
        by_entry[(s['level'], s['room'], s['in_cave'], s['entry'], s['visit'])].append(s['index'])
        by_room[(s['level'], s['room'], s['in_cave'])].append(s['index'])
    env = _native.Env(config, explicit_starts=True)
    game = _native.Game(seed=0)
    rows = []
    for path in files:
        label = path.stem
        row = {'teacher': teacher, 'family': family, 'key': key, 'harvested_fraction': config.harvested_fraction,
               'file': path.name, 'mtime': os.path.getmtime(path), 'problems': []}
        bare = label[len(ns) + 1:] if ns else label
        row['stem'] = bare.rsplit('_h', 1)[0] if '_h' in bare else bare.rsplit('_', 1)[0]
        if '_h' in bare:
            seed, n = bare.rsplit('_h', 1)[1].split('_')
            row['seed'], row['room_index'] = int(seed), int(n)
        state = game.restore(_native.Snapshot.from_state_file(path.read_bytes()))
        read = game.read
        containers, hearts = hearts_of(read)
        row.update(containers=containers, hearts=hearts, bombs=read(INV_BOMBS), max_bombs=read(INV_MAX_BOMBS),
                   keys=read(INV_KEYS), triforce=state.var('triforce'))
        named = _native.state_name(label)
        if named is None:
            row['problems'].append('NAME')
            rows.append(row)
            continue
        level, room, in_cave, entry, visit = named
        row.update(level=level, room=room, in_cave=in_cave, entry=entry, visit=visit)
        if not _native.start_on_route(route, label, state):
            row['problems'].append('PROGRESS')
        named_steps = by_entry.get((level, room, in_cave, entry, visit), [])
        if not named_steps:
            row['problems'].append('OFF-ENTRY')
        # The hearts and items of the named step, or of one of the room's steps for an entry the route never makes.
        candidates = named_steps or by_room.get((level, room, in_cave), [])
        if candidates:
            row['step'] = candidates[0]
            route_containers = [index.hearts(step) for step in candidates]
            if containers not in route_containers:
                row['problems'].append('CONTAINERS')
                row['route_containers'] = route_containers[0]
            items = []
            for name, addr in (('candle', INV_CANDLE), ('arrows', INV_ARROW), ('food', INV_FOOD)):
                if read(addr) and all(name not in items_at[step] for step in candidates):
                    items.append(name)
            items += [name for name, addr in (('potion', INV_POTION), ('letter', INV_LETTER)) if read(addr)]
            if row['bombs'] > 8 or row['max_bombs'] != 8:
                items.append('bombs')
            if items:
                row['problems'].append('ITEMS')
                row['items'] = items
        try:
            obs, info = env.reset(seed=0, options={'start_state': str(path)})
            target = obs['target']
            if target[4] >= 1.0 and not target[:4].any():
                row['problems'].append('NO-TARGET')
            row['clips_fixed'] = info.get('clips_fixed')
            if row['clips_fixed'] is False and config.clips.enabled:
                row['problems'].append('CLIPS-OFF')
        except Exception as e:  # pylint: disable=broad-except
            row['problems'].append('REFUSED')
            row['refused'] = str(e)[:300]
        rows.append(row)
    return rows


def invalid(row: dict) -> list[str]:
    return [p for p in row['problems'] if p in INVALID]


def write_invalid(rows: list[dict], out: Path) -> int:
    bad = sorted((r for r in rows if invalid(r)), key=lambda r: (r['family'], r['file']))
    with out.open('w', encoding='utf-8') as f:
        for r in bad:
            extra = f" items={','.join(r['items'])}" if 'items' in r else ''
            f.write(f"{r['family']}/{r['file']} {' '.join(invalid(r))}{extra}\n")
    return len(bad)


def audit(args) -> int:
    jobs.require_job()
    snapshot = Path(args.snapshot) if args.snapshot and Path(args.snapshot).is_dir() else (
        paths.data('harvested', args.snapshot) if args.snapshot else paths.data_dir('harvested'))
    work = [(t, k, str(snapshot), args.route) for t, k in teacher_families(args.families)]
    rows = []
    with Pool(args.jobs) as pool, Path(args.out).open('w', encoding='utf-8') as f:
        for done, family_rows in enumerate(pool.imap_unordered(audit_family, work), 1):
            for row in family_rows:
                f.write(json.dumps(row) + '\n')
            rows += family_rows
            jobs.progress(done, len(work), 'families')
    bad = sum(1 for r in rows if invalid(r))
    if args.invalid:
        write_invalid(rows, Path(args.invalid))
    print(f"{snapshot}: {len(rows)} states of {len(work)} families on route {args.route}, {bad} invalid; "
          f"rows in {args.out}" + (f", the invalid list in {args.invalid}" if args.invalid else ''))
    return 1 if bad else 0


def entry_of(prev: dict, room: dict) -> str:
    """The side a chain room record was entered from (routes.yaml entries)."""
    pl, pr, pc = prev['level'], int(prev['room'], 16), prev['in_cave']
    level, r, c = room['level'], int(room['room'], 16), room['in_cave']
    if c:
        return 'S'
    if pc:
        return 'cave'
    if pl != level:
        return 'cave' if level == 0 else 'S'
    return {0x10: 'N', -0x10: 'S', 1: 'W', -1: 'E'}.get(r - pr, '?')


def chain_hearts(folder: Path) -> dict:
    """(level, room, in_cave, entry) -> every visit's ``health_in`` in the chains of ``folder``."""
    out = defaultdict(list)
    for f in sorted(folder.glob('*.jsonl')):
        with f.open(encoding='utf-8') as lines:
            for line in lines:
                rooms = json.loads(line).get('rooms', [])
                for prev, room in zip(rooms, rooms[1:]):
                    out[(room['level'], int(room['room'], 16), room['in_cave'], entry_of(prev, room))].append(
                        room['health_in'])
    return out


def report(args) -> int:
    rows = [json.loads(line) for line in Path(args.audit).read_text(encoding='utf-8').splitlines()]
    chains = chain_hearts(Path(args.chains)) if args.chains else {}
    if args.invalid:
        write_invalid(rows, Path(args.invalid))
    fams = defaultdict(list)
    for r in rows:
        fams[(r['teacher'], r['key'])].append(r)
    lines = ['| teacher | family (harvested_fraction) | states | invalid | PROGRESS | CONTAINERS | ITEMS | CLIPS-OFF '
             '| REFUSED | OFF-ENTRY (valid) | NO-TARGET (valid) |', '|---|---|' + '---:|' * 9]
    total = Counter()
    for (teacher, key), rs in sorted(fams.items()):
        c = Counter(p for r in rs for p in set(r['problems']))
        bad = sum(1 for r in rs if invalid(r))
        flagged = {p: sum(1 for r in rs if p in r['problems'] and not invalid(r)) for p in ('OFF-ENTRY', 'NO-TARGET')}
        total.update({'states': len(rs), 'invalid': bad})
        lines.append(f"| {teacher} | {key} ({rs[0]['harvested_fraction']}) | {len(rs)} | {bad} | {c['PROGRESS']} | "
                     f"{c['CONTAINERS']} | {c['ITEMS']} | {c['CLIPS-OFF']} | {c['REFUSED']} | {flagged['OFF-ENTRY']} "
                     f"| {flagged['NO-TARGET']} |")
    lines.append(f"\nAll: {total['states']} states, {total['invalid']} invalid.\n")
    for (teacher, key), rs in sorted(fams.items()):
        lines += [f"\n### {key} ({teacher}, harvested_fraction {rs[0]['harvested_fraction']})\n",
                  '| room entry | states | invalid (why) | harvested (dates, seeds) | hearts: mean, ≤1 | valid only: mean, '
                  '≤1 | baseline chains: visits, mean, ≤1 |', '|---|---:|---|---|---|---|---|']
        by = defaultdict(list)
        for r in rs:
            by[r['stem']].append(r)
        for stem, srs in sorted(by.items()):
            bad = [r for r in srs if invalid(r)]
            why = ', '.join(f'{k} {v}' for k, v in Counter(p for r in bad for p in invalid(r)).most_common())
            dates = Counter(time.strftime('%m-%d', time.localtime(r['mtime'])) for r in srs)
            seeds = Counter((r.get('seed', -1) // 1000) * 1000 for r in srs)
            prov = ('; '.join(f'{d}: {n}' for d, n in sorted(dates.items())) + ' (seeds '
                    + ', '.join(f'{s}+: {n}' for s, n in sorted(seeds.items())) + ')')
            h = np.array([r['hearts'] for r in srs])
            good = np.array([r['hearts'] for r in srs if not invalid(r)])
            r0 = srs[0]
            base = np.array(chains.get((r0.get('level'), r0.get('room'), r0.get('in_cave'), r0.get('entry')), []))
            base_s = f"{len(base)}, {base.mean():.2f}, {np.mean(base <= 1.0):.0%}" if len(base) else '-'
            good_s = f"{good.mean():.2f}, {np.mean(good <= 1.0):.0%}" if len(good) else '-'
            lines.append(f"| {stem} | {len(srs)} | {len(bad)}{f' ({why})' if why else ''} | {prov} | "
                         f"{h.mean():.2f}, {np.mean(h <= 1.0):.0%} | {good_s} | {base_s} |")
    Path(args.md).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"wrote {args.md}" + (f", {args.invalid}" if args.invalid else ''))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog='python -m triforce.harvest_audit', description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('audit', help='audit a snapshot (exit 1 when a state is invalid)')
    a.add_argument('--snapshot', help='a harvested snapshot name or folder (default: the current one)')
    a.add_argument('--route', default='d3first')
    a.add_argument('--families', type=lambda s: s.split(','), help='families or policy keys (default: groups.yaml)')
    a.add_argument('--jobs', type=int, default=8)
    a.add_argument('--out', required=True)
    a.add_argument('--invalid', help='write the invalid states, one `<family>/<file> <problems>` a line')
    r = sub.add_parser('report', help='tables per family from an audit')
    r.add_argument('audit')
    r.add_argument('--chains', help='a folder of chain .jsonl whose room records give the hearts runs arrive with')
    r.add_argument('--md', required=True)
    r.add_argument('--invalid')
    args = p.parse_args(argv)
    return audit(args) if args.cmd == 'audit' else report(args)


if __name__ == '__main__':
    sys.exit(main())
