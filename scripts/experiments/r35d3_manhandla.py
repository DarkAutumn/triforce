"""Manhandla (3:0x4d, entered from W): how a model and the planner play the same entry states, from evaluation replays
recorded with their decisions (`triforce evaluate ... --replays DIR --streams decisions`) played frame by frame on
the emulator alone (as `r34d5_garden.py` for the Olive Garden).

    python scripts/experiments/r35d3_manhandla.py LABEL=REPLAY_DIR[:EVAL.json] [...] [--json OUT]

Manhandla is one $3C object in several slots (the base and four hands). Per replay: hearts and bombs on entering,
the ending; his parts alive frame by frame and the frame each dies (a part's death counted a bomb's when one of
Link's bombs (slots $10/$11) was exploding (state $13/$14) in the 24 frames before it, else the sword's); his parts'
health ($0485 + slot) dropping, by bomb or sword the same way; bombs placed (the count dropping) and when (before
the first part dies, between, after he is dead); every hit (hearts lost): the kind of the decision Link was
executing (the last decision before it), whether a bomb was placed in the 48 frames before it, the nearest object's
type and distance, where Link stood (the W door he enters by, a corner, the room's N/S edge rows, the W/E walls, the
middle), and the hands alive then; the fight's length (frames to his death) and the share of fight frames within
16 / 32 px of a part. With EVAL.json (the evaluation's records, matched by seed): bombs landed (`bombs.hits`).
Printed per label and by hearts on entering (<= 1, 1.5-2, 2.5-3)."""

import argparse
import collections
import json
from pathlib import Path

from triforce._native import Replay, ReplayPlayer

OBJ_TYPE, OBJ_X, OBJ_Y, OBJ_STATE, OBJ_HP = 0x34F, 0x70, 0x84, 0xAC, 0x485
HEARTS, PARTIAL, BOMBS = 0x66F, 0x670, 0x658
MANHANDLA = 0x3C
BOMB_SLOTS = (0x10, 0x11)
BLAST = (0x13, 0x14)
KINDS = ('MOVE', 'WAIT', 'SWORD', 'BEAMS', 'BOMB', 'BOOMERANG', 'ARROW', 'CLIP', 'RECORDER', 'FOOD', 'CANDLE')


def peek(ram, addr):
    return int(ram[addr])


def hearts(ram) -> float:
    hc, p = peek(ram, HEARTS), peek(ram, PARTIAL)
    filled, containers = hc & 0x0F, (hc >> 4) + 1
    part = 1.0 if p >= 0x80 else 0.5 if p else 0.0
    return float(containers) if filled >= containers else filled + part


def place(x, y) -> str:
    """Where Link stands: the room's floor runs x $20-$D0, y $5D-$CD; he enters by the W door at (16, 141)."""
    if x <= 0x28 and abs(y - 141) <= 16:
        return 'W door'
    edge_x, edge_y = x <= 0x30 or x >= 0xC0, y <= 0x6D or y >= 0xBD
    if edge_x and edge_y:
        return 'corner'
    if edge_y:
        return 'N row' if y <= 0x6D else 'S row'
    if edge_x:
        return 'W/E wall'
    return 'middle'


def band(h) -> str:
    return '<=1' if h <= 1 else '1.5-2' if h <= 2 else '2.5-3'


def analyse(path: Path) -> dict:
    data = path.read_bytes()
    replay = Replay(data)
    player = ReplayPlayer(data)
    dec = replay.decisions()
    dec_frames = [int(f) for f in dec['frame']] if dec else []
    dec_kinds = [int(a[0]) for a in dec['action']] if dec else []
    ram = player.ram()
    h0 = prev = hearts(ram)
    b0 = last_b = peek(ram, BOMBS)
    damage, bombs, deaths, death_by, dmg_by = [], [], [], [], collections.Counter()
    near16 = near32 = fight = 0
    places = collections.Counter()
    hit_prev, alive_prev, dead_at, last_blast = {}, None, None, -10**6
    di = 0
    seen = False
    for frame in range(replay.frames):
        player.step(1)
        ram = player.ram()
        if any(peek(ram, OBJ_STATE + s) in BLAST for s in BOMB_SLOTS):
            last_blast = frame
        by = 'bomb' if frame - last_blast <= 24 else 'sword'
        lx, ly = peek(ram, OBJ_X), peek(ram, OBJ_Y)
        parts, hp, nearest = [], {}, (0, 999)
        for s in range(1, 0x0C):
            t = peek(ram, OBJ_TYPE + s)
            if not t:
                continue
            d = max(abs(peek(ram, OBJ_X + s) - lx), abs(peek(ram, OBJ_Y + s) - ly))
            if d < nearest[1]:
                nearest = (t, d)
            if t == MANHANDLA:
                parts.append(d)
                hp[s] = peek(ram, OBJ_HP + s)
        alive = len(parts)
        seen |= alive > 0
        for s, v in hp.items():
            if s in hit_prev and v < hit_prev[s]:
                dmg_by[by] += 1
        hit_prev = hp
        if alive_prev is not None and alive < alive_prev:
            n = alive_prev - alive
            deaths.extend([frame] * n)
            death_by.extend([by] * n)
            if alive == 0 and dead_at is None:
                dead_at = frame
        alive_prev = alive
        if alive:
            fight += 1
            near16 += min(parts) <= 16
            near32 += min(parts) <= 32
            places[place(lx, ly)] += 1
        while di + 1 < len(dec_frames) and dec_frames[di + 1] <= frame:
            di += 1
        h = hearts(ram)
        if h < prev:
            kind = KINDS[dec_kinds[di]] if dec_frames and dec_frames[di] <= frame else '-'
            damage.append({'frame': frame, 'lost': prev - h, 'x': lx, 'y': ly, 'place': place(lx, ly),
                           'parts': alive, 'kind': kind, 'who': f'${nearest[0]:02X}', 'dist': nearest[1],
                           'after_bomb': any(frame - b <= 48 for b in bombs)})
        prev = h
        b = peek(ram, BOMBS)
        if b < last_b:
            bombs.extend([frame] * (last_b - b))
        last_b = b
    return {'replay': path.name, 'seed': int(replay.meta['seed']), 'ending': replay.ending, 'hearts_in': h0,
            'bombs_in': b0, 'hearts_out': prev, 'frames': replay.frames, 'damage': damage, 'bombs': bombs,
            'deaths': deaths, 'death_by': death_by, 'dmg_by': dict(dmg_by), 'dead_at': dead_at,
            'fight_frames': fight, 'near16': near16, 'near32': near32, 'places': dict(places), 'seen': seen}


def when(r, b) -> str:
    if not r['deaths'] or b < r['deaths'][0]:
        return 'before'
    if r['dead_at'] is not None and b >= r['dead_at']:
        return 'after'
    return 'between'


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else '-'


def report(label, rows, name='all'):
    n = len(rows)
    if not n:
        return
    endings = collections.Counter(r['ending'] for r in rows)
    won = sum(r['ending'] == 'win' for r in rows)
    hits = [d for r in rows for d in r['damage']]
    fight_hits = [d for d in hits if d['parts']]
    kinds = collections.Counter(d['kind'] for d in fight_hits)
    hit_places = collections.Counter(d['place'] for d in fight_hits)
    who = collections.Counter(d['who'] for d in fight_hits)
    death_by = collections.Counter(k for r in rows for k in r['death_by'])
    dmg_by = collections.Counter()
    for r in rows:
        dmg_by.update(r['dmg_by'])
    whens = collections.Counter(when(r, b) for r in rows for b in r['bombs'])
    landed = [r['landed'] for r in rows if r.get('landed') is not None]
    dead = [r['dead_at'] for r in rows if r['dead_at'] is not None]
    fight = sum(r['fight_frames'] for r in rows)
    places = collections.Counter()
    for r in rows:
        places.update(r['places'])
    print(f"== {label} [{name}]: {n} visits, won {won} ({won / n:.1%}), endings {dict(endings)}, hearts in "
          f"{sum(r['hearts_in'] for r in rows) / n:.2f}, bombs in {sum(r['bombs_in'] for r in rows) / n:.2f}")
    print(f"  hits {len(hits) / n:.2f} a visit ({len(fight_hits) / n:.2f} in the fight, {1000 * len(fight_hits) / max(fight, 1):.2f} "
          f"a 1000 fight frames); by the decision executing: {dict(kinds.most_common())}; with a bomb placed in the "
          f"48 frames before: {sum(d['after_bomb'] for d in fight_hits)}; nearest: {dict(who.most_common(4))}; "
          f"where: {dict(hit_places.most_common())}")
    print(f"  parts killed {sum(len(r['deaths']) for r in rows) / n:.2f} a visit: sword {death_by['sword'] / n:.2f}, "
          f"bomb {death_by['bomb'] / n:.2f}; health drops: sword {dmg_by['sword'] / n:.2f}, bomb {dmg_by['bomb'] / n:.2f}; "
          f"dead in {len(dead)} (frame median {med(dead)}); first hit frame median "
          f"{med([r['damage'][0]['frame'] for r in rows if r['damage']])}")
    print(f"  bombs placed {sum(len(r['bombs']) for r in rows) / n:.2f} a visit (before the first part dies "
          f"{whens['before'] / n:.2f}, between {whens['between'] / n:.2f}, after his death {whens['after'] / n:.2f})"
          + (f", landed {sum(landed) / len(landed):.2f}" if landed else '')
          + f"; first bomb frame median {med([r['bombs'][0] for r in rows if r['bombs']])}")
    print(f"  fight frames within 16 / 32 px of a part {sum(r['near16'] for r in rows) / max(fight, 1):.1%} / "
          f"{sum(r['near32'] for r in rows) / max(fight, 1):.1%}; fight frames a visit {fight / n:.0f}; where Link "
          f"spends the fight: " + ', '.join(f"{k} {v / max(fight, 1):.0%}" for k, v in places.most_common()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('dirs', nargs='+', help='LABEL=REPLAY_DIR[:EVAL.json]')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    out = {}
    for spec in args.dirs:
        label, rest = spec.split('=', 1)
        d, _, ev = rest.partition(':')
        rows = [analyse(p) for p in sorted(Path(d).glob('*.tfreplay'))]
        if ev:
            landed = {int(rec['seed']): rec['bombs']['hits'] for rec in json.loads(Path(ev).read_text())['records']}
            for r in rows:
                r['landed'] = landed.get(r['seed'])
        out[label] = rows
        report(label, rows)
        for b in ('<=1', '1.5-2', '2.5-3'):
            report(label, [r for r in rows if band(r['hearts_in']) == b], b)
    if args.json:
        args.json.write_text(json.dumps(out))


if __name__ == '__main__':
    main()
