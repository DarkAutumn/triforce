"""The Olive Garden (5:0x05 from E, six Blue Darknuts, the bag of 8): how a model and the planner play the same entry
states, from evaluation replays recorded with their decisions (`triforce evaluate ... --replays DIR --streams
decisions`) played frame by frame on the emulator alone (as `r34d3_lastpass.py` for Dungeon 3's last pass).

    python scripts/experiments/r34d5_garden.py LABEL=REPLAY_DIR[:EVAL.json] [...] [--json OUT]

Per replay: hearts on entering, the ending; the Darknuts alive frame by frame and the frame each dies, a kill counted
a bomb's when one of Link's bombs (slots $10/$11) was exploding (state $13/$14) in the 24 frames before it, else the
sword's; bombs placed (the count dropping) and when (before the first kill, between kills, after the last); every
hit (hearts lost): the kind of the decision Link was executing (the last decision before it), whether a bomb was
placed in the 48 frames before it, where Link stood (the E door he enters by, a corner, the room's N/S edge rows, the
middle) and how many Darknuts lived; after the last kill: the frames to the exit and the hits and deaths there.
With EVAL.json (the evaluation's records, matched by seed): bombs landed (`bombs.hits`).  Printed per label: endings,
hits a visit and their decisions and places, kills by sword and by bomb, bombs placed and landed and when, the fight's
length (frames to the last kill), the after-fight phase, the share of fight frames within 16 px of a Darknut and
where Link spends the fight."""

import argparse
import collections
import json
from pathlib import Path

from triforce._native import Replay, ReplayPlayer

OBJ_TYPE, OBJ_X, OBJ_Y, OBJ_STATE = 0x34F, 0x70, 0x84, 0xAC
HEARTS, PARTIAL, BOMBS = 0x66F, 0x670, 0x658
DARKNUTS = (0x0B, 0x0C)
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
    """Where Link stands: the room's floor runs x $20-$D0, y $5D-$CD; he enters by the E door at (224, 141)."""
    if x >= 0xC8 and abs(y - 141) <= 16:
        return 'E door'
    edge_x, edge_y = x <= 0x30 or x >= 0xC0, y <= 0x6D or y >= 0xBD
    if edge_x and edge_y:
        return 'corner'
    if edge_y:
        return 'N row' if y <= 0x6D else 'S row'
    if edge_x:
        return 'W/E wall'
    return 'middle'


def analyse(path: Path) -> dict:
    data = path.read_bytes()
    replay = Replay(data)
    player = ReplayPlayer(data)
    dec = replay.decisions()
    dec_frames = [int(f) for f in dec['frame']] if dec else []
    dec_kinds = [int(a[0]) for a in dec['action']] if dec else []
    ram = player.ram()
    h0 = prev = hearts(ram)
    last_b = peek(ram, BOMBS)
    damage, bombs, kills, kill_by = [], [], [], []
    near16 = fight = 0
    places = collections.Counter()
    alive_prev, all_dead, last_blast = None, None, -10**6
    di = 0
    for frame in range(replay.frames):
        player.step(1)
        ram = player.ram()
        if any(peek(ram, OBJ_STATE + s) in BLAST for s in BOMB_SLOTS):
            last_blast = frame
        lx, ly = peek(ram, OBJ_X), peek(ram, OBJ_Y)
        dk = []
        for s in range(1, 0x0C):
            if peek(ram, OBJ_TYPE + s) in DARKNUTS:
                dk.append(max(abs(peek(ram, OBJ_X + s) - lx), abs(peek(ram, OBJ_Y + s) - ly)))
        alive = len(dk)
        if alive_prev is not None and alive < alive_prev:
            n = alive_prev - alive
            kills.extend([frame] * n)
            kill_by.extend(['bomb' if frame - last_blast <= 24 else 'sword'] * n)
            if alive == 0 and all_dead is None:
                all_dead = frame
        alive_prev = alive
        if alive:
            fight += 1
            near16 += min(dk) <= 16
            places[place(lx, ly)] += 1
        while di + 1 < len(dec_frames) and dec_frames[di + 1] <= frame:
            di += 1
        h = hearts(ram)
        if h < prev:
            kind = KINDS[dec_kinds[di]] if dec_frames and dec_frames[di] <= frame else '-'
            damage.append({'frame': frame, 'lost': prev - h, 'x': lx, 'y': ly, 'place': place(lx, ly),
                           'darknuts': alive, 'kind': kind,
                           'after_bomb': any(frame - b <= 48 for b in bombs)})
        prev = h
        b = peek(ram, BOMBS)
        if b < last_b:
            bombs.extend([frame] * (last_b - b))
        last_b = b
    return {'replay': path.name, 'seed': int(replay.meta['seed']), 'ending': replay.ending, 'hearts_in': h0,
            'hearts_out': prev, 'frames': replay.frames, 'damage': damage, 'bombs': bombs, 'kills': kills,
            'kill_by': kill_by, 'all_dead': all_dead, 'fight_frames': fight, 'near16': near16,
            'places': dict(places)}


def when(r, b) -> str:
    """A bomb's placement against the kills: before the first, between kills, after the last."""
    if not r['kills'] or b < r['kills'][0]:
        return 'before'
    if r['all_dead'] is not None and b >= r['all_dead']:
        return 'after'
    return 'between'


def report(label, rows):
    n = len(rows)
    if not n:
        return
    endings = collections.Counter(r['ending'] for r in rows)
    won = sum(r['ending'] == 'win' for r in rows)
    hits = [d for r in rows for d in r['damage']]
    fight_hits = [d for d in hits if d['darknuts']]
    kinds = collections.Counter(d['kind'] for d in fight_hits)
    hit_places = collections.Counter(d['place'] for d in fight_hits)
    kills_by = collections.Counter(k for r in rows for k in r['kill_by'])
    whens = collections.Counter(when(r, b) for r in rows for b in r['bombs'])
    landed = [r['landed'] for r in rows if r.get('landed') is not None]
    dead = sorted(r['all_dead'] for r in rows if r['all_dead'] is not None)
    after = [r for r in rows if r['all_dead'] is not None]
    after_hits = sum(1 for r in after for d in r['damage'] if d['frame'] >= r['all_dead'])
    after_lost = sum(r['ending'] != 'win' for r in after)
    after_frames = sorted(r['frames'] - r['all_dead'] for r in after if r['ending'] == 'win')
    fight = sum(r['fight_frames'] for r in rows)
    places = collections.Counter()
    for r in rows:
        places.update(r['places'])
    gaps = sorted(b - a for r in rows for a, b in zip(r['kills'], r['kills'][1:]))
    print(f"== {label}: {n} visits, won {won} ({won / n:.1%}), endings {dict(endings)}, hearts in "
          f"{sum(r['hearts_in'] for r in rows) / n:.2f}")
    print(f"  hits {len(hits) / n:.2f} a visit ({len(fight_hits) / n:.2f} in the fight); fight hits by the decision "
          f"executing: {dict(kinds.most_common())}; with a bomb placed in the 48 frames before: "
          f"{sum(d['after_bomb'] for d in fight_hits)}; where: {dict(hit_places.most_common())}")
    print(f"  kills {sum(len(r['kills']) for r in rows) / n:.2f} a visit: sword {kills_by['sword'] / n:.2f}, bomb "
          f"{kills_by['bomb'] / n:.2f}; all six dead in {len(dead)} (frame median {dead[len(dead) // 2] if dead else '-'}); "
          f"frames between kills median {gaps[len(gaps) // 2] if gaps else '-'}")
    print(f"  bombs placed {sum(len(r['bombs']) for r in rows) / n:.2f} a visit (before the first kill "
          f"{whens['before'] / n:.2f}, between kills {whens['between'] / n:.2f}, after the last {whens['after'] / n:.2f})"
          + (f", landed {sum(landed) / len(landed):.2f}" if landed else '')
          + f"; first bomb frame median {sorted(r['bombs'][0] for r in rows if r['bombs'])[len([r for r in rows if r['bombs']]) // 2] if any(r['bombs'] for r in rows) else '-'}")
    print(f"  after the last kill: {len(after)} visits, lost {after_lost}, hits {after_hits}, frames to the exit "
          f"(won) median {after_frames[len(after_frames) // 2] if after_frames else '-'}; died before the last kill "
          f"{sum(r['ending'] != 'win' and r['all_dead'] is None for r in rows)}")
    print(f"  fight frames within 16 px of a Darknut {near16_share(rows, fight):.1%}; where Link spends the fight: "
          + ', '.join(f"{k} {v / fight:.0%}" for k, v in places.most_common()))
    for lo, hi in ((0, 3), (3.5, 5.5), (6, 7)):
        rs = [r for r in rows if lo <= r['hearts_in'] <= hi]
        if rs:
            print(f"  hearts in {lo}-{hi}: {len(rs)} visits, won {sum(r['ending'] == 'win' for r in rs) / len(rs):.1%}, "
                  f"hits {sum(len(r['damage']) for r in rs) / len(rs):.2f}")


def near16_share(rows, fight):
    return sum(r['near16'] for r in rows) / fight if fight else 0.0


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
    if args.json:
        args.json.write_text(json.dumps(out))


if __name__ == '__main__':
    main()
