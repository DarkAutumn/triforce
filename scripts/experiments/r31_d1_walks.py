"""Phase IX (docs/routes/phase9-walks.md §1): the new Dungeon 1 beyond the entrance door (`r31_d1_door.py`).

    python scripts/experiments/r31_d1_walks.py bombspot                 # 1:0x53's N wall: where a bomb opens it
    python scripts/experiments/r31_d1_walks.py cross43 [--seeds 20]     # 1:0x43 entered from S (the hole), out N
    python scripts/experiments/r31_d1_walks.py states OUT               # pre-entry states of 0x73 after the trick

`bombspot` stages 0x53 from the route's own state `1_53s_0` with its five Stalfos **pinned** (`r29_route_walks.Stager`:
kept inside the NW wall, so the room is not "all dead"); one bomb a trial, written when the bag is empty.  `cross43`
makes 0x43's pre-entry state from S once (0x53 emptied by staging, its wall bombed, the scroll N saved) and crosses it
with its five Gels live, every trial with the game's own randomness for its seed.  `states` writes `1_73s_<chain>` as a
run would find 0x73 on coming back in from 0:0x37 (the route's `1_73s_<chain>` loaded, out S, in again; the pre-entry
frame of the level entry, $EE = 8), with the hearts and bombs today's full-route chain brings to Dungeon 1
(`CHAIN_STATES`): the scratch route's start states for the models' walk of the new Dungeon 1 (§1.6)."""

import argparse
import collections
from pathlib import Path

from r26_walks import BOMBS, CUR_OPENED_DOORS, KEYS, STATES
from r27_walks import TMP, Tally, hearts_bytes, load, summary
from r29_route_walks import Stager
from r31_d1_door import in_again, out_south

# What today's full-route chain (r29land-1k-full, 822 runs entering 1:0x73) brings to Dungeon 1: keys 3 (812) or 2;
# hearts 0.5-4.5 of 5 (median 3.5-4.0); bombs 0 (100 runs), 3-4 (126), 5-7 (579).  One state a chain: the upper,
# middle and low end, the last with an empty bag (0x53's wall then needs a granted bomb).
CHAIN_STATES = {0: (4.5, 7), 1: (3.5, 4), 2: (2.5, 0)}


def stager(path, seed=0) -> Stager:
    w = Stager(seed=seed)
    w.game.load(str(path), randomize_rng=True, seed=seed)
    w.staging = True
    w.wait_play()
    return w


def cmd_bombspot(_args):
    spots = [((x, 0x5D), 'UP') for x in range(0x58, 0xA0, 8)]
    spots += [((0x78, y), 'UP') for y in range(0x65, 0xA5, 8)]
    spots += [((0x60, 0x5D), 'RIGHT'), ((0x68, 0x5D), 'RIGHT'), ((0x88, 0x5D), 'LEFT'), ((0x90, 0x5D), 'LEFT')]
    for spot, face in spots:
        w = stager(STATES / '1_53s_0.state')
        if not w.walk(('y', spot[0], spot[1])):
            print(f'{spot}: not reached ({w.where()})')
            continue
        w.game.write(BOMBS, max(1, w.r(BOMBS)))
        n = w.bomb_wall(face)
        print(f'bomb facing {face:5s} from ({spot[0]:#04x},{spot[1]:#04x}): '
              + (f'open {n} frames after B ($EE ${w.r(CUR_OPENED_DOORS):02X})' if n >= 0 else 'shut'))


def state_43s() -> Path:
    path = TMP / '1_43s.state'
    if path.exists():
        return path
    w = stager(STATES / '1_53s_0.state')
    w.clear_room()
    assert w.walk((0x78, 0x5D))
    w.game.write(BOMBS, max(1, w.r(BOMBS)))
    assert w.bomb_wall('UP') >= 0
    w.press('NONE', 10)
    assert w.transit('UP', save=path), w.where()
    return path


def cmd_cross43(args):
    path = state_43s()
    first = load(path, 0)
    first.wait_play()
    print(f'0x43 from S: first movable at ({first.x:#04x},{first.y:#04x}), hearts {first.hearts()}, keys '
          f'{first.r(KEYS)}, monsters {collections.Counter(o[1] for o in first.objects())}')
    for name, points in (('W of the middle block', [(0x78, 0x9D), (0x60, 0x9D), (0x60, 0x7D), (0x78, 0x7D)]),
                         ('E of the middle block', [(0x78, 0x9D), (0x90, 0x9D), (0x90, 0x7D), (0x78, 0x7D)])):
        rows = []
        for seed in range(args.seeds):
            w = load(path, seed)
            w.wait_play()
            t = Tally(w)
            ok = t.walk(*points) and t.leave('UP', 300, line=('x', 0x78)) and w.room == 0x33
            rows.append((ok, t.lost(), t.frames, t.by))
        print(f'0x43 S -> N, {name}: {summary(rows)}')


def cmd_states(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for chain, (hearts, bombs) in CHAIN_STATES.items():
        w = load(STATES / f'1_73s_{chain}.state', chain)
        w.wait_play()
        t = Tally(w)
        assert out_south(t) > 0
        # in again, keeping the level entry's last pre-entry frame
        from r27_walks import is_pre_entry       # pylint: disable=import-outside-toplevel
        snap = None
        f0 = t.frames
        for _ in range(40):
            if not w.r(0x5A):
                break
            t.press('DOWN')
        for _ in range(400):
            if w.level == 1 and is_pre_entry(w):
                snap = w.game.snapshot()
            if w.level == 1 and w.mode == 5:
                break
            t.press('UP')
        assert snap is not None and w.r(CUR_OPENED_DOORS) == 8, w.where()
        w.game.restore(snap)
        for name, value in hearts_bytes(hearts, containers=5).items():
            w.game.set_value(name, value)
        w.game.write(BOMBS, bombs)
        path = out / f'1_73s_{chain}.state'
        w.game.save_state_file(str(path))
        v = load(path, 0)
        v.wait_play()
        print(f'{path}: {v.where()} hearts {v.hearts()} bombs {v.r(BOMBS)} keys {v.r(KEYS)} '
              f'$EE ${v.r(CUR_OPENED_DOORS):02X} ({t.frames - f0} frames from 0:0x37)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['bombspot', 'cross43', 'states'])
    p.add_argument('out', nargs='?')
    p.add_argument('--seeds', type=int, default=20)
    args = p.parse_args()
    {'bombspot': cmd_bombspot, 'cross43': cmd_cross43, 'states': cmd_states}[args.cmd](args)


if __name__ == '__main__':
    main()
