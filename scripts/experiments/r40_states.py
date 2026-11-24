"""Round 40 (States40): the fresh generation's d3first start states read back, and 3:0x5b's first Darknut drop
measured.

    python scripts/experiments/r40_states.py read NAME...          # keys, bombs, kill counters, hearts of states
    python scripts/experiments/r40_states.py bombs NAME=N...       # set states' bombs (a job)
    python scripts/experiments/r40_states.py drop --kills 1200     # the first Red Darknut's drop at cycle 0 (a job)

``read`` loads each state through ``Game.load`` (its playable frame, ``randomize_rng=False``) and prints what the
patch and the rebuilds set: keys ``$66E``, bombs ``$658``, the drop cycle ``$52A`` (WorldKillCycle), ``$50``
(HelpDropCount), ``$627`` (WorldKillCount), hearts and containers.

``bombs`` sets each named state's bombs ``$658`` to the plan's count (decisions 2026-11-23 "The fresh generation's
Dungeon 3, continued": 3:0x5b's first Darknut drops 4 or nothing, 3:0x4a adds 4, 3:0x59 is entered with 4 or 8), as
``r40_route.py patch`` sets the keys: the state loaded at its action boundary (``controllable``), the byte written, the
file saved, the change noted in its manifest entry (``bombs``: was, now, by).

``drop``: resets of ``d3first_3_5bs_<seed % 3>`` (3:0x5b from S, the first pass) with the game's own randomness for
the seed (``randomize_rng``, as every start), the kill counters at 0; one Red Darknut killed by a scripted sword
strike from its side (its HP written to one hit, Link written beside it facing it: measurement only, the kill itself
is the game's ``Damage_KillMonster``); the slot followed until the death sparkle ends and ``Drop_SetUpItem`` has
decided (asm:Obj_EndMetastate: the cycle advanced 0 -> 1, the slot type $60 with ``Item_ObjItemId`` the item, or
freed). The rule (asm:Drop_SetUpItem, asm:Drop_GroupRates): a Darknut is drop group 2, column 1 of its row is a bomb
($00), and with no forced drop the item appears only if ``Random,X`` (the slot's byte of the shift register) is below
$68: 104/256 = 40.6%."""
import argparse
import collections
import json
import math
import os
import sys

from triforce.paths import STATES_DIR

KEYS, BOMBS, CYCLE, HELP, STREAK = 0x66E, 0x658, 0x52A, 0x50, 0x627
HEARTS = 0x66F
MODE, PLAYABLE = 0x12, (5, 9, 11)          # GameMode: play, a cellar, a cave
OBJ_TYPE, OBJ_X, OBJ_Y, OBJ_DIR, OBJ_META, OBJ_HP, OBJ_INVINC = 0x34F, 0x70, 0x84, 0x98, 0x405, 0x485, 0x4F0
ITEM_ID = 0xAC                       # Item_ObjItemId (alias of ObjState) of a dropped item's slot
GRID_OFFSET = 0x394                  # ObjGridOffset of Link
DARKNUT_RED, DROPPED = 0x0B, 0x60
VEC = {1: (1, 0), 2: (-1, 0), 4: (0, 1), 8: (0, -1)}
SIDES = {1: (4, 8), 2: (4, 8), 4: (1, 2), 8: (1, 2)}
ITEMS = {0x00: 'bomb', 0x22: 'heart', 0x18: 'rupee', 0x0F: '5 rupees', 0x23: 'fairy', 0x21: 'clock'}


def game():
    from triforce._native import Game  # pylint: disable=import-outside-toplevel
    return Game(seed=0, states_dir=str(STATES_DIR))


def hearts(g) -> str:
    return f"{g.state.health:g}/{(g.read(HEARTS) >> 4) + 1}"


def read(names):
    g = game()
    print('| state | keys | bombs | $52A | $50 | $627 | hearts |\n|---|---:|---:|---:|---:|---:|---|')
    for n in names:
        path = n if os.path.exists(n) else os.path.join(STATES_DIR, n if n.endswith('.state') else f'{n}.state')
        g.load(path, randomize_rng=False, controllable=True)
        for _ in range(600):
            if g.read(MODE) in PLAYABLE:
                break
            g.press('NONE', 1)
        assert g.read(MODE) in PLAYABLE, f"{n}: mode {g.read(MODE)}"
        print(f"| {os.path.basename(path)} | {g.read(KEYS)} | {g.read(BOMBS)} | {g.read(CYCLE)} | {g.read(HELP)} | "
              f"{g.read(STREAK)} | {hearts(g)} |")


def set_bombs(counts: dict):
    from triforce.events import write_json_atomic  # pylint: disable=import-outside-toplevel
    from triforce.jobs import require_job  # pylint: disable=import-outside-toplevel
    require_job()
    g, changed = game(), {}
    for n, count in counts.items():
        base = n if n.endswith('.state') else f'{n}.state'
        path = os.path.join(STATES_DIR, base)
        g.load(path, randomize_rng=False, controllable=True)
        was = g.read(BOMBS)
        if was == count:
            continue
        g.write(BOMBS, count)
        g.save_state_file(path)
        changed[base] = (was, count)
        print(f"{base}: bombs {was} -> {count}")
    manifest_path = os.path.join(STATES_DIR, 'entry_states.json')
    with open(manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)
    by = ("scripts/experiments/r40_states.py bombs (round 40: the first Darknut's drop or none, 0x4a's four, 0x59 "
          "entered with 4 or 8)")
    for entry in manifest:
        if entry.get('file') in changed:
            was, now = changed[entry['file']]
            entry['bombs'] = {'was': was, 'now': now, 'by': by}
    write_json_atomic(manifest_path, manifest, indent=1)


def strike(g, slot) -> bool:
    """Link written beside the Darknut in `slot` on a flank, facing it, its HP one wooden-sword hit; one swing.
    True when the hit lands (the Darknut enters its death sparkle)."""
    mx, my, md = g.read(OBJ_X + slot), g.read(OBJ_Y + slot), g.read(OBJ_DIR + slot)
    md = md if md in VEC else 4
    for blade in SIDES[md]:
        lx, ly = mx - VEC[blade][0] * 0x10, my - VEC[blade][1] * 0x10
        if 0x10 <= lx <= 0xE0 and 0x50 <= ly <= 0xC8:
            break
    g.apply_ram({'link_x': lx, 'link_y': ly})
    g.write(OBJ_DIR, blade)
    g.write(GRID_OFFSET, 0)
    g.write(OBJ_INVINC + slot, 0)
    g.write(OBJ_HP + slot, 0x10)
    g.press('A', 1)
    for _ in range(30):
        if g.read(OBJ_META + slot) & 0x10:
            return True
        g.press('NONE', 1)
    return False


def darknuts(g) -> list:
    """The slots of the Red Darknuts out of their spawn clouds and alive (metastate 0)."""
    return [s for s in range(1, 12) if g.read(OBJ_TYPE + s) == DARKNUT_RED and g.read(OBJ_META + s) == 0]


def drop(kills: int):
    from triforce.jobs import progress, require_job  # pylint: disable=import-outside-toplevel
    require_job()
    g = game()
    items, misses, failed, cycles = collections.Counter(), 0, 0, collections.Counter()
    for seed in range(kills + 10_000):
        if sum(items.values()) >= kills:
            break
        progress(sum(items.values()), kills, 'kills')
        g.load(os.path.join(STATES_DIR, f'd3first_3_5bs_{seed % 3}.state'), randomize_rng=True, seed=seed,
               controllable=True)
        for addr in (CYCLE, HELP, STREAK):
            g.write(addr, 0)
        # The spawn clouds clear (metastate 0) before the strike.
        for _ in range(120):
            if darknuts(g):
                break
            g.press('NONE', 1)
        # A Darknut turns or steps between the write and the blade's frames now and then: strike again (Link's
        # hearts written back, so a touch on a missed strike costs nothing), up to 12 times a reset.
        full, slot = g.read(HEARTS), None
        for _ in range(12):
            alive = darknuts(g)
            if not alive:
                break
            g.write(HEARTS, full)
            if strike(g, alive[0]):
                slot = alive[0]
                break
            misses += 1
            g.press('NONE', 16)
        if slot is None:
            failed += 1
            continue
        for _ in range(60):
            if g.read(OBJ_TYPE + slot) != DARKNUT_RED:
                break
            g.press('NONE', 1)
        cycles[g.read(CYCLE)] += 1
        t = g.read(OBJ_TYPE + slot)
        items[ITEMS.get(g.read(ITEM_ID + slot), hex(g.read(ITEM_ID + slot))) if t == DROPPED else 'nothing'] += 1
    n = sum(items.values())
    p = items['bomb'] / n
    print(f"{n} kills ({misses} strikes that missed, retried; {failed} resets without a landed strike skipped); "
          f"the cycle after the kill: {dict(cycles)}")
    print(f"drops: {dict(items)}")
    print(f"bombs appear: {items['bomb']} of {n} = {p:.4f} +- {math.sqrt(p * (1 - p) / n):.4f} "
          f"(the rule: $68/256 = {0x68 / 256:.4f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    sub = parser.add_subparsers(dest='what', required=True)
    sub.add_parser('read').add_argument('names', nargs='+')
    sub.add_parser('bombs').add_argument('counts', nargs='+', metavar='NAME=N')
    sub.add_parser('drop').add_argument('--kills', type=int, default=1200)
    args = parser.parse_args()
    if args.what == 'read':
        read(args.names)
    elif args.what == 'bombs':
        set_bombs(dict((n, int(c)) for n, c in (a.split('=') for a in args.counts)))
    else:
        drop(args.kills)
    return 0


if __name__ == '__main__':
    sys.exit(main())
