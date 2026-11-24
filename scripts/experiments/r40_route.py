"""Round 40 (Route40): the fresh generation's d3first, its key and bomb ledger, and its start states set to it.

    python scripts/experiments/r40_route.py steps            # the route's steps (index, room, entry, exit, keys)
    python scripts/experiments/r40_route.py ledger           # every key event and locked door; the bomb ledger
    python scripts/experiments/r40_route.py check            # every d3first state against the ledger (no write)
    python scripts/experiments/r40_route.py patch            # set the states that differ to it (a queued job)

The ledgers are computed from the route data (``_native.route_steps('d3first')`` on the clip way, the route's
game.yaml rooms ``_native.game_rooms('d3first')``), not written by hand: a key is taken where a step's room holds one
(game.yaml ``treasure: key``) and the step takes its treasure (not ``skip``/``refuse``/``clear``), the first time; a key
door costs one the first time the route passes it from either side (asm:Door_TouchKey, asm:RoomFlags_SetDoorOpened),
but for Dungeon 1's 0x73 N, which the door trick opens for nothing (round 31, asm:Door_SetEnteringDoorwayOpened). The
key events are checked against round 39's list (``r39_keys.py``, cross-checked with 1000-run records), changed by
round 40: 3:0x6b's key left, the Moldorm's (2:0x3e) taken, 9:0x56's kept.

The bomb ledger is a run with no drop: a room's bombs (game.yaml ``treasure: bombs``, four) are taken the first time a
step takes its treasure while Link holds fewer than 8 (``bomb_rules: treasure_below: 8``); every bomb wall or rock wall
the route leaves through costs one, cheated when the bag is empty; the fights that use bombs as a weapon (the weapon
rooms, a step's ``bomb_reserve: 0``) are listed with what Link brings and spend nothing here (what they spend is play).
3:0x5b's first pass adds the first Red Darknut's drop where it falls (the cycle at 0: 41%, asm:Drop_GroupRates).

``check``/``patch`` read each state's keys (``$66E`` asm:InvKeys) and the kill counters (``$52A`` WorldKillCycle,
``$50`` HelpDropCount, ``$627`` WorldKillCount): a state at step N holds the count after step N - 1, and every state
before 3:0x5b's first pass holds no kill (the route's walk kills nothing before it).  ``patch`` loads each state at its
action boundary (a load and save without a write gives the same file), writes what differs, saves it, and notes the
change in its manifest entry (``keys``, ``kills``, as ``r39_keys.py`` did)."""
import argparse
import json
import os
import sys

ROUTE = 'd3first'
KEYS, CYCLE, HELP, STREAK = 0x66E, 0x52A, 0x50, 0x627
BAG = 8
FREE_DOORS = {(1, 0x73, 'N')}        # the door trick (round 31): 1:0x73's N key door opens with no key
OPPOSITE = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
# Round 39's key events (r39_keys.py: step -> room, change), with round 40's changes: the route's ledger must match.
R39 = {7: ('3:0x7b', 1), 8: ('3:0x6b', 1), 10: ('3:0x4b', 0), 12: ('3:0x49', 1), 17: ('3:0x59', -1),
       33: ('4:0x51', 1), 35: ('4:0x40', 1), 36: ('4:0x30', -1), 41: ('4:0x31', -1), 48: ('4:0x01', 1),
       65: ('1:0x53', 1), 67: ('1:0x33', 0), 68: ('1:0x23', 0), 74: ('1:0x43', -1), 76: ('1:0x45', 0),
       96: ('2:0x7e', 1), 99: ('2:0x4e', 1), 100: ('2:0x3e', 0), 127: ('5:0x06', -1), 135: ('5:0x66', 1),
       140: ('5:0x27', 0), 142: ('5:0x25', -1), 163: ('7:0x38', -1), 165: ('7:0x18', -1), 170: ('7:0x1b', -1),
       197: ('6:0x79', -1), 200: ('6:0x58', 1), 204: ('6:0x29', 1), 209: ('6:0x2d', 1), 210: ('6:0x2c', -1),
       221: ('8:0x5e', 1), 222: ('8:0x4e', -1), 246: ('9:0x56', 1), 249: ('9:0x14', -1), 262: ('9:0x16', -1),
       266: ('9:0x63', -1), 268: ('9:0x61', 1)}
R40 = {8: ('3:0x6b', 0), 100: ('2:0x3e', 1)}


def name(level, room, in_cave=False):
    return f"{level}:{room:#04x}" + (' cellar' if in_cave and level else ' cave' if in_cave else '')


def route(way='clips'):
    """The route's steps on one way (the clip way unless named), with each step's route index."""
    from triforce._native import route_steps  # pylint: disable=import-outside-toplevel
    return [s for s in route_steps(ROUTE) if s['way'] in (None, way)]


def raw_rooms():
    """game.yaml's base room entries by (level, room, in_cave): the keys `game_rooms` leaves out."""
    import yaml  # pylint: disable=import-outside-toplevel
    from triforce.paths import ROOT  # pylint: disable=import-outside-toplevel
    with open(os.path.join(ROOT, 'triforce', 'game.yaml'), encoding='utf-8') as f:
        data = yaml.safe_load(f)
    out = {}
    for r in data['rooms']:
        if 'routes' not in r:
            out[(r['level'], int(str(r['location']), 16), bool(r.get('in_cave')))] = r
    return out


def takes_treasure(step):
    return step['treasure'] not in ('skip', 'refuse', 'clear')


def key_events(steps, rooms):
    """[(index, room, change, what)] of every key pickup and key door on the way, in order."""
    taken, opened, events = set(), set(), []
    for s in steps:
        at = (s['level'], s['room'], s['in_cave'])
        room = rooms[at]
        change, what = 0, []
        if room['treasure'] == 'key' and takes_treasure(s) and at not in taken:
            taken.add(at)
            change += 1
            what.append('its key')
        door = room['exits'].get(s['exit'])
        if door and door['locked']:
            ident = frozenset({(at, s['exit']), (tuple(door['to']), OPPOSITE[s['exit']])})
            if ident not in opened:
                opened.add(ident)
                if (s['level'], s['room'], s['exit']) in FREE_DOORS:
                    what.append(f"{s['exit']} key door (the door trick: no key)")
                else:
                    change -= 1
                    what.append(f"{s['exit']} key door")
        if what:
            events.append((s['index'], name(*at), change, ', '.join(what)))
    return events


# The fights that throw bombs as a weapon (routes.yaml: a step's own `bomb_reserve` for its fight, or game.yaml's
# `bomb_weapon`), by (level, room, entry).
WEAPON_FIGHTS = {(3, 0x59, 'N'): "0x59's five Darknuts (reserve 4)", (3, 0x4d, 'W'): 'Manhandla',
                 (2, 0x0e, 'S'): 'Dodongo', (5, 0x05, 'E'): 'the Olive Garden', (5, 0x24, 'E'): 'Digdogger',
                 (7, 0x2a, 'W'): 'Aquamentus'}


def bomb_events(steps, rooms, raw, darknut_drop=False, spend=False):
    """[(index, room, what, bombs after, cheated so far, reserve)] on a run with no drop (`darknut_drop`: 3:0x5b's
    first Darknut drops its four). The fights that throw bombs (`WEAPON_FIGHTS`) spend nothing, or with `spend` every
    bomb above their reserve (a weapon room, game.yaml `bomb_weapon`, whatever the reserve): the two bounds."""
    count, cheated, taken, opened, out = 0, 0, set(), set(), []
    reserve = 0
    for s in steps:
        at = (s['level'], s['room'], s['in_cave'])
        room, base = rooms[at], raw.get(at, {})
        if s['bomb_reserve'] is not None:
            reserve = s['bomb_reserve']
        what = []
        if darknut_drop and at == (3, 0x5b, False) and s['entry'] == 'S':
            count = min(BAG, count + 4)
            what.append('the first Darknut drops 4')
        fight = WEAPON_FIGHTS.get((s['level'], s['room'], s['entry']))
        if fight:
            keep = 0 if base.get('bomb_weapon') else reserve
            what.append(f"{fight}: brought {count}" + (f", spends {max(count - keep, 0)}" if spend else ''))
            if spend:
                count = min(count, keep)
        if room['treasure'] == 'bombs' and takes_treasure(s) and at not in taken and count < BAG:
            taken.add(at)
            count = min(BAG, count + 4)
            what.append('its 4 bombs')
        door = room['exits'].get(s['exit'])
        wall = bool(door and door['bomb_wall'])
        rock = s['exit'] == 'cave' and s['level'] == 0 and not s['in_cave'] and (
            (base.get('entrance') or {}).get('how') == 'bomb' or base.get('bomb_secret'))
        if wall or rock:
            ident = frozenset({(at, s['exit'])} | ({(tuple(door['to']), OPPOSITE[s['exit']])} if wall else set()))
            if ident not in opened:
                opened.add(ident)
                if count == 0:
                    cheated += 1
                    what.append(f"{s['exit']} {'wall' if wall else 'rock wall'}: CHEATED")
                else:
                    count -= 1
                    what.append(f"{s['exit']} {'wall' if wall else 'rock wall'}")
        if what:
            out.append((s['index'], name(*at), '; '.join(what), count, cheated, reserve))
    return out


def ledger_after(steps, events):
    """The key count after each route step index (the states of the next step hold it)."""
    change = {i: c for i, _, c, _ in events}
    count, after = 0, {}
    for s in steps:
        count += change.get(s['index'], 0)
        after[s['index']] = count
    return after


def show_steps():
    from triforce._native import route_steps  # pylint: disable=import-outside-toplevel
    for s in route_steps(ROUTE):
        extra = {k: v for k, v in s.items() if k in ('treasure', 'bomb_reserve', 'bombs', 'waypoint', 'way') and v}
        print(f"{s['index']:3d} {name(s['level'], s['room'], s['in_cave']):14s} {s['entry']:>8s} -> {s['exit']:8s} "
              f"{s['specialist'] or '':18s} {extra if extra else ''}")


def show_ledger():
    from triforce._native import game_rooms  # pylint: disable=import-outside-toplevel
    steps, rooms, raw = route(), game_rooms(ROUTE), raw_rooms()
    events = key_events(steps, rooms)
    after = ledger_after(steps, events)
    want = {**R39, **R40}
    got = {i: (room, c) for i, room, c, _ in events}
    print('## Keys (a run that takes every key the route takes)\n')
    print('| Step | Room | Event | Change | Keys after | Round 39 |\n|---:|---|---|---:|---:|---:|')
    old = ledger_after(steps, [(i, r, c, '') for i, (r, c) in R39.items()])
    for i, room, c, what in events:
        print(f"| {i} | {room} | {what} | {c:+d} | {after[i]} | {old[i]} |")
    doors = [i for i, _, _, what in events if 'key door' in what and 'no key' not in what]
    low = min(after[i] for i in doors)
    print(f"\nKeys gained {sum(1 for e in events if 'its key' in e[3])}, key doors paid {len(doors)}; lowest after a "
          f"key door {low} (steps {[i for i in doors if after[i] == low]}), never below 0: "
          f"{min(after.values()) >= 0}, left at the end {after[steps[-1]['index']]}.")
    without = ledger_after(steps, [e for e in events if e[1] != '9:0x56'])
    print(f"Without 9:0x56's key: lowest {min(without.values())} (step "
          f"{min(without, key=lambda i: (without[i], i))}); without the Moldorm's: "
          f"{min(ledger_after(steps, [e for e in events if e[1] != '2:0x3e']).values())}.")
    mismatch = {i for i in set(got) | set(want) if got.get(i, (None, 0))[1] != want.get(i, (None, 0))[1]
                or (i in got and i in want and got[i][0] != want[i][0])}
    print(f"Against round 39's list with round 40's changes: {'identical' if not mismatch else sorted(mismatch)}")
    for darknut in (False, True):
        print(f"\n## Bombs, a run with no drop{' but the first Darknut' if darknut else ''}\n")
        print('| Step | Room | Event | Bombs after | Cheated | Reserve |\n|---:|---|---|---:|---:|---:|')
        for i, room, what, count, cheated, reserve in bomb_events(steps, rooms, raw, darknut):
            print(f"| {i} | {room} | {what} | {count} | {cheated} | {reserve} |")
    return after, mismatch


def states():
    """(game, path, step, name) of every d3first state."""
    import glob  # pylint: disable=import-outside-toplevel
    from triforce._native import Game  # pylint: disable=import-outside-toplevel
    from triforce.entry_states import RouteIndex  # pylint: disable=import-outside-toplevel
    from triforce.paths import STATES_DIR  # pylint: disable=import-outside-toplevel
    index, game = RouteIndex(ROUTE), Game(seed=0, states_dir=str(STATES_DIR))
    paths = sorted(glob.glob(os.path.join(STATES_DIR, f'{ROUTE}_*.state')))
    for n, path in enumerate(paths):
        game.load(path, randomize_rng=False, controllable=True)
        yield game, path, index.step_of(path), os.path.basename(path), (n, len(paths))


def check_or_patch(patch: bool):
    from triforce import jobs  # pylint: disable=import-outside-toplevel
    from triforce._native import game_rooms  # pylint: disable=import-outside-toplevel
    from triforce.events import write_json_atomic  # pylint: disable=import-outside-toplevel
    from triforce.paths import STATES_DIR  # pylint: disable=import-outside-toplevel
    if patch:
        jobs.require_job()
    steps = route()
    rooms = game_rooms(ROUTE)
    after = ledger_after(steps, key_events(steps, rooms))
    # Dungeon 4's plain way (steps 43-46) has states too: those read the plain way's ledger (the two agree from step
    # 47 on). A state at step N holds the count after the way's step before N.
    plain = route('plain')
    plain_only = {s['index'] for s in plain} - set(after)
    after_plain = ledger_after(plain, key_events(plain, rooms))
    first_pass = next(s['index'] for s in steps if (s['level'], s['room'], s['entry']) == (3, 0x5b, 'S'))
    changed = {}
    for game, path, step, base, (n, total) in states():
        if patch:
            jobs.progress(n, total, 'states')
        ledger = after_plain if step in plain_only else after
        want = {KEYS: ledger[max(i for i in ledger if i < step)] if step else 0}
        if step <= first_pass:
            want.update({CYCLE: 0, HELP: 0, STREAK: 0})
        have = {addr: game.read(addr) for addr in want}
        if have == want:
            continue
        print(f"{base} step {step}: " + ', '.join(f"${a:03X} {have[a]} -> {want[a]}" for a in want if have[a] != want[a]))
        if patch:
            for addr, value in want.items():
                game.write(addr, value)
            game.save_state_file(path)
            changed[base] = (have, want)
    print(f"{len(changed) if patch else 'check:'} states {'set to the ledger' if patch else ''}")
    if not changed:
        return
    manifest_path = os.path.join(STATES_DIR, 'entry_states.json')
    with open(manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)
    by = ("scripts/experiments/r40_route.py patch (round 40: 3:0x6b's key left, the Moldorm's taken, no kill "
          "before 3:0x5b)")
    for entry in manifest:
        if entry.get('file') in changed:
            have, want = changed[entry['file']]
            if have[KEYS] != want[KEYS]:
                entry['keys'] = {'was': have[KEYS], 'now': want[KEYS], 'by': by}
            kills = {f"${a:03X}": have[a] for a in (CYCLE, HELP, STREAK) if a in want and have[a] != want[a]}
            if kills:
                entry['kills'] = {'was': kills, 'now': 0, 'by': by}
    write_json_atomic(manifest_path, manifest, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('what', choices=('steps', 'ledger', 'check', 'patch'))
    args = parser.parse_args()
    if args.what == 'steps':
        show_steps()
    elif args.what == 'ledger':
        _, mismatch = show_ledger()
        return 1 if mismatch else 0
    else:
        check_or_patch(args.what == 'patch')
    return 0


if __name__ == '__main__':
    sys.exit(main())
