"""Round 39: d3first's key ledger, and its start states set to it (the owner, Nov 22: Moldorm crossed, 9:0x56's key
taken instead).

    python scripts/experiments/r39_keys.py ledger          # the count after every key event, both ledgers
    python scripts/experiments/r39_keys.py check           # every d3first state's keys against the ledger
    python scripts/experiments/r39_keys.py patch           # set the states that differ (from step 101) to it

Keys are one counter across levels (asm:InvKeys, RAM $66E); a key door spends one on first touch. The events are
the route's (step numbers of 2.0.0.7's 290 steps), checked against the 1000-run records by the round 38 scout
(target/owner-review/r38-moldorm-key.md): 21 keys gained and 20 spent with Moldorm's key. A state at step N is
saved before N's room is entered, so it holds the count after step N - 1. ``patch`` loads each state at its action
boundary (a load and save without a write gives the same file, byte for byte), writes $66E and saves it, and notes
the change in the manifest entry (``keys``); the states before step 101 (Moldorm's room's own state among them) and
the plain stretch's 4:0x30 (step 44, whose key door is 0x30's) are left as they are."""
import argparse
import json
import os
import sys

KEYS = 0x66E                                        # asm:InvKeys

# step -> (room, change, what)
EVENTS = {
    7: ('3:0x7b', +1, 'key in the open'), 8: ('3:0x6b', +1, '10 Gels'), 10: ('3:0x4b', 0, '+1 (3 Zols), W key door'),
    12: ('3:0x49', +1, 'key'), 17: ('3:0x59', -1, 'E key door'), 33: ('4:0x51', +1, 'key'), 35: ('4:0x40', +1, 'key'),
    36: ('4:0x30', -1, 'E key door'), 41: ('4:0x31', -1, 'N key door'), 48: ('4:0x01', +1, 'key'),
    65: ('1:0x53', +1, 'key'), 67: ('1:0x33', 0, '+1, N key door'), 68: ('1:0x23', 0, '+1 (Goriyas), W key door'),
    74: ('1:0x43', -1, 'E key door'), 76: ('1:0x45', 0, '+1, N key door'), 96: ('2:0x7e', +1, '5 Ropes'),
    99: ('2:0x4e', +1, '5 Ropes'), 100: ('2:0x3e', +1, 'Moldorm'), 127: ('5:0x06', -1, 'W key door'),
    135: ('5:0x66', +1, 'its Gibdo'), 140: ('5:0x27', 0, '+1, W key door'), 142: ('5:0x25', -1, 'W key door'),
    163: ('7:0x38', -1, 'N key door'), 165: ('7:0x18', -1, 'E key door'), 170: ('7:0x1b', -1, 'E key door'),
    197: ('6:0x79', -1, 'W key door'), 200: ('6:0x58', +1, 'key'), 204: ('6:0x29', +1, 'key'),
    209: ('6:0x2d', +1, 'key'), 210: ('6:0x2c', -1, 'N key door (Gohma)'), 221: ('8:0x5e', +1, '5 Blue Darknuts'),
    222: ('8:0x4e', -1, 'N key door'), 249: ('9:0x14', -1, 'E key door'), 262: ('9:0x16', -1, 'N key door'),
    266: ('9:0x63', -1, 'W key door (into 0x62)'), 268: ('9:0x61', +1, 'Patra'),
}
# Round 39: Moldorm crossed (no key), 9:0x56's key taken (2 Zols, 2 Like-Likes).
CHANGES = {100: ('2:0x3e', 0, 'crossed'), 246: ('9:0x56', +1, '2 Zols, 2 Like-Likes')}
DOORS = {step for step, (_, change, what) in EVENTS.items() if 'door' in what}
STEPS = 290


def ledger(events: dict) -> list:
    """The count after each step 0..STEPS-1."""
    count, after = 0, []
    for step in range(STEPS):
        count += events.get(step, (None, 0))[1]
        after.append(count)
    return after


def show():
    old, new = ledger(EVENTS), ledger({**EVENTS, **CHANGES})
    print('| Step | Room | Event | Before | Round 39 |\n|---:|---|---|---:|---:|')
    for step in sorted({**EVENTS, **CHANGES}):
        room, _, what = CHANGES[step] if step in CHANGES else EVENTS[step]
        if step in CHANGES and step in EVENTS:
            what = f"{EVENTS[step][2]} -> {what}"
        print(f"| {step} | {room} | {what} | {old[step]} | {new[step]} |")
    for name, after in (('before', old), ('round 39', new)):
        low = min(after[s] for s in DOORS)
        print(f"{name}: lowest after a key door {low} (steps {[s for s in sorted(DOORS) if after[s] == low]}), "
              f"never below 0: {min(after) >= 0}, end {after[-1]}")


def states(route: str = 'd3first'):
    """(path, step, keys) of every state of ``route``."""
    import glob  # pylint: disable=import-outside-toplevel
    from triforce._native import Game  # pylint: disable=import-outside-toplevel
    from triforce.entry_states import RouteIndex  # pylint: disable=import-outside-toplevel
    from triforce.paths import STATES_DIR  # pylint: disable=import-outside-toplevel
    index, game = RouteIndex(route), Game(seed=0, states_dir=str(STATES_DIR))
    for path in sorted(glob.glob(os.path.join(STATES_DIR, f'{route}_*.state'))):
        game.load(path, randomize_rng=False, controllable=True)
        yield game, path, index.step_of(path), game.read(KEYS)


def check_or_patch(patch: bool):
    from triforce.events import write_json_atomic  # pylint: disable=import-outside-toplevel
    from triforce.paths import STATES_DIR  # pylint: disable=import-outside-toplevel
    want = ledger({**EVENTS, **CHANGES})
    changed = {}
    for game, path, step, keys in states():
        expected = want[step - 1] if step else 0
        if keys == expected:
            continue
        name = os.path.basename(path)
        print(f"{name} step {step}: {keys} keys, the ledger {expected}" + ('' if step > 100 else ' (left)'))
        if patch and step > 100:
            game.write(KEYS, expected)
            game.save_state_file(path)
            changed[name] = (keys, expected)
    if not changed:
        return
    manifest_path = os.path.join(STATES_DIR, 'entry_states.json')
    with open(manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)
    for entry in manifest:
        if entry.get('file') in changed:
            was, now = changed[entry['file']]
            entry['keys'] = {'was': was, 'now': now, 'by': 'scripts/experiments/r39_keys.py patch (round 39: '
                             "Moldorm's key not taken, 9:0x56's taken)"}
    write_json_atomic(manifest_path, manifest, indent=1)
    print(f"{len(changed)} states set to the ledger")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('what', choices=('ledger', 'check', 'patch'))
    args = parser.parse_args()
    if args.what == 'ledger':
        show()
    else:
        check_or_patch(args.what == 'patch')
    return 0


if __name__ == '__main__':
    sys.exit(main())
