"""Round 26 observation design (docs/observation-next.md): the RAM each proposed feature reads, watched in the game.

    .venv/bin/python scripts/experiments/r26_obs_probes.py [PROBE ...]     # default: every probe

Each probe puts Link into a dungeon room with `triforce.probe.Probe` (no savestate: room id, level block and
Link's position written, see that module), keeps his hearts topped up and prints the bytes a feature would read,
one line per frame on which any of them changes (or every frame where the probe says so). Throwaway in method,
kept because the document cites its output; nothing here is how the game is played.  It plays the game: a job of
the machine's queue (``jobs.require_job``: ``triforce job run ... --timeout 20m -- .venv/bin/python
scripts/experiments/r26_obs_probes.py ...``).

RAM (zelda-asm/src/Variables.inc, ObjVars.inc, CommonVars.inc): ObjType $34F, ObjX $70, ObjY $84, ObjDir $98,
ObjState $AC, ObjTimer $28, ObjMetastate $405, ObjHP $485, ObjInvincibilityTimer $4F0, ObjRemDistance $394,
FrameCounter $15, SwordBlockedLongTimer $4C, IsSwordBlocked $52E, Gohma_ObjEyeState $46B, Gohma_ObjEyeOpenTimer
$444, Gohma_ObjNextOpenEyeCounter $42C, Ganon_ScenePhase $445, Ganon_ObjDyingCount $42C.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True,
                           cwd=Path(__file__).parent, check=True).stdout.strip())
sys.path.insert(0, str(ROOT))
from triforce.jobs import require_job     # noqa: E402  pylint: disable=wrong-import-position
from triforce.probe import Probe          # noqa: E402  pylint: disable=wrong-import-position

TYPE, X, Y, DIR, STATE, TIMER, META, HP, INV, REM = 0x34F, 0x70, 0x84, 0x98, 0xAC, 0x28, 0x405, 0x485, 0x4F0, 0x394
FRAME, SWORD_LONG, SWORD_BLOCKED, LONG_CYCLE = 0x15, 0x4C, 0x52E, 0x26
HEARTS, PARTIAL = 0x66F, 0x670


class Watch:
    """Steps the game a frame at a time with Link's hearts held, printing ``columns(game)`` when it changes."""

    def __init__(self, probe: Probe, hold_hearts=True):
        self.probe, self.game, self.hold = probe, probe.game, hold_hearts
        self.hearts, self.partial = self.game.read(HEARTS), self.game.read(PARTIAL)
        self.frame = 0

    def step(self, button='NONE', before=None):
        if before is not None:
            before(self.game)
        self.game.press(button, 1)
        if self.hold:
            self.game.write(HEARTS, self.hearts)
            self.game.write(PARTIAL, self.partial)
        self.frame += 1

    def run(self, frames, columns, button='NONE', before=None, every=False):
        last = None
        for _ in range(frames):
            self.step(button, before)
            now = columns(self.game)
            if every or now != last:
                print(f'{self.frame:4d} fc {self.game.read(FRAME):3d}  {now}')
            last = now


def slots(game, types):
    """Slots 1-$B whose ObjType is in ``types``."""
    return [s for s in range(1, 0xC) if game.read(TYPE + s) in types]


def red_wizzrobe(probe):
    """6:0x78 (five Red Wizzrobes, $24): ObjState through two cycles, its position while hidden, the $59 shots."""
    probe.enter_room(6, 0x78)
    w = Watch(probe)
    s = slots(w.game, {0x24})[0]

    def columns(g):
        state = g.read(STATE + s)
        group = ('hidden', 'fade-out', 'solid', 'appear/fade-in')[state >> 6]
        shots = len(slots(g, {0x59}))
        return (f'slot {s} state ${state:02X} {group:14s} xy ({g.read(X + s)},{g.read(Y + s)}) dir {g.read(DIR + s)} '
                f'fade ${g.read(REM + s):02X} shots {shots}')
    # Only the group changes and the jumps: print when the group, position or shot count changes.
    last = None
    for _ in range(440):
        w.step()
        g = w.game
        state = g.read(STATE + s)
        key = (state >> 6, g.read(X + s), g.read(Y + s), len(slots(g, {0x59})), state in (0xFF, 0xB0, 0x4F, 0x00))
        if key != last:
            print(f'{w.frame:4d} fc {g.read(FRAME):3d}  {columns(g)}')
        last = key


def blue_wizzrobe(probe):
    """9:0x15 (three Blue Wizzrobes, $23): ObjTimer (walk/stand), ObjRemDistance (teleport), position, $58 shots."""
    probe.enter_room(9, 0x15)
    w = Watch(probe)
    s = slots(w.game, {0x23})[0]

    def columns(g):
        rem = g.read(REM + s)
        timer = g.read(TIMER + s)
        mode = 'teleport' if rem else ('stand' if timer < 0x10 else 'walk')
        return (f'slot {s} timer {timer:3d} rem {rem:2d} {mode:8s} xy ({g.read(X + s)},{g.read(Y + s)}) '
                f'shots {len(slots(g, {0x58}))} fc&1F {g.read(FRAME) & 0x1F}')
    last = None
    for _ in range(600):
        w.step()
        g = w.game
        rem, timer = g.read(REM + s), g.read(TIMER + s)
        key = (bool(rem), timer < 0x10, len(slots(g, {0x58})))
        if key != last:
            print(f'{w.frame:4d} fc {g.read(FRAME):3d}  {columns(g)}')
        last = key


def gohma(probe):
    """6:0x1c (Red Gohma, $34): eye state, open timer, next-open counter, X, through two openings."""
    probe.enter_room(6, 0x1C)
    w = Watch(probe)
    s = slots(w.game, {0x33, 0x34})[0]

    def columns(g):
        return (f'slot {s} eye {g.read(0x46B + s)} open {g.read(0x444 + s):3d} next {g.read(0x42C + s):3d} '
                f'x {g.read(X + s)} y {g.read(Y + s)}')

    last = None
    for _ in range(1200):
        w.step()
        g = w.game
        key = (g.read(0x46B + s), g.read(0x444 + s) == 0)
        if key != last or w.frame % 100 == 0:
            print(f'{w.frame:4d} fc {g.read(FRAME):3d}  {columns(g)}')
        last = key


def bubble(probe):
    """7:0x49 (two Bubbles, $2B): Link put onto a Bubble until it touches him; then SwordBlockedLongTimer ($4C) and
    IsSwordBlocked ($52E) each frame until the sword is back."""
    probe.enter_room(7, 0x49)
    w = Watch(probe)
    for _ in range(120):
        w.step()
        if slots(w.game, {0x2B}):
            break
    s = slots(w.game, {0x2B})[0]
    touched = None
    for _ in range(300):
        def onto(g):
            g.write(X, g.read(X + s))
            g.write(Y, g.read(Y + s))
        w.step(before=onto if touched is None else None)
        if touched is None and w.game.read(SWORD_LONG):
            touched = w.frame
            print(f'{w.frame:4d} touched by slot {s}: $4C {w.game.read(SWORD_LONG)} $52E {w.game.read(SWORD_BLOCKED)}')
            break
    if touched is None:
        print('no touch in 300 frames')
        return
    last = None
    for _ in range(200):
        w.step()
        g = w.game
        key = (g.read(SWORD_LONG), g.read(SWORD_BLOCKED))
        if key != last:
            print(f'{w.frame:4d} (+{w.frame - touched:3d}) $4C {key[0]:2d} $52E {key[1]} cycle $26 {g.read(LONG_CYCLE)}')
        last = key
        if key == (0, 0):
            break


def patra(probe):
    """9:0x61 (Patra $47 and 8 children $25): Patra's ObjState, how many children exist, their states."""
    probe.enter_room(9, 0x61)
    w = Watch(probe)

    def columns(g):
        body = slots(g, {0x47, 0x48})
        kids = slots(g, {0x25, 0x26})
        states = ''.join(str(g.read(STATE + k)) for k in kids)
        b = body[0] if body else None
        return (f'body slot {b} state {g.read(STATE + b) if b else None} meta {g.read(META + b) if b else None} '
                f'children {len(kids)} states {states}')
    w.run(300, columns)


def ganon(probe):
    """9:0x42 (Ganon $3E): scene phase, ObjState (blue 0 / brown), ObjTimer (visible after a hit), position while he
    is invisible, HP, dying count. Then two phases entered by writing what a hit writes (marked WRITE): ObjTimer $40
    (a sword hit, asm:Ganon_CheckCollisions) and ObjState $FF (his HP run out: brown)."""
    # The room stays dark: the entry scene ends when its fade to light does (Ganon_ScenePhase0_LiftInDark); a room
    # lit beforehand never fades and Link stays halted holding the piece.
    probe.enter_room(9, 0x42, lit=False)
    w = Watch(probe)
    s = slots(w.game, {0x3E})[0]

    def columns(g):
        return (f'scene {g.read(0x445)} state {g.read(STATE + s):3d} timer {g.read(TIMER + s):2d} '
                f'xy ({g.read(X + s)},{g.read(Y + s)}) hp ${g.read(HP + s):02X} dying {g.read(0x42C + s)} '
                f'meta {g.read(META + s)} link-state ${g.read(STATE):02X} fireballs {len(slots(g, {0x56}))}')

    def watch(frames, every):
        last = None
        for _ in range(frames):
            w.step()
            g = w.game
            key = (g.read(0x445), g.read(STATE + s) > 0, g.read(TIMER + s) > 0, len(slots(g, {0x56})))
            if key != last or w.frame % every == 0:
                print(f'{w.frame:4d} fc {g.read(FRAME):3d}  {columns(g)}')
            last = key
    watch(420, 20)
    print(f'{w.frame:4d} WRITE ObjTimer ${0x40:02X} (what a sword hit sets)')
    w.game.write(TIMER + s, 0x40)
    watch(80, 8)
    print(f'{w.frame:4d} WRITE ObjState $FF (brown: what the 4th hit sets)')
    w.game.write(STATE + s, 0xFF)
    watch(540, 32)


def lanmola(probe):
    """9:0x55 (two red Lanmolas $3A, slots 1-$A): each segment's ObjDir (0 = not emerged) until all are out."""
    probe.enter_room(9, 0x55)
    w = Watch(probe)

    def columns(g):
        return ' '.join(f'{s}:{g.read(DIR + s)}' for s in slots(g, {0x3A, 0x3B}))
    last = None
    for _ in range(400):
        w.step()
        now = columns(w.game)
        emerged = sum(1 for part in now.split() if part.split(':')[1] != '0')
        if emerged != last:
            print(f'{w.frame:4d} fc {w.game.read(FRAME):3d}  emerged {emerged}  {now}')
        last = emerged


def goriya_boomerang(probe):
    """7:0x1b (six Blue Goriyas, $05): a Goriya's ObjState while its boomerang ($5C) flies, and the boomerang's
    ObjState out and back (a shot harms only in states $10-$1F, asm:Link_CheckMonsterTouch_NoReset)."""
    probe.enter_room(7, 0x1B)
    w = Watch(probe)

    def columns(g):
        gor = [(s, g.read(STATE + s)) for s in slots(g, {0x05})]
        boom = [(s, g.read(STATE + s)) for s in slots(g, {0x5C})]
        return f'goriyas {gor} boomerangs {boom}'
    w.run(400, columns)


def manhandla(probe):
    """8:0x6e (Manhandla $3C): which slot is the base (its HP and invincibility timer every frame)."""
    probe.enter_room(8, 0x6E)
    w = Watch(probe)

    def columns(g):
        return [(s, g.read(HP + s) >> 4, g.read(INV + s), g.read(X + s), g.read(Y + s)) for s in slots(g, {0x3C})]
    w.run(5, columns, every=True)


def wallmaster(probe):
    """7:0x0d (five Wallmasters $27): ObjState (0 in the wall, 1 on a trip), position in the wall, slot 1's shared
    ObjTimer; Link stands still at the W wall, as he enters (captured after ~161 frames on the walk)."""
    probe.enter_room(7, 0x0D, entry='W')
    w = Watch(probe)

    def columns(g):
        hands = [(s, g.read(STATE + s), g.read(X + s), g.read(Y + s)) for s in slots(g, {0x27})]
        return (f'mode {g.read(0x12)} room ${g.read(0xEB):02X} link ({g.read(X)},{g.read(Y)}) timer1 '
                f'{g.read(TIMER + 1)} hands (slot, state, x, y) {hands}')
    last = None
    for _ in range(300):
        w.step()
        g = w.game
        key = (g.read(0x12), g.read(0xEB), tuple(g.read(STATE + s) for s in slots(g, {0x27})),
               g.read(TIMER + 1) in (0, 0x60))
        if key != last:
            print(f'{w.frame:4d} fc {g.read(FRAME):3d}  {columns(g)}')
        last = key


# ------------------------------------------------------------------ round 26 build: the code-only rows, walked
# Each holds Link on one object every frame (his invincibility timer and knockback cleared before the frame, his
# hearts held), counts the frames the game harmed him (Shove_Begin sets his invincibility timer: nonzero after the
# frame) and the frames the object was drawn (sprites of the OAM buffer $200 in its 16x16 box), and prints them per
# phase beside what the schema 6 observation says (`_native.observe`: touch_harms, weapon_target, visible, phase).

INV_CLOCK, SHOVE_DIR, SHOVE_DIST, OAM = 0x66C, 0xC0, 0xD3, 0x200


def drawn(g, s):
    """Sprites in the OAM buffer within slot ``s``'s 16x16 box (Y on screen is the OAM byte + 1), Link's (sprite
    palette 0, where Link is held) left out."""
    x, y = g.read(X + s), g.read(Y + s)
    n = 0
    for i in range(64):
        sy, attr, sx = g.read(OAM + 4 * i) + 1, g.read(OAM + 4 * i + 2), g.read(OAM + 4 * i + 3)
        if sy < 0xF0 and attr & 3 and x - 1 <= sx <= x + 16 and y - 2 <= sy <= y + 16:
            n += 1
    return n


def features(g, s):
    """(touch_harms, weapon_target, visible) of slot ``s`` in the schema 6 observation."""
    from triforce._native import ENTITY_FEATURES, observe  # pylint: disable=import-outside-toplevel
    row = observe(g.state)['entity_features'][s - 1]
    return tuple(round(float(row[ENTITY_FEATURES.index(n)]), 3)
                 for n in ('touch_harms', 'weapon_target', 'visible'))


class Touch:
    """Holds Link on slot ``s`` frame by frame; ``run`` tallies harm and drawing per ``phase(game)`` key."""

    def __init__(self, probe, s, before=None):
        self.w, self.s, self.before = Watch(probe), s, before
        self.rows = {}

    def frame(self, phase):
        g = self.w.game
        key = (phase(g), features(g, self.s))

        def onto(game):
            if self.before:
                self.before(game)
            game.write(X, game.read(X + self.s))
            game.write(Y, game.read(Y + self.s))
            for addr in (INV, SHOVE_DIR, SHOVE_DIST):
                game.write(addr, 0)
        self.w.step(before=onto)
        # Link_HarmByObjType starts his knockback (Shove_Begin); his invincibility timer is no test under the clock
        # (Mode5Play_UpdateWorld adds $10 to it every frame then).
        harmed = g.read(SHOVE_DIR) != 0 or g.read(SHOVE_DIST) != 0
        seen = drawn(g, self.s) > 0
        row = self.rows.setdefault(key, [0, 0, 0])
        row[0] += 1
        row[1] += harmed
        row[2] += seen

    def run(self, frames, phase):
        for _ in range(frames):
            if self.w.game.read(TYPE + self.s) == 0:
                break
            self.frame(phase)
        print('  phase | observed (touch_harms, weapon_target, visible) | frames | harmed | drawn')
        for (key, obs), (n, harmed, seen) in self.rows.items():
            print(f'  {key} | {obs} | {n} | {harmed} | {seen}')


def no_shots(g):
    """Clears magic shots and fireballs, so that a harm is the object's touch."""
    for s in range(1, 0xC):
        if g.read(TYPE + s) in (0x56, 0x58, 0x59):
            g.write(TYPE + s, 0)


def red_wizzrobe_touch(probe):
    """6:0x78: Link held on a Red Wizzrobe through two cycles: harm and drawing per state group (solid, fading in
    and out, hidden, choosing); then the same with the clock held (InvClock := 1)."""
    probe.enter_room(6, 0x78)
    s = slots(probe.game, {0x24})[0]
    groups = {0: 'hidden', 1: 'fade-out', 2: 'solid', 3: 'fade-in'}
    t = Touch(probe, s, no_shots)
    t.run(420, lambda g: 'choosing $FF' if g.read(STATE + s) == 0xFF else groups[g.read(STATE + s) >> 6])
    print('  clock held:')
    t = Touch(probe, s, lambda g: (no_shots(g), g.write(INV_CLOCK, 1)))
    t.run(120, lambda g: groups[g.read(STATE + s) >> 6])


def blue_wizzrobe_touch(probe):
    """9:0x15: Link held on a Blue Wizzrobe flashing (its invincibility timer held at $10): harm and drawing walking,
    standing and teleporting."""
    probe.enter_room(9, 0x15)
    s = slots(probe.game, {0x23})[0]

    def flash(g):
        no_shots(g)
        g.write(INV + s, 0x10)

    def phase(g):
        return 'teleport' if g.read(REM + s) else ('stand' if g.read(TIMER + s) < 0x10 else 'walk')
    Touch(probe, s, flash).run(500, phase)


def park(g, s, objs):
    """Moves slot ``s`` (WRITE) to a floor spot at least 40 px (Chebyshev) from every object of ``objs`` (slots) unless
    it is already: its own spot first, then four fixed candidates."""
    def far(x, y):
        return all(max(abs(x - g.read(X + o)), abs(y - g.read(Y + o))) >= 40 for o in objs)
    if far(g.read(X + s), g.read(Y + s)):
        return
    for x, y in ((0x30, 0x60), (0xC0, 0x60), (0x30, 0xB0), (0xC0, 0xB0)):
        if far(x, y):
            g.write(X + s, x)
            g.write(Y + s, y)
            return
    raise RuntimeError('no spot 40 px from every object')


def patra_touch(probe):
    """9:0x61: Link held on one Patra child per run (slots 9 and 2) from the frame the children are made: harm and
    drawing waiting (state 0), out before Patra's state 1, and after. A waiting child's position is read by nothing
    (asm:Patra_UpdateChild sets it on appearing), so it is WRITTEN, if needed, to a spot 40 px from Patra and the
    orbiting children each frame: a harm there is the waiting child's own; an orbiting child is $18-$2C px from
    Patra's middle."""
    for slot in (9, 2):
        print(f'  child in slot {slot}:')
        probe.enter_room(9, 0x61)
        g = probe.game
        t = Touch(probe, slot)
        while g.read(TYPE + slot) not in (0x25, 0x26):
            t.w.step()

        def hold(game, slot=slot):
            if game.read(STATE + slot) == 0:
                park(game, slot, [o for o in range(1, 10) if o != slot and (
                    game.read(TYPE + o) in (0x47, 0x48) or game.read(STATE + o) == 1)])
        t.before = hold
        t.run(160, lambda g, slot=slot: f'child state {g.read(STATE + slot)} patra state {g.read(STATE + 1)}')


def lanmola_touch(probe):
    """9:0x55: Link held on a Lanmola tail (slot 1, the last segment set in motion) from the first frame: harm and
    drawing before and after it moves (ObjDir 0 = not yet). A segment in direction 0 is read by nothing
    (asm:Lanmola_Update returns at once) and starts from its own position, so it is WRITTEN, if needed, to a spot 40
    px from the moving segments each frame: a harm there is the waiting segment's own; once it moves, Link stays on
    it."""
    probe.enter_room(9, 0x55)

    def hold(g):
        if g.read(DIR + 1) == 0:
            park(g, 1, [o for o in range(2, 0xB) if g.read(DIR + o) != 0])
    Touch(probe, 1, hold).run(90, lambda g: f'dir {"0" if g.read(DIR + 1) == 0 else "set"}')


def ganon_dying(probe):
    """9:0x42: the fight, then Ganon_ObjDyingCount := 1 (WRITE: what the silver arrow sets) with Link held on him:
    harm and drawing while the count runs to $A0, and when the room item appears."""
    probe.enter_room(9, 0x42, lit=False)
    g = probe.game
    s = slots(g, {0x3E})[0]
    w = Watch(probe)
    while g.read(0x445) < 2:
        w.step()
    w.run(10, lambda g: '')
    g.write(0x42C + s, 1)
    Touch(probe, s).run(200, lambda g: f'dying ${g.read(0x42C + s) & 0xF0:02X}+ item {g.read(0x34F + 0x13)}')


def ringleader(probe):
    """6:0x78 (secret trigger 2): slot 1 emptied (WRITE: its death's end); the other Red Wizzrobes' metastate."""
    from triforce._native import observe  # pylint: disable=import-outside-toplevel
    probe.enter_room(6, 0x78)
    g = probe.game
    w = Watch(probe)
    w.run(20, lambda g: '')
    print('  observed ringleader by slot 1-6:', [float(r[-1]) for r in observe(g.state)['entity_features'][:6]])
    g.write(TYPE + 1, 0)
    w.run(4, lambda g: [(s, g.read(META + s)) for s in slots(g, {0x24})], every=True)


def wallmaster_lure(probe):
    """7:0x0d: Link walked in from W stands at ($20,$8D) still, or is walked one pixel on (Right 2 frames: ($21,$8D),
    grid offset 1) and stands there: per spot, frames by (Link after the frame, observed `wallmaster_lure`, slot 1's
    shared timer was 0 before the frame, a hand came out). (Writing Link's X without his grid offset $394 does not
    hold: the game moves him back onto the wall, so the spot is walked, not written.)"""
    from triforce._native import ROOM_STATE_FEATURES, observe  # pylint: disable=import-outside-toplevel
    lure_at = ROOM_STATE_FEATURES.index('wallmaster_lure')
    for presses in (0, 2):
        probe.enter_room(7, 0x0D, entry='W')
        w = Watch(probe)
        g = w.game
        rows = {}
        for frame in range(300):
            before = {s for s in slots(g, {0x27}) if g.read(STATE + s) == 1}
            timer = g.read(TIMER + 1)
            w.step('RIGHT' if frame < presses else 'NONE')
            out = {s for s in slots(g, {0x27}) if g.read(STATE + s) == 1} - before
            lure = float(observe(g.state)['room_state'][lure_at])
            key = (f'(${g.read(X):02X},${g.read(Y):02X})', lure, timer == 0, bool(out))
            rows[key] = rows.get(key, 0) + 1
        print(f'  Right {presses} frames: (Link after the frame, observed lure, timer was 0, a hand came out) -> frames')
        for key, n in sorted(rows.items()):
            print(f'    {key} -> {n}')


def bubble_block(probe):
    """7:0x49: the Bubble's touch, then `sword_blocked` and `sword_blocked_time` each frame until the sword is back."""
    from triforce._native import LINK_FEATURES, observe  # pylint: disable=import-outside-toplevel
    probe.enter_room(7, 0x49)
    w = Watch(probe)
    g = w.game
    for _ in range(120):
        w.step()
        if slots(g, {0x2B}):
            break
    s = slots(g, {0x2B})[0]
    for _ in range(300):
        w.step(before=lambda game: (game.write(X, game.read(X + s)), game.write(Y, game.read(Y + s))))
        if g.read(SWORD_LONG):
            break
    touched, last = w.frame, None
    for _ in range(200):
        link = observe(g.state)['link']
        now = (g.read(SWORD_LONG), float(link[LINK_FEATURES.index('sword_blocked')]),
               round(float(link[LINK_FEATURES.index('sword_blocked_time')]), 4))
        if now != last:
            print(f'  +{w.frame - touched:3d}  $4C {now[0]:2d}  sword_blocked {now[1]}  sword_blocked_time {now[2]}')
        last = now
        if now[0] == 0:
            break
        w.step()


PROBES = {f.__name__.replace('_', '-'): f for f in
          (red_wizzrobe, blue_wizzrobe, gohma, bubble, patra, ganon, lanmola, goriya_boomerang, manhandla, wallmaster,
           red_wizzrobe_touch, blue_wizzrobe_touch, patra_touch, lanmola_touch, ganon_dying, ringleader,
           wallmaster_lure, bubble_block)}


def main(argv):
    names = argv or list(PROBES)
    for name in names:
        if name not in PROBES:
            raise SystemExit(f'unknown probe {name}; known: {", ".join(PROBES)}')
    require_job()
    for name in names:
        print(f'== {name}: {PROBES[name].__doc__.split(chr(10))[0]}')
        PROBES[name](Probe(ROOT / 'triforce' / 'states'))
        print()


if __name__ == '__main__':
    main(sys.argv[1:])
