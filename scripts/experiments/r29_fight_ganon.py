"""Round 29: Ganon (9:0x42), the Triforce of Power and the walk to Zelda (9:0x32) measured and played in the running
game (docs/routes/d9-fights-walks.md items 4 and 5).  Link comes in by the S shutter (`r29_fight_walks.d9_state`: 15
containers, the magic sword, bow, silver arrow, 10 rupees); every trial loads that state with the game's own
randomness for its seed.  The B item is set by RAM (SelectedItemSlot: what the sub-screen would set).

    python scripts/experiments/r29_fight_ganon.py scene [--seeds N]     # the entry scene, with 8 and with 7 pieces
    python scripts/experiments/r29_fight_ganon.py moves [--seeds N]     # his walk while invisible, the fireballs, the touch
    python scripts/experiments/r29_fight_ganon.py sword [--seeds N]     # where a stab finds him, the four hits, the jump
    python scripts/experiments/r29_fight_ganon.py brown [--seeds N]     # the brown phase: its length, the arrow, a miss
    python scripts/experiments/r29_fight_ganon.py kill [--seeds N] [--eyes ram|screen|tells|sweep]   # the scripted kills
    python scripts/experiments/r29_fight_ganon.py end [--seeds N]       # from his room's door to game mode $13
    python scripts/experiments/r29_fight_ganon.py death                 # Link dies in the room: what follows
"""

import argparse
import collections
import statistics

from r29_fight_walks import (ARROW_STATE, DIR_OF, FIREBALL, FRAME_COUNTER, GANON, GUARD_FIRE, HEART_PARTIAL, HEARTS,
                             INV_POWER_TRIFORCE, LINK_INVINC, MODE, OBJ_DIR, OBJ_HP, OBJ_INVINC, OBJ_META, OBJ_STATE,
                             OBJ_TIMER, OBJ_TYPE, OBJ_X, OBJ_Y, PLAY, REACH, RUPEES, SELECTED_SLOT, SWORD_STATE, Bot,
                             d9_state, hp, in_reach, load, summary)

SCENE, DYING, TURN_COUNTER = 0x445, 0x42C + 1, 0x412 + 1       # Ganon_ScenePhase, Ganon_ObjDyingCount, the turn counter
OPEN_DOORS, ITEM_SLOT = 0xEE, 0x13                             # CurOpenedDoors; the room item's object slot
VEC = {1: (1, 0), 2: (-1, 0), 4: (0, 1), 8: (0, -1), 0: (0, 0)}
JUMP_SPOTS = ((0x30, 0xA0), (0xB0, 0xA0))                      # asm:Ganon_StartXs, asm:Ganon_RandomizeLocation
G = 1                                                          # his object slot
FIREBALL_WAIT = 16                                             # frames a fireball stands before it flies (`moves`)


def state(room=0x42, ring=0, hearts=15, **extra):
    inv = {'rupees': 10}
    inv.update(extra)
    return d9_state(room, 'S', ring=ring, hearts=hearts, silver=inv.pop('silver', True), extra=inv)


def to_fight(w, limit=400) -> int:
    """Waits out the entry scene; the frames until Link can move (scene phase 2)."""
    for f in range(limit):
        if w.r(SCENE) == 2 and w.r(OBJ_STATE) == 0:
            return f
        w.press('NONE', 1)
    return -1


def phase(w) -> str:
    if w.r(DYING):
        return 'dying'
    if w.r(OBJ_STATE + G):
        return 'brown'
    return 'shown' if w.r(OBJ_TIMER + G) else 'hidden'


class GanonBot(Bot):
    """The fighter for Ganon.  For the sword's box and for his touch he is a 16-pixel monster at his position + 8
    on both axes (his middle is +16, asm:Ganon_CheckCollisions).  `eyes` says what the bot knows of him while he is
    not drawn: 'ram' his position every frame (ObjX/ObjY, his direction and turn counter); 'screen' only what a
    player sees: where he stood when he was last drawn, where each fireball is born (4 pixels right of him) and the
    two spots he jumps to after a hit ('screen' walks to every birth; 'tells' stays by the spot it believes in
    and takes a birth for his place for 12 frames only); 'sweep' the same without the fireballs' births."""

    touch = 12

    def __init__(self, w, eyes='ram'):
        self.eyes = eyes
        self.belief, self.belief_age, self.seen_balls, self.tried = None, 0, set(), 0
        self.last_phase, self.guess, self.aim, self.tell, self.tell_age = None, 0, None, None, 99
        super().__init__(w, {GANON})

    # ---- what the bot knows of him
    def _track(self):
        super()._track()
        w = self.w
        if w.r(OBJ_TYPE + G) != GANON:
            return
        ph = phase(w)
        pos = (w.r(OBJ_X + G), w.r(OBJ_Y + G))
        self.belief_age += 1
        if ph != 'hidden' or self.belief is None:
            self.belief, self.belief_age = pos, 0                # he is drawn (or the scene has just shown him)
        elif self.last_phase == 'shown':
            # he has just jumped: one of two spots, the bot cannot see which; it takes the nearer, then the other
            near = sorted(JUMP_SPOTS, key=lambda s: abs(s[0] - w.x) + abs(s[1] - w.y))
            self.belief, self.belief_age, self.guess, self.tried = near[0], 0, 0, 0
            self.spots = near
        self.tell_age += 1
        if self.eyes in ('screen', 'tells'):
            for s in range(2, 0x0C):
                if w.r(OBJ_TYPE + s) == FIREBALL and s not in self.seen_balls:
                    birth = (w.r(OBJ_X + s) - 4, w.r(OBJ_Y + s))
                    if self.eyes == 'screen':                    # the chaser: the belief moves to every birth
                        self.belief, self.belief_age, self.spots = birth, 0, None
                    else:                                        # the waiter: a birth is believed for 12 frames
                        self.tell, self.tell_age = birth, 0
            self.seen_balls = {s for s in range(2, 0x0C) if w.r(OBJ_TYPE + s) == FIREBALL}
        self.last_phase = ph

    def where(self):
        """His position as the bot knows it."""
        if self.eyes == 'ram':
            return self.w.r(OBJ_X + G), self.w.r(OBJ_Y + G)
        if self.eyes == 'tells' and self.tell_age <= 12 and phase(self.w) == 'hidden':
            return self.tell
        return self.belief

    def future(self, s, k):
        if s != G:
            return super().future(s, k)
        x, y = self.where()
        if self.eyes == 'ram' and phase(self.w) == 'hidden':
            dx, dy = VEC.get(self.w.r(OBJ_DIR + G), (0, 0))
            x, y = x + dx * k, y + dy * k
        return x + 8, y + 8

    def spread(self, s, k):
        if s != G and self.prev[s][0] == FIREBALL and self.vel.get(s, (0, 0)) == (0, 0):
            # a fireball stands 16 frames where it is born, then flies at Link at up to 2 pixels a frame
            return max(0, 2 * (k - max(0, FIREBALL_WAIT - (self.frames - self.born.get(s, self.frames)))))
        if s != G or phase(self.w) != 'hidden':
            return 0
        if self.eyes == 'ram':
            to_turn = (64 - (self.w.r(TURN_COUNTER) & 0x3F)) % 64
            return max(0, k - to_turn)
        return 0                                                 # a belief has no error bar the bot could know

    def hittable(self, s):
        return phase(self.w) == 'hidden' and self.w.r(SCENE) == 2

    def harms(self, s):
        if s == G:
            return self.w.r(SCENE) == 2 and not self.w.r(DYING)
        return super().harms(s)

    # ---- the blue phase
    def try_stab(self):
        if self.eyes == 'ram' or phase(self.w) != 'hidden':
            return super().try_stab()
        # by belief: stab when the believed spot is in the sword's box; a stale belief is swept every 20 frames
        w = self.w
        fresh = self.belief_age <= 12 or (self.eyes == 'tells' and self.tell_age <= 12)
        if not fresh and self.frames % 20:
            return False
        ok = super().try_stab()
        if ok and self.w.r(SWORD_STATE):
            self.tried += 1
            if getattr(self, 'spots', None) and self.tried >= 2 and self.guess == 0:
                self.belief, self.guess, self.tried = self.spots[1], 1, 0        # not at the first spot: the other
        return ok

    # ---- the brown phase
    def arrow_spots(self):
        """The squares on his row or column from which an arrow reaches his middle (within $0B, the arrow's point at
        Link + (8, 6) flying sideways, + (6, 8) flying up or down), two to five squares away."""
        gx, gy = self.w.r(OBJ_X + G), self.w.r(OBJ_Y + G)
        out = {}
        for node in self.graph:
            dx, dy = node[0] - gx, node[1] - gy
            if abs(node[1] + 6 - (gy + 16)) < 9 and 40 <= abs(dx - 8) <= 104:
                out[node] = 'LEFT' if dx > 0 else 'RIGHT'
            elif abs(node[0] + 6 - (gx + 16)) < 9 and 40 <= abs(dy - 8) <= 88:
                out[node] = 'UP' if dy > 0 else 'DOWN'
        return out

    def shoot(self) -> bool:
        """One frame of the brown phase: to an arrow spot, turn, B.  True when an arrow was fired this frame."""
        w = self.w
        w.game.write(SELECTED_SLOT, 2)
        if not self.can_act() or w.r(ARROW_STATE):
            self.press('NONE')
            return False
        spots = self.arrow_spots()
        here = self.node()
        if here in spots:
            self.aim = spots[here]
        if self.aim:                                             # on a spot (or the turn's pixel off it): turn, B
            if w.r(OBJ_DIR) != DIR_OF[self.aim]:
                self.press(self.aim)
                return False
            self.press('B')
            self.aim = None
            return True
        if here is None:
            self.heading = self.off_grid()
            self.press(self.heading)
            return False
        step = self.plan(set(spots)) or 'NONE'
        if step != 'NONE':
            self.heading = step
        self.press(step)
        return False


def fight(w, eyes='ram', limit=12000):
    """Scene, sword phase, arrow: returns (bot, killed, counters)."""
    wait = to_fight(w)
    bot = GanonBot(w, eyes)
    bot.frames += wait
    info = collections.Counter()
    last = 'hidden'
    while bot.frames < limit and bot.alive() and not w.r(DYING):
        ph = phase(w)
        if ph == 'brown' and last != 'brown':
            info['browns'] += 1
        if (ph == 'shown' and last == 'hidden') or (ph == 'brown' and last != 'brown'):
            info['sword hits'] += 1                              # the fourth one turns him brown at once
        last = ph
        if ph == 'brown':
            if bot.shoot():
                info['arrows'] += 1
        elif ph == 'shown':
            bot.move_shown()
        else:
            bot.fight_frame()
    return bot, bool(w.r(DYING)) and bot.alive(), info


def _move_shown(self):
    """While he stands shown after a hit: to the square from which a stab reaches the spot he will jump to ('ram':
    the frame counter's parity when his timer runs out says which; else the nearer one)."""
    w = self.w
    if not self.can_act():
        self.press('NONE')
        return
    if self.eyes == 'ram':
        parity = (w.r(FRAME_COUNTER) + w.r(OBJ_TIMER + G) - 1) & 1
        spot = JUMP_SPOTS[parity]
    else:
        spot = min(JUMP_SPOTS, key=lambda s: abs(s[0] - w.x) + abs(s[1] - w.y))
    goals = {n for n in self.graph if any(in_reach(b, n[0], n[1], spot[0] + 8, spot[1] + 8, 4, 12) for b in REACH)
             and not self.danger(n)}
    if self.node() is None:
        self.heading = self.off_grid()
        self.press(self.heading)
        return
    step = self.plan(goals)
    if step is None:
        step = self.flee() if self.danger(self.node()) else 'NONE'
    if step != 'NONE':
        self.heading = step
    self.press(step)


GanonBot.move_shown = _move_shown


def kill(args):
    for eyes in args.eyes.split(','):
        for ring, hearts in ((0, 15), (0, 8), (2, 15), (2, 8)):
            rows, infos = [], collections.defaultdict(list)
            for seed in range(args.seeds):
                w = load(state(ring=ring, hearts=hearts), seed)
                r0 = w.r(RUPEES)
                bot, ok, info = fight(w, eyes)
                rows.append((ok, bot.lost(), bot.frames, bot.by))
                for key in ('sword hits', 'browns', 'arrows'):
                    infos[key].append(info[key])
                infos['swings'].append(bot.swings)
                infos['rupees'].append(r0 - w.r(RUPEES))
            print(f'Ganon by rule, eyes {eyes}, {"red ring" if ring else "no ring"}, from {hearts} hearts '
                  f'({args.seeds} seeds; frames from the door, the 272 of the scene included): {summary(rows)}; ' +
                  ', '.join(f'{k} median {statistics.median(v):g} max {max(v)}' for k, v in infos.items()))


# ---- the facts -------------------------------------------------------------------------------------------------------

OAM, LIFT_ITEM, LIFT_TIMER = 0x200, 0x505, 0x506               # the sprite buffer; ItemTypeToLift, ItemLiftTimer


def drawn(w, size=32) -> int:
    """Sprites of the OAM buffer inside his box that are not Link's (sprite palette 0)."""
    x, y = w.r(OBJ_X + G), w.r(OBJ_Y + G)
    n = 0
    for i in range(64):
        sy, attr, sx = w.r(OAM + 4 * i) + 1, w.r(OAM + 4 * i + 2), w.r(OAM + 4 * i + 3)
        if sy < 0xF0 and attr & 3 and x - 1 <= sx <= x + size and y - 2 <= sy <= y + size:
            n += 1
    return n


def hold(w, hearts=0xEE):
    w.game.write(HEARTS, hearts)
    w.game.write(HEART_PARTIAL, 0xFF)


def put(w, x, y, facing=None):
    w.game.apply_ram({'link_x': x, 'link_y': y})
    w.game.write(0x394, 0)
    if facing:
        w.game.write(OBJ_DIR, DIR_OF[facing])


def scene(args):
    for pieces in (0xFF, 0x7F):
        rows = collections.Counter()
        for seed in range(args.seeds):
            w = load(state(triforce=pieces), seed)
            first = (w.x, w.y)
            log = {}
            lifted, shut, seen = set(), None, collections.Counter()
            for f in range(400):
                ph = (w.r(SCENE), w.r(OBJ_STATE))
                log.setdefault(ph, f)
                if w.r(LIFT_ITEM):
                    lifted.add(w.r(LIFT_ITEM))
                if shut is None and f and w.r(OPEN_DOORS) == 0:
                    shut = f
                if w.r(SCENE) < 2:
                    seen[(w.r(SCENE), drawn(w) > 0)] += 1
                if ph == (2, 0):
                    break
                w.press('NONE', 1)
            rows[f'Link in at (${first[0]:02X},${first[1]:02X}); halted (ObjState $40) from frame {log.get((0, 0x40))}; '
                 f'scene 1 from {log.get((1, 0x40))}; the fight (scene 2, Link free) from {log.get((2, 0))}; holds '
                 f'up item {sorted(f"${i:02X}" for i in lifted)}; doors shut (CurOpenedDoors 0) from frame {shut}; '
                 f'Ganon at (${w.r(OBJ_X + G):02X},${w.r(OBJ_Y + G):02X}), frames (scene, drawn): '
                 f'{dict(sorted(seen.items()))}'] += 1
        print(f'entry scene with InvTriforce ${pieces:02X} ({args.seeds} seeds):')
        for k, v in rows.items():
            print(f'   {v}: {k}')


def moves(args):
    # 1. his walk, Link still where the scene leaves him, hearts held
    first, steps, turn_gaps, axes, span, shown, counter = (collections.Counter(), collections.Counter(),
                                                           collections.Counter(), collections.Counter(), [],
                                                           collections.Counter(), collections.Counter())
    births, offsets, waits, speeds, lives = (collections.Counter(), collections.Counter(), collections.Counter(),
                                             collections.Counter(), [])
    for seed in range(args.seeds):
        w = load(state(), seed)
        t0 = to_fight(w)
        pos, d, last_turn, moved = (w.r(OBJ_X + G), w.r(OBJ_Y + G)), w.r(OBJ_DIR + G), None, None
        xs, ys, balls = [], [], {}
        for f in range(1500):
            hold(w)
            w.game.write(LINK_INVINC, 2)                         # nothing knocks him off his spot
            w.press('NONE', 1)
            now, nd = (w.r(OBJ_X + G), w.r(OBJ_Y + G)), w.r(OBJ_DIR + G)
            step = abs(now[0] - pos[0]) + abs(now[1] - pos[1])
            steps[step] += 1
            shown[drawn(w) > 0] += 1
            if step and moved is None:
                moved = f
                first[(f, {1: 'E', 2: 'W', 4: 'S', 8: 'N'}.get(nd, nd))] += 1
            if nd != d:
                if (w.r(TURN_COUNTER) & 0x3F) == 0:
                    if last_turn is not None:
                        turn_gaps[f - last_turn[0]] += 1
                        axes['other axis' if (nd in (1, 2)) != last_turn[1] else 'same axis'] += 1
                    last_turn = (f, nd in (1, 2))
                    counter['turn with the counter at a multiple of $40'] += 1
                else:
                    counter['reversal at a wall'] += 1
            pos, d = now, nd
            xs.append(now[0])
            ys.append(now[1])
            for s in range(2, 0x0C):
                if w.r(OBJ_TYPE + s) == FIREBALL:
                    p = (w.r(OBJ_X + s), w.r(OBJ_Y + s))
                    if s not in balls:
                        balls[s] = [f, p, p, None]
                        births[w.r(FRAME_COUNTER) & 0x3F] += 1
                        offsets[(p[0] - now[0], p[1] - now[1])] += 1
                    else:
                        if p != balls[s][2] and balls[s][3] is None:
                            balls[s][3] = f
                            waits[f - balls[s][0]] += 1
                        elif balls[s][3] is not None and f - balls[s][3] == 16:
                            q = balls[s][1]
                            speeds[round(((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5 / 16, 1)] += 1
                        balls[s][2] = p
                elif s in balls:
                    lives.append(f - balls[s][0])
                    del balls[s]
        span.append((min(xs), max(xs), min(ys), max(ys)))
    print(f'Ganon while hidden, Link still at the door spot, hearts held ({args.seeds} seeds x 1500 frames of fight):')
    print(f'   first step (frame of the fight, direction): {dict(first)}; px moved per frame {dict(steps)}; frames '
          f'with a sprite of his drawn: {dict(shown)}')
    print(f'   direction changes: {dict(counter)}; frames between turns {dict(turn_gaps)}; a turn is on the '
          f'{dict(axes)}; his X {min(s[0] for s in span):02X}-{max(s[1] for s in span):02X}, Y '
          f'{min(s[2] for s in span):02X}-{max(s[3] for s in span):02X}')
    print(f'   fireballs: FrameCounter & $3F at birth {dict(births)}; born at (his X, Y) + {dict(offsets)}; frames '
          f'standing before the first move {dict(waits)}; px a frame over the next 16 frames '
          f'{dict(sorted(speeds.items()))}; frames alive {min(lives)}-{max(lives)}')
    # 2. damage: a fireball on a still Link, and his touch (Link written onto his middle), by ring
    for ring in (0, 1, 2):
        ball, touch = collections.Counter(), collections.Counter()
        for seed in range(args.seeds):
            w = load(state(ring=ring), seed)
            to_fight(w)
            for f in range(600):
                h = hp(w)
                w.press('NONE', 1)
                if hp(w) < h:
                    near = min(((abs(w.r(OBJ_X + s) - w.x) + abs(w.r(OBJ_Y + s) - w.y), w.r(OBJ_TYPE + s))
                                for s in range(1, 0x0C) if w.r(OBJ_TYPE + s)), default=(0, 0))[1]
                    (ball if near == FIREBALL else touch)[round(h - hp(w), 3)] += 1
                    hold(w)
            for label in ('hidden', 'shown'):
                w = load(state(ring=ring), seed)
                to_fight(w)
                w.press('NONE', 70)
                if label == 'shown':
                    w.game.write(OBJ_TIMER + G, 0x30)            # WRITTEN: as after a sword hit
                hold(w)
                w.game.write(LINK_INVINC, 0)
                h = hp(w)
                put(w, w.r(OBJ_X + G) + 8, w.r(OBJ_Y + G) + 8)
                w.press('NONE', 2)
                touch[(label, round(h - hp(w), 3))] += 1
        print(f'   ring {ring} ({("none", "blue", "red")[ring]}): fireball on a still Link, hearts a hit {dict(ball)}; '
              f'his touch {dict(touch)}')


FACE = ('RIGHT', 'LEFT', 'UP', 'DOWN')


def sword(args):
    # 1. where a stab finds him: in the first 64 frames of the fight he stands still, hidden, at ($A0,$5D).  Link is
    #    WRITTEN to every offset of a grid with each facing, A, 14 frames; a hit shows him (his timer $40).
    w = load(state(hearts=14.5), 0)
    to_fight(w)
    gx, gy = w.r(OBJ_X + G), w.r(OBJ_Y + G)
    snap = w.game.snapshot()
    box = {}
    for facing in FACE:
        hits = []
        for dx in range(-48, 49, 1 if args.fine else 4):
            for dy in range(-48, 49, 4):
                w.game.restore(snap)
                x, y = gx + dx, gy + dy
                if not (0x10 <= x <= 0xE0 and 0x40 <= y <= 0xD0):
                    continue
                put(w, x, y, facing)
                w.game.write(LINK_INVINC, 0x10)
                w.press('A', 1)
                for _ in range(13):
                    w.game.write(LINK_INVINC, 0x10)
                    w.press('NONE', 1)
                if w.r(OBJ_TIMER + G):
                    hits.append((dx, dy))
        box[facing] = (min(h[0] for h in hits), max(h[0] for h in hits), min(h[1] for h in hits),
                       max(h[1] for h in hits), len(hits)) if hits else None
    print(f'where a stab finds him (he stands hidden at (${gx:02X},${gy:02X}); Link written to a grid of offsets '
          f'(4 px) from his position, A, 14 frames): Link X - his X, Link Y - his Y that hit:')
    for facing, b in box.items():
        print(f'   facing {facing}: ' + (f'X {b[0]:+d}..{b[1]:+d}, Y {b[2]:+d}..{b[3]:+d} ({b[4]} grid points)' if b
                                         else 'none'))
    # 2. other weapons at the standing, hidden Ganon: the beam (full hearts), a bomb, the boomerang, a silver arrow
    res = collections.Counter()
    for weapon, slot in (('magic beam', None), ('bomb', 1), ('boomerang', 0), ('silver arrow', 2)):
        w = load(state(hearts=15 if weapon == 'magic beam' else 14.5), 0)
        to_fight(w)
        put(w, gx - 0x30, gy + 8, 'RIGHT')
        if slot is not None:
            w.game.write(SELECTED_SLOT, slot)
        if weapon == 'bomb':
            put(w, gx - 0x10, gy + 8, 'RIGHT')
        w.press('RIGHT', 1)
        w.press('A' if slot is None else 'B', 1)
        if weapon == 'bomb':
            put(w, 0x30, 0xBD)
        hit = False
        for _ in range(50 if weapon != 'bomb' else 60):
            w.game.write(LINK_INVINC, 0x10)
            w.press('NONE', 1)
            hit = hit or bool(w.r(OBJ_TIMER + G)) or w.r(OBJ_HP + G) != 0xF0 or bool(w.r(DYING))
        res[f'{weapon}: {"HIT" if hit else "nothing"}'] += 1
    print(f'   other weapons at the hidden Ganon while he stands: {dict(res)}')
    # 3. the four hits, the 64 shown frames, the jump: from the RAM-eyed kills
    hps, shown_for, jumps, gaps, hurt_shown = (collections.Counter(), collections.Counter(), collections.Counter(),
                                               [], collections.Counter())
    for seed in range(args.seeds):
        w = load(state(), seed)
        wait = to_fight(w)
        bot = GanonBot(w, 'ram')
        last, seq, t_hit, t_last_hit = 'hidden', [0xF0], None, None
        while bot.frames < 6000 and bot.alive() and not w.r(DYING) and phase(w) != 'brown':
            ph = phase(w)
            if ph == 'shown' and last == 'hidden':
                seq.append(w.r(OBJ_HP + G))
                if t_last_hit is not None:
                    gaps.append(bot.frames - t_last_hit)
                t_hit = t_last_hit = bot.frames
                fc_hit = w.r(FRAME_COUNTER)
            if ph == 'hidden' and last == 'shown':
                shown_for[bot.frames - t_hit] += 1
                jumps[(f'FrameCounter at the hit {"odd" if fc_hit & 1 else "even"}',
                       f'(${w.r(OBJ_X + G) & 0xF0:02X},${(w.r(OBJ_Y + G) + 2) & 0xF0:02X})')] += 1
            last = ph
            if ph == 'shown':
                bot.move_shown()
            else:
                bot.fight_frame()
        hps[tuple(f'${h:02X}' for h in seq) + (phase(w),)] += 1
    print(f'the sword phase by the RAM-eyed rule ({args.seeds} seeds): his HP after each hit, then {dict(hps)}; '
          f'frames shown after a hit {dict(shown_for)}; the jump {dict(sorted(jumps.items()))}; frames between '
          f'hits median {int(statistics.median(gaps))} min {min(gaps)}')
    # 4. a stab at him while he is shown
    w = load(state(hearts=14.5), 0)
    to_fight(w)
    w.game.write(OBJ_TIMER + G, 0x30)                            # WRITTEN: shown
    put(w, gx - 8, gy + 8, 'RIGHT')
    w.press('A', 1)
    for _ in range(13):
        w.game.write(LINK_INVINC, 0x10)
        w.press('NONE', 1)
    print(f'   a stab from the hit spot while he is shown (timer WRITTEN $30): HP ${w.r(OBJ_HP + G):02X}')


def to_brown(w, seed_limit=8000):
    """Plays the RAM-eyed rule to the brown phase; returns the bot (phase brown) or None."""
    to_fight(w)
    bot = GanonBot(w, 'ram')
    while bot.frames < seed_limit and bot.alive() and phase(w) != 'brown':
        if phase(w) == 'shown':
            bot.move_shown()
        else:
            bot.fight_frame()
    return bot if phase(w) == 'brown' and bot.alive() else None


def brown(args):
    lasts, hp0, back, balls, flick, sword_hp = (collections.Counter(), collections.Counter(), collections.Counter(),
                                                collections.Counter(), collections.Counter(), collections.Counter())
    for seed in range(args.seeds):
        w = load(state(), seed)
        bot = to_brown(w)
        if bot is None:
            lasts['no brown phase'] += 1
            continue
        hp0[f'${w.r(OBJ_HP + G):02X}'] += 1
        spot = (w.r(OBJ_X + G), w.r(OBJ_Y + G))
        # a stab at him at once (the sword is not tested), then far away
        put(w, spot[0] - 8, spot[1] + 8, 'RIGHT')
        w.press('A', 1)
        for _ in range(13):
            w.game.write(LINK_INVINC, 0x10)
            w.press('NONE', 1)
        sword_hp[f'state ${w.r(OBJ_STATE + G) & 0xF0:02X}+, HP ${w.r(OBJ_HP + G):02X}, dying {w.r(DYING)}'] += 1
        put(w, 0x20 if spot[0] > 0x78 else 0xD0, 0x65 if spot[1] > 0x8D else 0xB5)
        n, moved, first_flicker, born = 14, False, None, 0
        while phase(w) == 'brown' and n < 700:
            hold(w)
            w.game.write(LINK_INVINC, 2)
            w.press('NONE', 1)
            n += 1
            moved = moved or (w.r(OBJ_X + G), w.r(OBJ_Y + G)) != spot and phase(w) == 'brown'
            if first_flicker is None and phase(w) == 'brown' and not drawn(w):
                first_flicker = n
            born += sum(1 for s in range(2, 0x0C) if w.r(OBJ_TYPE + s) == FIREBALL and
                        (w.r(OBJ_STATE + s) & 0xF0) != 0x10)
        lasts[n] += 1
        flick[first_flicker] += 1
        balls[f'moved {moved}'] += 1
        back[f'blue at (${w.r(OBJ_X + G):02X},${w.r(OBJ_Y + G):02X}) HP ${w.r(OBJ_HP + G):02X}'] += 1
    print(f'the brown phase left alone ({args.seeds} seeds): HP when it starts {dict(hp0)}; a stab at once: '
          f'{dict(sword_hp)}; frames brown {dict(sorted(lasts.items()))}; first frame not drawn {dict(flick)}; '
          f'{dict(balls)}; then {dict(back)}')
    # the arrow: silver along his row, wooden, no rupee; and from which Link Y (shooting sideways) / X (up, down)
    for label, extra in (('silver arrow, 10 rupees', {}), ('wooden arrow', {'silver': False}),
                         ('silver arrow, 0 rupees', {'rupees': 0})):
        res, cost = collections.Counter(), collections.Counter()
        for seed in range(args.seeds):
            w = load(state(**extra), seed)
            bot = to_brown(w)
            if bot is None:
                res['no brown phase'] += 1
                continue
            gx, gy = w.r(OBJ_X + G), w.r(OBJ_Y + G)
            r0 = w.r(RUPEES)
            x = gx - 0x40 if gx >= 0x70 else gx + 0x50
            put(w, x, gy + 8, 'RIGHT' if x < gx else 'LEFT')
            w.game.write(SELECTED_SLOT, 2)
            w.press('RIGHT' if x < gx else 'LEFT', 1)
            w.press('B', 1)
            flew, t = False, None
            for f in range(60):
                flew = flew or bool(w.r(ARROW_STATE))
                if w.r(DYING) and t is None:
                    t = f
                w.game.write(LINK_INVINC, 0x10)
                w.press('NONE', 1)
            res[f'an arrow flew: {flew}; ' + (f'his death began {t} frames after B' if t is not None else
                                              f'no death, phase {phase(w)}')] += 1
            cost[r0 - w.r(RUPEES)] += 1
        print(f'   {label}, along his row from 64-80 px ({args.seeds} seeds): {dict(res)}; rupees spent {dict(cost)}')
    w = load(state(), 0)
    bot = to_brown(w)
    gx, gy = w.r(OBJ_X + G), w.r(OBJ_Y + G)
    snap = w.game.snapshot()
    rows = {}
    for facing in FACE:
        ok = []
        for off in range(-40, 41, 1):
            w.game.restore(snap)
            if facing in ('RIGHT', 'LEFT'):
                x, y = (gx - 0x30, gy + off) if facing == 'RIGHT' else (gx + 0x40, gy + off)
            else:
                x, y = (gx + off, gy + 0x40) if facing == 'UP' else (gx + off, gy - 0x30)
            if not (0x18 <= x <= 0xD8 and 0x50 <= y <= 0xC8):
                continue
            put(w, x, y, facing)
            w.game.write(SELECTED_SLOT, 2)
            w.game.write(RUPEES, 10)
            w.press('B', 1)
            for _ in range(40):
                w.game.write(LINK_INVINC, 0x10)
                w.press('NONE', 1)
            if w.r(DYING):
                ok.append(off)
        rows[facing] = (min(ok), max(ok), len(ok)) if ok else None
    print(f'   which silver arrows kill (seed 0, he stands brown at (${gx:02X},${gy:02X}); Link WRITTEN beside him '
          f'with his facing, B): Link Y - his Y (sideways) or Link X - his X (up, down) from..to (count): {rows}')


def after_death(bot, limit=1500):
    """From the frame his death begins: the ashes, the Triforce of Power, the fanfare, the shutters, N into Zelda's
    room.  Returns a dict of frame counts and facts."""
    w, out, t0 = bot.w, {}, bot.frames
    bot.harmful = set()
    out['burst (dying count $50)'] = bot.hold('NONE', lambda w: w.r(DYING) >= 0x50, 200)
    n = bot.hold('NONE', lambda w: w.r(DYING) >= 0xA0, 200)
    out['item (dying count $A0)'] = bot.frames - t0
    out['item at'] = f'(${w.r(OBJ_X + ITEM_SLOT):02X},${w.r(OBJ_Y + ITEM_SLOT):02X}) id ${w.r(0xAB):02X}'
    out['doors before the item is taken'] = f'${w.r(OPEN_DOORS):02X}'
    ix, iy = w.r(OBJ_X + ITEM_SLOT), w.r(OBJ_Y + ITEM_SLOT)
    node = min(bot.graph, key=lambda n: abs(n[0] - ix) + abs(n[1] - iy))
    t1 = bot.frames
    bot.walk_to(node, 600)
    bot.hold('NONE', lambda w: w.r(OBJ_STATE) == 0x40 or w.r(LIFT_ITEM), 30)
    out['walk to the item'] = bot.frames - t1
    t2 = bot.frames
    bot.hold('NONE', lambda w: w.r(INV_POWER_TRIFORCE) == 1, 400)
    out['fanfare (Link halted)'] = bot.frames - t2
    t3 = bot.frames
    bot.hold('NONE', lambda w: w.r(OPEN_DOORS) & 0x0C == 0x0C, 200)
    out['shutters open after the fanfare'] = bot.frames - t3
    out['doors'] = f'${w.r(OPEN_DOORS):02X}'
    out['hearts before/after the fanfare'] = None
    t4 = bot.frames
    ok = bot.walk_to((0x78, 0x5D), 600)
    bot.hold('UP', lambda w: w.room == 0x32 and w.mode == PLAY, 400)
    out['walk N into 0x32'] = bot.frames - t4
    out['in 0x32'] = w.room == 0x32 and ok
    return out


def zelda_room(bot, way='stab'):
    """In 0x32 from its S door to the first frame of game mode $13.  `way`: 'stab' = one stab up from ($78,$AD) at
    the two fires of the middle path, then up; 'walk' = straight up at X $78."""
    w = bot.w
    t0, h0 = bot.frames, bot.lost()
    fires0 = sorted((w.r(OBJ_X + s), w.r(OBJ_Y + s)) for s in range(1, 0x0C) if w.r(OBJ_TYPE + s) == GUARD_FIRE)
    bot.hold('NONE', lambda w: (w.x, w.y) == (w.x, w.y) and w.r(OBJ_STATE) == 0 and w.y <= 0xBD, 60)
    stabs = 0
    if way == 'stab':
        for _ in range(200):
            if w.y <= 0xAD or w.mode != PLAY:
                break
            bot.press('UP')
        for _ in range(3):
            if not any(w.r(OBJ_TYPE + s) == GUARD_FIRE and w.r(OBJ_META + s) == 0 and w.r(OBJ_Y + s) == 0x9D
                       for s in range(1, 0x0C)):
                break
            bot.press('A')
            stabs += 1
            bot.press('NONE', 16)
    for _ in range(400):
        if w.mode != PLAY:
            break
        if w.x != 0x78 and w.y % 8 == 5:
            bot.press('RIGHT' if w.x < 0x78 else 'LEFT')
        else:
            bot.press('UP')
    return {'mode': w.mode, 'frames': bot.frames - t0, 'lost': round(bot.lost() - h0, 3), 'fires': fires0,
            'stabs': stabs, 'at': (w.x, w.y)}


def end(args):
    for ring in (0, 2):
        for way in ('stab', 'walk'):
            rows, notes, parts, zel = [], collections.Counter(), collections.defaultdict(list), collections.Counter()
            for seed in range(args.seeds):
                w = load(state(ring=ring), seed)
                bot, ok, _ = fight(w, 'ram')
                t_kill, lost_kill = bot.frames, bot.lost()
                if not ok:
                    rows.append((False, bot.lost(), bot.frames, bot.by))
                    continue
                out = after_death(bot)
                for key, value in out.items():
                    if isinstance(value, int) and not isinstance(value, bool):
                        parts[key].append(value)
                    else:
                        notes[(key, value)] += 1
                done = {'mode': None}
                if out['in 0x32']:
                    done = zelda_room(bot, way)
                    zel[f"fires at {[f'${x:02X},${y:02X}' for x, y in done['fires']]}, {done['stabs']} stab(s), hearts "
                        f"lost in the room {done['lost']}, mode ${done['mode']:02X} with Link at "
                        f"(${done['at'][0]:02X},${done['at'][1]:02X})"] += 1
                    parts['0x32 door to mode $13'].append(done['frames'])
                parts['his death to mode $13'].append(bot.frames - t_kill)
                rows.append((done['mode'] == 0x13, bot.lost(), bot.frames, bot.by))
            print(f'from his door to game mode $13, {"red ring" if ring else "no ring"}, 0x32 by "{way}" '
                  f'({args.seeds} seeds): {summary(rows)}')
            print('   frames: ' + '; '.join(f'{k} {min(v)}-{max(v)}' for k, v in parts.items()))
            print('   ' + '; '.join(f'{k[0]}: {k[1]} ({v})' for k, v in notes.items() if k[1] is not None))
            print(f'   Zelda\'s room: {dict(zel)}')
    # the fires: what kills them, what a touch costs
    res = collections.Counter()
    for weapon, slot in (('sword from below', None), ('silver arrow from below', 2), ('boomerang', 0)):
        w = load(d9_state(0x32, 'S', hearts=14.5, silver=True), 0)
        w.press('NONE', 40)
        fires = [s for s in range(1, 0x0C) if w.r(OBJ_TYPE + s) == GUARD_FIRE]
        put(w, 0x70, 0xAD if slot is None else 0xBD, 'UP')
        if slot is not None:
            w.game.write(SELECTED_SLOT, slot)
        w.press('UP', 1)
        w.press('A' if slot is None else 'B', 1)
        w.press('NONE', 40)
        res[f'{weapon}: {len(fires)} fires -> '
            f'{sum(1 for s in fires if w.r(OBJ_TYPE + s) == GUARD_FIRE and w.r(OBJ_META + s) == 0)}'] += 1
    for ring in (0, 2):
        w = load(d9_state(0x32, 'S', hearts=14.5, ring=ring), 0)
        w.press('NONE', 40)
        h = hp(w)
        w.game.write(LINK_INVINC, 0)
        put(w, 0x70, 0x9D)
        w.press('NONE', 2)
        res[f'touch, ring {ring}: {round(h - hp(w), 3)} hearts'] += 1
    print(f'the guard fires: {dict(res)}')


def death(args):
    """Link dies in Ganon's room: the modes that follow, and where CONTINUE puts him."""
    w = load(state(hearts=1), 0)
    to_fight(w)
    put(w, w.r(OBJ_X + G) + 8, w.r(OBJ_Y + G) + 8)
    w.game.write(LINK_INVINC, 0)
    modes, last = [], None
    for f in range(1500):
        m = (w.mode, w.r(0x13))
        if m != last:
            modes.append((f, f'mode ${m[0]:02X} sub {m[1]}'))
            last = m
        if w.mode == 8 and f % 30 == 0:
            w.press('START', 1)                                  # the first choice of the menu: CONTINUE
        else:
            w.press('NONE', 1)
        if w.mode == PLAY and f > 100 and w.r(HEARTS) & 0x0F:
            break
    print(f'death in 0x42 (Link written onto Ganon at 1 heart): {modes}')
    print(f'   then: level {w.level} room 0x{w.room:02x} at (${w.x:02X},${w.y:02X}), hearts {hp(w):.2f} of '
          f'{(w.r(HEARTS) >> 4) + 1}, InvArrow {w.r(0x659)}, rupees {w.r(RUPEES)}, RoomHistory '
          f'{[hex(w.r(0x621 + i)) for i in range(6)]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('what', choices=['scene', 'moves', 'sword', 'brown', 'kill', 'end', 'death'])
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--eyes', default='ram,screen,tells,sweep')
    parser.add_argument('--fine', action='store_true', help='sword: every pixel of X instead of every fourth')
    args = parser.parse_args()
    globals()[args.what](args)


if __name__ == '__main__':
    main()
