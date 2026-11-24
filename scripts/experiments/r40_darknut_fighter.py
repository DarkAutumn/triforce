"""Round 40 (docs/experiments/round40.md, "The owner's Darknut fighter"): the owner's rules for fighting Darknuts
(docs/owner/darknuts.md, his takes in replays/owner/darknut/) as a scripted rule over the environment's own actions,
its measurement, and its demonstrations for the teachers.

    python scripts/experiments/r40_darknut_fighter.py measure --room garden|59 --bombs B [--per-state N] [--seed0 S]
        [--look D] [--json OUT] [--workers W]
    python scripts/experiments/r40_darknut_fighter.py demos OUT.npz --room garden|59 --bombs B [--per-state N]
        [--seed0 S] [--json OUT] [--workers W]
    python scripts/experiments/r40_darknut_fighter.py trace --room garden|59 --bombs B SEED
    python scripts/experiments/r40_darknut_fighter.py summarize RECORDS.jsonl...
    python scripts/experiments/r40_darknut_fighter.py merge OUT.npz PIECE.npz...

The rules (the owner, Nov 24, quoted whole in docs/owner/darknuts.md) and what the game's code says about each:

- **Never on a Darknut's line of travel behind it** ("You never try to get to a darknut moving away from you on your
  row by following them directly").  At each 16-pixel grid point a Darknut turns toward its chase target with a chance
  of 129 in 256 when the target is within 8 pixels of it on one axis (asm:Wanderer_MoveAndTurn): behind it on its row,
  that is every grid point.  A position within 8 pixels of the line of a Darknut walking away, up to `CHASE` pixels
  behind it, is refused while another is safe.
- **One Darknut, shadowed at 90 degrees** ("you always work one tile from them shadowing them in parallel [...]
  closer to a 1.25 tiles N or S"; the take `play_with_anticipation`: the landed swings 15-24 pixels off its line and
  5-13 ahead of it): the fighter picks one Darknut (`Fighter.target`) and walks to the spot `OFF` pixels off its line
  on the side Link is on and `AHEAD` pixels ahead of it along its travel (`shadow`); 20-24 pixels off, Link is
  outside the 8 pixels of the turn rule, so the only turn at him is the 90-degree one when it comes level, away from
  which he is already walking.
- **Sword its side, then move at once the way it was going** ("When I hit I IMMEDIATELY start moving again in the same
  direction he is").  The shield refuses a hit only when the blade points straight at its front
  (asm:Damage_HandleWeaponHit@CheckDarknut); a landed hit makes it harmless about 32 frames (timer `$10`, counted
  every second frame, asm:Shove_Begin, asm:Collision_CheckMonsterVsWeaponsAndLink) and knocks it `$40` pixels the
  blade's way along its own axis (across it, between grid points, the knockback is dropped: asm:Shove_Update).  A
  strike is taken when the blade's box (the game's: the sword's point against the monster's middle,
  asm:Collision_CheckStabVsMonster, `BLADE`) holds the Darknut on a frame of the swing's eight hitting frames, the
  Darknut does not face the blade, and it reaches no grid point before then (where it could turn its shield round);
  then the goal is the way it had been walking (`after`).
- **When it turns toward Link, keep walking the way it had been going** ("If he turns towards me I can either keep
  moving the direction he was going"): the goal while the target faces Link and is not flashing.
- **Near a wall, expect the turn and be where its side will be** ("He gets close to the east wall, I know he will
  stop and turn [...] I anticipate he will and sword towards the wall south of where I think he'll be"): when its
  next grid point is closed ahead, the shadow spot is taken on the line it will walk after turning toward Link's
  side (a blocked walker turns perpendicular, asm:Walker_CheckTileCollision).
- **Bombs only for two or more** ("you always want to hit 2+ darknuts with one bomb.  Just hitting a single darknut
  with a bomb is mostly a waste"), only above the route's reserve: a BOMB is laid where the blast (24 pixels of the
  bomb's middle on both axes, 49-72 frames after it is laid, asm:Bomb_CheckBombOrFireVsMonster, asm:Bomb_StateTimes)
  is predicted to catch two or more Darknuts that do not face its direction (a Darknut facing against the direction
  Link faced when he laid it takes no damage, asm:Damage_HandleWeaponHit@CheckDarknut).  With bombs to spend and two
  or more Darknuts alive Link's place is the hotspot (128, 128): the monsters' chase target flips between Link and
  his mirror (255 - X, 255 - Y) (asm:Mode5Play_UpdateWorld@CheckChaseTarget), the same point there, so every Darknut
  is sent at him and they clump.  After laying one behind Darknuts walking away he does not walk on ("a bomb behind a
  darknut hitting their back will shoot them forward", asm:Shove_Begin).

Every decision reads only what a player sees (the RAM the env exposes: positions, facings, flash timers,
knockback, spawn clouds, the bag); the prediction walks each Darknut along its facing at its speed (red 0.5, blue
0.625 pixel a frame, asm:Darknut_Init) and turns it on grid points as above (its own random turn timer is not
visible and is left out).  Safety is `r40_d3_69_line`'s: a candidate is safe when no way on from it is touched within
`HORIZON` frames.  `--look D` (training and demonstrations only: the owner, Nov 23, moving the emulator ahead and back
is allowed when building demonstrations, never at play) times the fighter with a second env: each decision takes the
first of the fighter's ranked actions that, played and followed by the fighter's own choices for D decisions, is not
hit.

`measure` plays the fighter from the route step's entry states (`d3first_5_05e_*`, the Olive Garden from E;
`d3first_3_59n_*`, 3:0x59 from N) through the darknut task's environment (`specialists/d3first/darknut.yaml`) with
the bag fixed (`--bombs`) and the reserve the owner set (the Garden's step 0; 3:0x59 5, decisions 2026-11-24), seeds
`--seed0`.. as `triforce evaluate` draws them (episode `i` seeded `seed0 + i` on the entry states in turn, the
start's random numbers, spawn and hp draw), and writes a record a visit: the ending, hits taken, hearts lost, frames,
swings, bombs used and the Darknuts each bomb hurt, and the situation of every hit.  `demos` builds demonstrations: the
fighter timed by its look-ahead, then a backtracking search seeded by it where it is still hit (`teach`: back to a
decision before the hit and the next of the fighter's own choices there); the visits that leave by the exit unhit are
written as demonstration rows (`triforce.demos`).

It plays many runs: a job of the machine's queue (``jobs.require_job``: ``triforce job run ... --express --timeout
10m -- .venv/bin/python scripts/experiments/r40_darknut_fighter.py ...``).
"""

import argparse
import collections
import heapq
import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r40_d3_69_line import (CLOUD, FLASH, GAME_DIR, GROW, LINK_SPEED, MARGIN, OPPOSITE, REPEAT, SHOVE, SIDEWAYS,  # noqa: E402
                            SWING_FRAMES, TOUCH, VEC, mask_ok, on_grid)

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / 'specialists' / 'd3first' / 'darknut.yaml'
OBJ_X, OBJ_Y, OBJ_DIR, OBJ_TYPE, OBJ_META, OBJ_INVINC, OBJ_HP = 0x70, 0x84, 0x98, 0x34F, 0x405, 0x4F0, 0x485
OBJ_SHOVE_DIR, OBJ_SHOVE_LEFT = 0xC0, 0xD3          # asm:ObjShoveDir, asm:ObjShoveDistance
BOMB_STATE = (0xBC, 0xBD)                           # slots $10/$11 ObjState (asm:Bomb_UpdateBombOrFire)
INV_BOMBS = 0x658                                   # asm:InvBombs
INV_CLOCK = 0x66C                                   # asm:InvClock: monsters frozen, Link invincible
OBJ_GRID_OFFSET = 0x394                             # asm:ObjGridOffset
RED, BLUE = 0x0B, 0x0C
SPEED = {RED: 0.5, BLUE: 0.625}                     # asm:Darknut_Init: q-speed $20 / $28
HORIZON = 48                                        # frames looked ahead for safety (as r40_d3_69_line)
SWING_HIT = (4, 11)                                 # the swing's hitting frames after the A press (zelda-asm items.md)
# The sword's box, the game's (asm:Collision_CheckStabVsMonster: the sword's point, placed by
# asm:Sword_LinkToWeaponOffsetsX/Y, against the monster's middle: under $10 along Link's facing, under $0C across):
# the Darknut's top-left minus Link's, (dx low, dx high, dy low, dy high), each bound exclusive.
BLADE = {'N': (-15, 9, -26, 6), 'S': (-13, 11, -3, 29), 'W': (-27, 5, -11, 13), 'E': (-5, 27, -11, 13)}
BEAM_BOX = 12                                       # a sword shot's box, under $0C (asm:SwordShot_CheckMonsterCollision)
BEAM_SPEED = 3                                      # pixels a frame, from the swing's frame 12 (zelda-asm items.md)
BLAST = 24                                          # under $18 pixels, both axes (asm:Bomb_CheckBombOrFireVsMonster)
FUSE, BLAST_FRAMES = 49, 24                         # the blast starts 49 frames after the bomb is laid, lasts 24
BOMB_AHEAD = 16                                     # a bomb is laid $10 pixels ahead (asm:Link_PlaceWeaponAhead)
OFF, AHEAD = 20, 8                                  # the shadow spot: off the line, ahead along the travel (the take)
CHASE = 64                                          # pixels behind a Darknut within which its line is refused
HOTSPOT = (128, 125)                                # (128, 128) on Link's 8-pixel lattice (Y = $3D + 8k)
STALL = 40                                          # decisions in one place before standing is refused (env: 50)
MIN_CAUGHT = 0.6                                    # a Darknut counts toward a bomb's catch at this chance or more
TIMER_TURN = 0.25                                   # the guessed chance a Darknut's hidden turn timer has run out
BOMB_EXPECT = 2.0                                   # the Darknuts a bomb must be expected to hurt, chances summed
BOMB_FRAMES = 17                                    # frames a BOMB takes through the env, Link held (measured: 15-19)
MARGIN_FLASH = 2                                    # frames after a flash ends before a strike counts on landing
CROWD = 32                                          # pixels (both axes) within which another Darknut crowds a spot
TURNED = 6                                          # decisions after a turn at Link that Link keeps the old way
MARGIN_BOX = 1                                      # pixels inside the blade's box a predicted Darknut must be

# The rooms: the route step's entry states, the bomb reserve the owner set, and what clears the room's exit.
ROOMS = {
    'garden': {'states': 'd3first_5_05e', 'room': (5, 0x05), 'reserve': 0, 'exit': 'stairs',
               'block': (112, 141), 'stairs': (208, 93)},
    '59': {'states': 'd3first_3_59n', 'room': (3, 0x59), 'reserve': 5, 'exit': 'S'},
}


class Floor:
    """Where a body may stand (top-left), from the state's tile grid (`GameState.render_grid`: '#' a wall, 'P' a push
    block).  A Darknut fills 16 x 16 pixels from Y + 3 (as `r40_d3_69_line.Floor`); Link only his foot row, the 8
    pixels from Y + $0B, across his 16-pixel width: his upper body is never tested and overlaps blocks from below
    (zelda-asm mechanics.md "Why Link overlaps blocks", asm:Tile_GetAheadOfObj).  The room bounds hold both (a
    dungeon room: X $20-$D0, Y $5D-$BD); the doorways' corridors are Link's alone."""

    def __init__(self, grid: str):
        rows = grid.splitlines()[1:]
        self.cells = [[c not in '#P' for c in row[4:]] for row in rows]
        self.seen = {}                                        # (X, Y, link) -> the tile test, whole pixels

    def open(self, x: float, y: float, link: bool = True) -> bool:
        if not (0x20 <= x <= 0xD0 and 0x5D <= y <= 0xBD):
            return link and ((x == 0x78 and (0x3D <= y < 0x5D or 0xBD < y <= 0xDD)) or
                             (y == 0x8D and (0x00 <= x < 0x20 or 0xD0 < x <= 0xF0)))
        key = (int(x), int(y), link)
        found = self.seen.get(key)
        if found is None:
            found = self.seen[key] = self.tiles(*key)
        return found

    def __deepcopy__(self, memo):
        return self                                           # the room's floor never changes: a copied fighter shares it

    def tiles(self, x: int, y: int, link: bool) -> bool:
        if link:
            r0, r1 = (y + 0x0B - 0x40) // 8, (y + 0x12 - 0x40) // 8
        else:
            r0, r1 = (y - 0x3D) // 8, (y - 0x3D + 15) // 8
        c0, c1 = x // 8, (x + 15) // 8
        return all(self.cells[r][c] for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                   if 0 <= r < len(self.cells) and 0 <= c < len(self.cells[r]))


class Nut:
    """One Darknut as a player sees it: kind, position, facing, flash timer, knockback, spawn cloud (the RAM); with
    the clock taken (asm:InvClock) it stands where it is and Link is invincible (asm:Wanderer_Update,
    asm:Link_CheckMonsterTouch)."""
    __slots__ = ('slot', 'kind', 'speed', 'x', 'y', 'dir', 'flash', 'cloud', 'shove_dir', 'shove', 'hp', 'frozen')

    def __init__(self, ram: bytes, slot: int):
        self.slot = slot
        self.kind = ram[OBJ_TYPE + slot]
        self.frozen = ram[INV_CLOCK] != 0
        self.speed = 0.0 if self.frozen else SPEED[self.kind]
        self.x, self.y = ram[OBJ_X + slot], ram[OBJ_Y + slot]
        self.dir = GAME_DIR.get(ram[OBJ_DIR + slot] & 0x0F)
        self.flash = 2 * ram[OBJ_INVINC + slot]
        self.cloud = ram[OBJ_META + slot] != 0
        self.shove_dir = GAME_DIR.get(ram[OBJ_SHOVE_DIR + slot] & 0x0F)
        self.shove = ram[OBJ_SHOVE_LEFT + slot] if self.shove_dir else 0
        self.hp = ram[OBJ_HP + slot] >> 4

    def to_grid(self) -> float:
        """Frames to its next grid point along its facing (0: on one; never while frozen)."""
        if self.dir is None:
            return 0
        if self.frozen:
            return float('inf')
        along = self.x if self.dir in 'EW' else self.y - 0x0D
        ahead = (16 - along % 16) % 16 if self.dir in 'ES' else along % 16
        return ahead / self.speed

    def ways_at(self, floor: Floor, x: float, y: float, heading: str, link, timer: float = 0.0
                ) -> list[tuple[str, float]]:
        """The ways it may take at the grid point (x, y), with their chances: toward Link when he is within 8
        pixels of its column or row (129 in 256, asm:Wanderer_MoveAndTurn); else, with chance `timer` (its turn
        timer run out: not visible, a guess), perpendicular toward him (asm:Wanderer_MoveAndTurn@TurnIfTime); else
        on; a closed way: a perpendicular open one, either (asm:Walker_CheckTileCollision)."""
        def free(d):
            return floor.open(x + VEC[d][0] * 8, y + VEC[d][1] * 8, link=False)

        def onward(d, p, ways):
            if free(d):
                ways[d] += p
                return
            sides = [s for s in SIDEWAYS[d] if free(s)] or [OPPOSITE[d]]
            for s in sides:
                ways[s] += p / len(sides)

        ways = collections.Counter()
        lx, ly = link
        toward = ('S' if ly > y else 'N') if abs(lx - x) < 9 else ('E' if lx > x else 'W') if abs(ly - y) < 9 \
            else None
        rest = 1.0
        if toward and free(toward):
            ways[toward] += 129 / 256
            rest = 127 / 256
        if timer:
            side = ('E' if lx >= x else 'W') if heading in 'NS' else ('N' if ly < y else 'S')
            onward(side, rest * timer, ways)
            rest *= 1 - timer
        onward(heading, rest, ways)
        return list(ways.items())

    def futures(self, floor: Floor, link, horizon: int = HORIZON, struck: str | None = None, hit_at: int = 0,
                depth: int = 1, timer: float = 0.0):
        """[(positions [horizon + 1, 2], facings [horizon + 1], harmless [horizon + 1], chance)]: the ways it may
        go.  A knockback first (4 pixels a frame, asm:Shove_Update, until spent or stopped by a wall); then it walks
        its facing and may turn on a grid point (`ways_at`), branching at its first `depth` grid points.
        `struck` (a strike's way): the blade lands at frame `hit_at`; it flashes `FLASH` frames and is knocked
        `SHOVE` pixels that way, along its own axis or from a grid point."""
        harmless = np.zeros(horizon + 1, bool)
        harmless[:min(self.flash, horizon + 1)] = True
        if self.cloud:
            harmless[:CLOUD] = True
        if struck:
            harmless[hit_at:hit_at + FLASH] = True
        if self.frozen:
            harmless[:] = True
            return [(np.full((horizon + 1, 2), (float(self.x), float(self.y))), [self.dir] * (horizon + 1), harmless,
                     1.0)]
        out = []
        start = (float(self.x), float(self.y), None if self.cloud else self.dir, self.shove_dir, self.shove)
        stack = [([start[:2]], [start[2]], start, 1.0, depth, None)]
        while stack:
            path, faces, (x, y, heading, shove_dir, shove), chance, left, turned_at = stack.pop()
            while len(path) <= horizon:
                if struck and len(path) == hit_at and (heading in (struck, OPPOSITE[struck]) or on_grid(x, y)):
                    shove_dir, shove = struck, SHOVE
                if shove > 0:
                    step = min(4, shove)
                    shove -= step
                    nx, ny = x + VEC[shove_dir][0] * step, y + VEC[shove_dir][1] * step
                    if floor.open(nx, ny, link=False):
                        x, y = nx, ny
                    else:
                        shove = 0
                elif heading is not None:
                    if len(path) > 1 and on_grid(x, y) and left > 0 and (x, y) != turned_at and \
                            not (struck and len(path) < hit_at + FLASH and harmless[len(path)]):
                        for way, p in self.ways_at(floor, x, y, heading, link, timer):
                            stack.append((list(path), list(faces), (x, y, way, None, 0), chance * p, left - 1,
                                          (x, y)))
                        break
                    if on_grid(x, y) and len(path) > 1 and not floor.open(x + VEC[heading][0] * 8,
                                                                         y + VEC[heading][1] * 8, link=False):
                        heading = next((d for d in SIDEWAYS[heading] if floor.open(
                            x + VEC[d][0] * 8, y + VEC[d][1] * 8, link=False)), OPPOSITE[heading])
                    nx, ny = x + VEC[heading][0] * self.speed, y + VEC[heading][1] * self.speed
                    # a step that passes a grid point stops on it (the game moves whole pixels and turns there)
                    if heading in 'EW' and int(x // 16) != int(nx // 16) and x % 16:
                        nx = (x // 16 + (heading == 'E')) * 16
                    elif heading in 'NS' and int((y - 0x0D) // 16) != int((ny - 0x0D) // 16) and (y - 0x0D) % 16:
                        ny = ((y - 0x0D) // 16 + (heading == 'S')) * 16 + 0x0D
                    if floor.open(nx, ny, link=False):
                        x, y = nx, ny
                path.append((x, y))
                faces.append(heading)
            else:
                out.append((np.array(path), faces, harmless, chance))
        return out


def darknuts(ram: bytes) -> list[Nut]:
    """The room's Darknuts, alive or in their spawn cloud (a dying one, metastate $10 and up, is left out)."""
    return [Nut(ram, s) for s in range(1, 12)
            if ram[OBJ_TYPE + s] in SPEED and ram[OBJ_META + s] < 0x10]


def contact(paths: np.ndarray, futures) -> np.ndarray:
    """Per Link path ([P, HORIZON + 1, 2]): the first frame some way of some Darknut touches him (HORIZON + 1: none);
    touch under 9 pixels on both axes (asm:Link_CheckMonsterTouch_NoReset), the box widening with the frames."""
    frames = np.arange(HORIZON + 1)
    box = np.where(frames < 4, TOUCH, TOUCH + MARGIN + GROW * frames)
    first = np.full(len(paths), HORIZON + 1)
    for ways in futures:
        for pos, _, harmless, _ in ways:
            d = np.abs(paths - pos[None, :HORIZON + 1])
            touch = (d[..., 0] < box) & (d[..., 1] < box) & ~harmless[None, :HORIZON + 1]
            touch[:, 0] = False
            first = np.minimum(first, np.where(touch, frames[None], HORIZON + 1).min(axis=1))
    return first


def lattice(x: float, y: float) -> tuple[int, int]:
    """The nearest point of Link's 8-pixel lattice (X = 8k, Y = $3D + 8k)."""
    return int(round(x / 8)) * 8, 0x3D + int(round((y - 0x3D) / 8)) * 8


def grid_offset(ram: bytes) -> int:
    """Link's signed pixels from the grid point he left (asm:ObjGridOffset, slot 0)."""
    value = ram[OBJ_GRID_OFFSET]
    return value - 256 if value >= 128 else value


def aligned(x: float, y: float, facing: str | None, goff: float) -> tuple[float, float]:
    """Where a way across the axis Link faces takes him first when he is off his 8-pixel grid on it
    (asm:Link_ModifyDirOnGridLine): 4 or more pixels from the grid point he left, on the way he faces to the next
    grid point; under 4, back to the one he left."""
    if facing not in VEC:
        return x, y
    upright = facing in 'NS'
    along = (y - 0x3D) if upright else x
    if along % 8 == 0:
        return x, y
    if abs(goff) >= 4:
        target = (along // 8 + 1) * 8 if facing in 'ES' else (along // 8) * 8
    else:
        target = along - goff
    return (x, target + 0x3D) if upright else (target, y)


class Fighter:
    """The rule for one visit: `ranked(ram, ok)` -> the env actions best first; `note` keeps the visit's counts."""

    def __init__(self, room: str, floor: Floor, bombs_used_ok=True):
        self.room, self.floor = ROOMS[room], floor
        self.reserve = self.room['reserve']
        self.target = None                                    # the Darknut's slot being shadowed
        self.facing = {}                                      # slot -> its facing at the last decision
        self.turned = {}                                      # slot -> (decision, the way it walked before) at its turn
        self.struck_at = {}                                   # slot -> (decision, the way it was walking)
        self.decision = 0
        self.still, self.last = 0, None
        self.swings = self.bombs = 0
        self.bomb_held = None                                 # (the way laid, decisions left to stay)
        self.predicted = {}                                   # BOMB way -> the Darknuts it is predicted to hurt
        self.bomb_pred = []                                   # per bomb laid: that prediction
        self.pushed = False
        self.bombs_ok = bombs_used_ok

    # --- geometry --------------------------------------------------------------------------------------------------

    def walk(self, x, y, facing, goff, plan, horizon=HORIZON):
        """Link's positions frame by frame for `plan` [(way, frames)] (a direction or None: standing).  A way across
        the axis he faces, off his 8-pixel grid on it, first takes him to the grid point `aligned` gives (his grid
        offset `goff` is followed along the plan)."""
        path, px, py = [(x, y)], float(x), float(y)
        for way, frames in plan:
            for _ in range(frames):
                if len(path) > horizon:
                    break
                if way:
                    if facing in VEC and (way in 'NS') != (facing in 'NS'):
                        tx, ty = aligned(px, py, facing, goff)
                        if (px, py) != (tx, ty):
                            dx = max(-LINK_SPEED, min(LINK_SPEED, tx - px))
                            dy = max(-LINK_SPEED, min(LINK_SPEED, ty - py))
                            goff += dx + dy
                            px, py = px + dx, py + dy
                            path.append((px, py))
                            continue
                        goff = 0
                    facing = way
                    nx, ny = px + VEC[way][0] * LINK_SPEED, py + VEC[way][1] * LINK_SPEED
                    if self.floor.open(*lattice(nx, ny)) or self.floor.open(nx, ny):
                        along = (py - 0x3D) if way in 'NS' else px
                        goff = (0 if along % 8 == 0 else goff) + LINK_SPEED * (VEC[way][0] + VEC[way][1])
                        px, py = nx, ny
                        if ((py - 0x3D) if way in 'NS' else px) % 8 == 0:
                            goff = 0
                path.append((px, py))
        while len(path) <= horizon:
            path.append(path[-1])
        return np.array(path)

    @staticmethod
    def lead(x, y, facing, goff, way) -> tuple[float, float, list]:
        """A strike across the way Link faces, off his 8-pixel grid, first walks him onto a grid point (`aligned`;
        docs/architecture.md "Attacks by buttons"): (where he strikes from, the frames walked there)."""
        if facing in VEC and (way in 'NS') != (facing in 'NS'):
            sx, sy = aligned(x, y, facing, goff)
            if (sx, sy) != (x, y):
                step = ('E' if sx > x else 'W') if sx != x else ('S' if sy > y else 'N')
                return sx, sy, [(step, round((abs(sx - x) + abs(sy - y)) / LINK_SPEED))]
        return x, y, []

    def lands(self, d: Nut, sx, sy, way, delay) -> int | None:
        """The frame (from now) the blade swung `way` from (sx, sy) after `delay` frames first touches `d` unshielded,
        or None: in the box (`BLADE`) on two hitting frames running (a pixel's slack in the prediction), not facing
        the blade, its flash over `MARGIN_FLASH` frames before, and no grid point reached before that frame (where
        it could turn its shield round)."""
        if d.cloud or d.dir is None or d.dir == OPPOSITE[way] or d.shove:
            return None
        x0, x1, y0, y1 = BLADE[way]
        to_grid = d.to_grid()
        inside = []
        for f in range(delay + SWING_HIT[0], delay + SWING_HIT[1] + 1):
            if 0 < to_grid < f or (to_grid == 0 and on_grid(d.x, d.y)):
                break
            nx, ny = d.x + VEC[d.dir][0] * d.speed * f, d.y + VEC[d.dir][1] * d.speed * f
            inside.append(f >= d.flash + MARGIN_FLASH and x0 + MARGIN_BOX < nx - sx < x1 - MARGIN_BOX and
                          y0 + MARGIN_BOX < ny - sy < y1 - MARGIN_BOX)
        for k in range(len(inside) - 1):
            if inside[k] and inside[k + 1]:
                return delay + SWING_HIT[0] + k
        return None

    def beam_lands(self, d: Nut, sx, sy, way, delay) -> int | None:
        """A sword shot's flight (from the swing's frame 12, `BEAM_SPEED` a frame) against `d` walking on to its
        next grid point: the frame it first overlaps it unshielded, or None."""
        if d.cloud or d.dir is None or d.dir == OPPOSITE[way] or d.shove:
            return None
        to_grid = d.to_grid()
        bx, by = sx + VEC[way][0] * 8, sy + VEC[way][1] * 8
        for f in range(delay + 12, delay + 12 + 60):
            k = f - delay - 12
            px, py = bx + VEC[way][0] * BEAM_SPEED * k, by + VEC[way][1] * BEAM_SPEED * k
            if not (0x10 <= px <= 0xE0 and 0x4D <= py <= 0xCD):
                return None
            if 0 < to_grid < f:
                return None
            nx, ny = d.x + VEC[d.dir][0] * d.speed * f, d.y + VEC[d.dir][1] * d.speed * f
            if abs(nx - px) < BEAM_BOX and abs(ny - py) < BEAM_BOX and f >= d.flash:
                return f
        return None

    def strikes(self, ram, nuts, ok) -> list[tuple[str, list, Nut, int]]:
        """The strikes that would hurt a Darknut: (action, Link's frames, the Darknut, the frame it lands)."""
        x, y, facing = ram[OBJ_X], ram[OBJ_Y], GAME_DIR.get(ram[OBJ_DIR] & 0x0F)
        out = []
        for way in BLADE:
            text = next((f'{kind} {way}' for kind in ('SWORD', 'BEAMS') if ok(f'{kind} {way}')), None)
            if text is None:
                continue
            sx, sy, lead = self.lead(x, y, facing, grid_offset(ram), way)
            delay = sum(n for _, n in lead)
            best = None
            for d in nuts:
                f = self.lands(d, sx, sy, way, delay)
                if f is None and text.startswith('BEAMS'):
                    f = self.beam_lands(d, sx, sy, way, delay)
                if f is not None and (best is None or (d.slot == self.target) > (best[0].slot == self.target)):
                    best = (d, f)
            if best:
                out.append((text, lead + [(None, SWING_FRAMES - 4)], best[0], best[1]))
        return out

    def catch(self, ram, nuts, way, bomb_xy) -> list[tuple[Nut, float]]:
        """The Darknuts a bomb laid `way` at `bomb_xy` now would hurt, with their chances: in the blast's box on
        a frame of the blast, not facing against `way` then (the shield), not flashing."""
        bx, by = bomb_xy
        link = (ram[OBJ_X], ram[OBJ_Y])
        horizon = FUSE + BLAST_FRAMES
        out = []
        for d in nuts:
            chance = 0.0
            for path, faces, harmless, p in d.futures(self.floor, link, horizon=horizon, depth=3, timer=TIMER_TURN):
                for f in range(FUSE, horizon + 1):
                    if harmless[f] or faces[f] == OPPOSITE[way]:
                        continue
                    if abs(path[f][0] - bx) < BLAST and abs(path[f][1] - by) < BLAST:
                        chance += p
                        break
            if chance > 0:
                out.append((d, chance))
        return out

    def bomb_options(self, ram, nuts, ok) -> list[tuple[str, list, list]]:
        """BOMB ways that catch two or more (`catch`, each at `MIN_CAUGHT` or more), above the reserve only."""
        if not self.bombs_ok or ram[INV_BOMBS] <= self.reserve or any(ram[s] in range(0x11, 0x13) for s in BOMB_STATE):
            return []
        x, y = ram[OBJ_X], ram[OBJ_Y]
        out = []
        for way in VEC:
            text = f'BOMB {way}'
            if not ok(text):
                continue
            bomb = (x + VEC[way][0] * BOMB_AHEAD, y + VEC[way][1] * BOMB_AHEAD)
            every = self.catch(ram, nuts, way, bomb)
            caught = [(d, c) for d, c in every if c >= MIN_CAUGHT]
            if len(caught) >= 2 and sum(c for _, c in every) >= BOMB_EXPECT:
                out.append((text, [(None, BOMB_FRAMES)], caught))
                self.predicted[text] = round(sum(c for _, c in caught), 2)
        return out

    # --- the goal ----------------------------------------------------------------------------------------------------

    def choose_target(self, x, y, nuts) -> Nut | None:
        """Keeps the Darknut being shadowed while it lives; else the nearest one out of its cloud, one walking toward
        Link along his line last."""
        live = [d for d in nuts if not d.cloud]
        if not live:
            return None
        keep = next((d for d in live if d.slot == self.target), None)
        if keep:
            return keep

        def cost(d):
            facing_link = d.dir and (VEC[d.dir][0] * (x - d.x) + VEC[d.dir][1] * (y - d.y)) > 0 and \
                (abs(x - d.x) < 9 or abs(y - d.y) < 9)
            return abs(d.x - x) + abs(d.y - y) + (48 if facing_link else 0)
        best = min(live, key=cost)
        self.target = best.slot
        return best

    def shadow(self, x, y, d: Nut) -> tuple[int, int]:
        """The spot `OFF` pixels off `d`'s line on Link's side and `AHEAD` pixels ahead of it along its travel; when
        its next grid point is closed ahead (a wall), on the line it will walk after turning toward Link's side."""
        heading = d.dir
        px, py = float(d.x), float(d.y)
        if heading:
            t = d.to_grid()
            gx, gy = px + VEC[heading][0] * d.speed * t, py + VEC[heading][1] * d.speed * t
            if not self.floor.open(gx + VEC[heading][0] * 8, gy + VEC[heading][1] * 8, link=False):
                side = [w for w in SIDEWAYS[heading]
                        if self.floor.open(gx + VEC[w][0] * 8, gy + VEC[w][1] * 8, link=False)]
                if side:
                    toward = min(side, key=lambda w: (VEC[w][0] * (gx - x) + VEC[w][1] * (gy - y)))
                    heading, px, py = toward, gx, gy
        else:
            heading = 'E'
        vx, vy = VEC[heading]
        spots = []
        for sign in (1, -1):
            ox, oy = (vy * sign, vx * sign) if True else (0, 0)
            sx, sy = px + ox * OFF + vx * AHEAD, py + oy * OFF + vy * AHEAD
            spots.append(((sx - x) * ox + (sy - y) * oy < 0, sx, sy))   # Link's own side first
        for _, sx, sy in sorted(spots, key=lambda s: not s[0]):
            spot = lattice(sx, sy)
            if self.floor.open(*spot):
                return spot
        return lattice(px, py)

    def strike_spot(self, x, y, d: Nut) -> tuple[int, int]:
        """For a Darknut that stands still (the clock): the nearest lattice point from which a blade not facing its
        shield holds it in its box."""
        best = None
        for px in range(int(d.x) - 32, int(d.x) + 33, 8):
            for py in range(0x3D + (int(d.y) - 32 - 0x3D) // 8 * 8, int(d.y) + 33, 8):
                if not self.floor.open(px, py):
                    continue
                for way, (x0, x1, y0, y1) in BLADE.items():
                    if way != OPPOSITE.get(d.dir) and x0 + 2 < d.x - px < x1 - 2 and y0 + 2 < d.y - py < y1 - 2:
                        cost = abs(px - x) + abs(py - y)
                        if best is None or cost < best[0]:
                            best = (cost, (px, py))
        return best[1] if best else self.shadow(x, y, d)

    def goal(self, ram, nuts) -> tuple[int, int]:
        """Where Link wants to be: after a strike, on the way the struck one had been walking; when the target has
        just turned at him, on the way it had been walking; with bombs to spend and two or more Darknuts, the
        hotspot; else (and always while the clock holds them) its shadow spot."""
        x, y = ram[OBJ_X], ram[OBJ_Y]
        d = self.choose_target(x, y, nuts)
        if d is None:
            return HOTSPOT
        if d.frozen:
            return self.strike_spot(x, y, d)
        struck = self.struck_at.get(d.slot)
        if struck and d.flash and abs(d.x - x) < 32 and abs(d.y - y) < 32:
            way = struck[1]
            return lattice(x + VEC[way][0] * 24, y + VEC[way][1] * 24)
        turned = self.turned.get(d.slot)
        if d.dir and not d.flash and turned and self.decision - turned[0] <= TURNED and \
                VEC[d.dir][0] * (x - d.x) + VEC[d.dir][1] * (y - d.y) > 0 and \
                (abs(x - d.x) < 24 if d.dir in 'NS' else abs(y - d.y) < 24):
            way = turned[1]                                   # it turned at Link: on the way it had been going
            return lattice(x + VEC[way][0] * 24, y + VEC[way][1] * 24)
        if self.bombs_ok and ram[INV_BOMBS] > self.reserve and sum(not n.cloud for n in nuts) >= 2:
            return HOTSPOT
        return self.shadow(x, y, d)

    def field(self, goal, nuts) -> dict:
        """Dijkstra's distance to `goal` over Link's 8-pixel lattice (the floor and the doorways), a step 8 pixels;
        cells near a Darknut, and on a Darknut's line behind it, cost more."""
        def penalty(cx, cy):
            p = 2 * sum(not self.floor.open(cx + vx * 16, cy + vy * 16, link=False) for vx, vy in VEC.values())
            for d in nuts:
                if d.cloud or d.flash > 8:
                    continue
                if abs(cx - d.x) < 20 and abs(cy - d.y) < 20:
                    p += 6
                if self.chasing(cx, cy, d):
                    p += 4
            return p
        start = lattice(*goal)
        dist = {start: 0}
        heap = [(0, start)]
        while heap:
            c, (cx, cy) = heapq.heappop(heap)
            if c > dist.get((cx, cy), 1e9):
                continue
            for vx, vy in VEC.values():
                nx, ny = cx + vx * 8, cy + vy * 8
                if not self.floor.open(nx, ny):
                    continue
                nc = c + 1 + penalty(nx, ny)
                if nc < dist.get((nx, ny), 1e9):
                    dist[(nx, ny)] = nc
                    heapq.heappush(heap, (nc, (nx, ny)))
        return dist

    @staticmethod
    def chasing(x, y, d: Nut) -> bool:
        """Link at (x, y) is on `d`'s line of travel behind it: within 8 pixels of its line (the turn rule's), up to
        `CHASE` pixels behind it."""
        if d.dir is None:
            return False
        vx, vy = VEC[d.dir]
        along = (x - d.x) * vx + (y - d.y) * vy
        across = abs((x - d.x) * vy) + abs((y - d.y) * vx)
        return across < 9 and -CHASE < along < 0

    # --- the decision --------------------------------------------------------------------------------------------

    def ranked(self, ram: bytes, ok, grid: str | None = None) -> list[str]:
        """The actions the fighter would take here, best first."""
        x, y = ram[OBJ_X], ram[OBJ_Y]
        facing = GAME_DIR.get(ram[OBJ_DIR] & 0x0F)
        nuts = darknuts(ram)
        if not nuts:
            return self.leave(ram, ok, grid)
        outward = None                                        # in a doorway: never the way out (a wrong exit)
        if not (0x20 <= x <= 0xD0 and 0x5D <= y <= 0xBD):
            outward = 'E' if x > 0xD0 else 'W' if x < 0x20 else 'N' if y < 0x5D else 'S'
        goal = self.goal(ram, nuts)
        dist = self.field(goal, nuts)

        def togo(px, py):
            return dist.get(lattice(px, py), 999) + (abs(px - goal[0]) + abs(py - goal[1])) / 16

        options = []                                          # (text, Link's first frames, kind, extra)
        for step in list(VEC) + [None]:
            text = f'MOVE {step} 4' if step else 'WAIT 4'
            if ok(text) and not (step is None and self.still >= STALL) and step != outward:
                options.append((text, [(step, 4)], 'move', None))
        for text, first, d, f in self.strikes(ram, nuts, ok):
            options.append((text, first, 'strike', (d, f)))
        for text, first, caught in self.bomb_options(ram, nuts, ok):
            options.append((text, first, 'bomb', caught))
        plain = [d.futures(self.floor, (x, y)) for d in nuts]
        conts = [None] + list(VEC)
        scored = []
        for rank, (text, first, kind, extra) in enumerate(options):
            futures = plain
            if kind == 'strike':
                d, f = extra
                futures = [d.futures(self.floor, (x, y), struck=text.split()[1], hit_at=f) if n is d else p
                           for n, p in zip(nuts, plain)]
            plans = [first + [(then, HORIZON)] for then in conts]
            paths = np.stack([self.walk(x, y, facing, grid_offset(ram), plan) for plan in plans])
            if kind == 'move' and text != 'WAIT 4' and np.allclose(paths[0, 4], (x, y)):
                continue                                      # a wall
            touch = contact(paths, futures)
            if kind == 'strike':                              # a miss must leave him time to step away after it
                plain_touch = contact(paths, plain)
                swing_end = sum(n for _, n in first) + 4
                touch = np.where(plain_touch <= swing_end, plain_touch, touch)
            ends = [togo(*paths[i, 4 + sum(n for _, n in first)]) for i in range(len(plans))]
            i = min(range(len(plans)), key=lambda i: (-touch[i], ends[i]))
            after = paths[0, sum(n for _, n in first)]
            chase = any(self.chasing(after[0], after[1], d) for d in nuts if not d.cloud and not d.flash)
            stay = bool(kind == 'move' and self.bomb_held and text == f'MOVE {self.bomb_held[0]} 4')
            value = {'strike': 0, 'bomb': 1, 'move': 2}[kind]
            target_first = 0 if kind != 'strike' or extra[0].slot == self.target else 1
            crowd = sum(abs(after[0] - d.x) < CROWD and abs(after[1] - d.y) < CROWD
                        for d in nuts if d.slot != self.target and not d.cloud)
            robust = -min(int((touch > HORIZON).sum()), 2)    # ways on that stay unhit: more is better, up to 2
            scored.append((int(touch[i]), chase or stay, robust, value, target_first, togo(*after) + crowd, rank,
                           text))
        if not scored:
            return ['WAIT 4']
        safe = [s for s in scored if s[0] > HORIZON]
        unsafe = [s for s in scored if s[0] <= HORIZON]
        safe.sort(key=lambda s: s[1:7])
        unsafe.sort(key=lambda s: (-s[0], s[1], s[5], s[6]))
        return [s[7] for s in safe + unsafe]

    def leave(self, ram, ok, grid) -> list[str]:
        """The room is clear: to its exit (3:0x59's S door; the Garden's block pushed S from above it, then its
        stairs), along Dijkstra's lattice path.  Off the lattice a move across the axis Link walks first takes him to
        the nearest lattice point (the game's own alignment, `walk`), so the lattice point he rounds to is where he
        is."""
        x, y = ram[OBJ_X], ram[OBJ_Y]
        room = self.room
        here = lattice(x, y)
        if room['exit'] == 'S':
            goal = (0x78, 0xBD)
            if here == goal or (x == 0x78 and y > 0xBD):
                return ['MOVE S 4']
        else:
            block = [(c, r) for r, row in enumerate((grid or '').splitlines()[1:]) for c, ch in enumerate(row[4:])
                     if ch == 'P']
            if grid is not None and (not block or min(block) != (room['block'][0] // 8, (room['block'][1] + 3 - 0x40)
                                                                   // 8)):
                self.pushed = True                            # the block slid: the stairs are open
            if not self.pushed:
                bx, by = room['block']
                goal = (bx, by - 16)
                if here == goal:
                    return ['MOVE S 8']
            else:
                goal = room['stairs']
                if here == goal and (x, y) != goal:
                    # the stairs take Link on a frame that starts on their grid point (asm:Warp_CheckUnderLink); a
                    # step across his axis aligns him onto it first (the room's E bound holds him at X $D0)
                    return ['MOVE E 4' if y != goal[1] else 'MOVE N 4']
        dist = self.field(goal, [])
        best = sorted((dist.get((here[0] + VEC[w][0] * 8, here[1] + VEC[w][1] * 8), 999), w) for w in VEC)
        return [f'MOVE {w} 4' for _, w in best]

    def note(self, ram: bytes, text: str, nuts=None):
        """Keeps the visit's counts for the action taken."""
        nuts = darknuts(ram) if nuts is None else nuts
        x, y = ram[OBJ_X], ram[OBJ_Y]
        self.still = self.still + 1 if (x, y) == self.last else 0
        self.last = (x, y)
        for d in nuts:
            prev = self.facing.get(d.slot)
            if d.dir and prev and d.dir != prev:
                self.turned[d.slot] = (self.decision, prev)
            if d.dir:
                self.facing[d.slot] = d.dir
        kind = text.split()[0]
        if kind in ('SWORD', 'BEAMS'):
            self.swings += 1
            way = text.split()[1]
            for d in nuts:
                sx, sy, lead = self.lead(x, y, GAME_DIR.get(ram[OBJ_DIR] & 0x0F), grid_offset(ram), way)
                if d.dir and self.lands(d, sx, sy, way, sum(n for _, n in lead)) is not None:
                    self.struck_at[d.slot] = (self.decision, d.dir)
        if kind == 'BOMB':
            self.bombs += 1
            way = text.split()[1]
            self.bomb_held = (way, 6)
            self.bomb_pred.append(self.predicted.get(text))
        elif self.bomb_held:
            self.bomb_held = (self.bomb_held[0], self.bomb_held[1] - 1) if self.bomb_held[1] > 1 else None
        if self.target is not None and not any(d.slot == self.target for d in nuts):
            self.target = None
        self.decision += 1

    def decide(self, ram: bytes, ok, grid=None) -> str:
        text = self.ranked(ram, ok, grid)[0]
        self.note(ram, text)
        return text


# --- playing visits ---------------------------------------------------------------------------------------------------

def task_file(room: str, bombs: int) -> Path:
    """The case's task (`experiments/r40-darknut/<room>-b<bombs>.yaml`: the darknut task with the room's entry
    states and the bag fixed), the one `triforce evaluate` plays today's network on."""
    return ROOT / 'experiments' / 'r40-darknut' / f'{room}-b{bombs}.yaml'


def make_env(room: str, bombs: int, states_dir=None):
    """The case's env (`task_file`): every reset names its state, holds exactly `bombs` bombs."""
    from triforce._native import Env, SpecialistConfig
    return Env(SpecialistConfig.load(str(task_file(room, bombs))), explicit_starts=True, strict_actions=False, seed=0,
               states_dir=None if states_dir is None else str(states_dir))


def bomb_hurts(before: bytes, after: bytes) -> int:
    """The Darknuts whose HP fell (or died) between two RAMs while one of Link's bombs blasted."""
    hurt = 0
    for s in range(1, 12):
        if before[OBJ_TYPE + s] in SPEED and before[OBJ_META + s] < 0x10:
            died = after[OBJ_TYPE + s] not in SPEED or after[OBJ_META + s] >= 0x10
            if died or after[OBJ_HP + s] < before[OBJ_HP + s]:
                hurt += 1
    return hurt


class Tally:
    """A visit's counts, step by step: hits and their situations, hearts, frames, swings, bombs and what each hurt."""

    def __init__(self, env):
        self.health = env.state.health
        self.start_health = self.health
        self.hits, self.frames, self.where, self.lost = 0, 0, [], 0.0
        self.bomb_hurt = []                                   # per bomb laid: Darknuts it hurt
        self.live_bomb = None                                 # index into bomb_hurt while its blast may land

    def step(self, ram_before: bytes, text: str, env, info, fighter: Fighter | None):
        ram = bytes(env.state.ram)
        self.frames += info.get('frames', 0)
        if text.startswith('BOMB') and ram_before[INV_BOMBS] > ram[INV_BOMBS]:
            self.bomb_hurt.append(0)
            self.live_bomb = (len(self.bomb_hurt) - 1, self.frames)
        if self.live_bomb is not None:
            index, laid = self.live_bomb
            blasting = any(ram[s] == 0x13 for s in BOMB_STATE) or any(ram_before[s] == 0x13 for s in BOMB_STATE)
            if blasting:
                self.bomb_hurt[index] += bomb_hurts(ram_before, ram)
            if self.frames - laid > FUSE + BLAST_FRAMES + 20:
                self.live_bomb = None
        now = env.state.health
        if now < self.health:
            self.hits += 1
            self.lost += self.health - now
            x, y = ram_before[OBJ_X], ram_before[OBJ_Y]
            nuts = darknuts(ram_before)
            near = min(nuts, key=lambda d: abs(d.x - x) + abs(d.y - y), default=None)
            self.where.append({
                'frame': self.frames, 'link': [x, y], 'action': text,
                'darknut': near and [near.x, near.y, near.dir, near.flash, near.slot],
                'target': fighter.target if fighter else None,
                'situation': situation(ram_before, text, near, fighter)})
        self.health = now


def situation(ram, text, d: Nut | None, fighter: Fighter | None) -> str:
    """Where a hit was taken: 'chasing' (Link on its line behind it), 'turned' (it walked level with Link and
    turned at him), 'swing' (Link swinging), 'bomb' (laying a bomb), 'flash-ended' (it had been struck), 'wall'
    (it turned at a wall), 'other'."""
    if d is None:
        return 'other'
    x, y = ram[OBJ_X], ram[OBJ_Y]
    if text.startswith(('SWORD', 'BEAMS')):
        return 'swing'
    if text.startswith('BOMB'):
        return 'bomb'
    if fighter and d.slot in fighter.struck_at and fighter.decision - fighter.struck_at[d.slot][0] < 6:
        return 'flash-ended'
    if Fighter.chasing(x, y, d):
        return 'chasing'
    if d.dir and not fighter.floor.open(d.x + VEC[d.dir][0] * 16, d.y + VEC[d.dir][1] * 16, link=False):
        return 'wall'
    if d.dir and (abs(x - d.x) < 9 or abs(y - d.y) < 9):
        return 'turned'
    return 'other'


def outlook(probe, snapshot, route, fighter_state, first: str, health: float, depth: int, grid) -> int:
    """Training-time look-ahead: `probe` plays `first` from `snapshot`, then the fighter's own choices, up to
    `depth` decisions in all; returns how many it played before Link was hit (`depth` + 1: none, or it left by
    the exit)."""
    import copy
    probe.set_route(*route)
    obs, _ = probe.restore(snapshot, 'look-ahead')
    fighter = copy.deepcopy(fighter_state)
    step = first
    for k in range(depth):
        if k:
            step = fighter.decide(bytes(probe.state.ram), mask_ok(obs), grid)
        else:
            fighter.note(bytes(probe.state.ram), step)
        obs, _, terminated, truncated, info = probe.step(step)
        if probe.state.health < health:
            return k
        if terminated or truncated:
            return depth + 1 if info.get('ending') == 'exit-correct' else k
    return depth + 1


LIMIT = 3000                                                  # decisions a visit at most (the env stops it first)


def play(env, room: str, seed: int, start: str, recorder=None, probe=None, depth: int = 0) -> dict:
    """One visit by the fighter; with `probe` and `depth`, each decision the first of the fighter's ranked actions
    whose `outlook` is clean (none: the one hit latest)."""
    obs, _ = env.reset(seed=seed, options={'start_state': start})
    fighter = Fighter(room, Floor(env.state.render_grid()))
    tally = Tally(env)
    info = {}
    for _ in range(LIMIT):
        ram = bytes(env.state.ram)
        grid = env.state.render_grid() if not darknuts(ram) else None
        ok = mask_ok(obs)
        order = fighter.ranked(ram, ok, grid)
        step = order[0]
        if probe is not None and depth and len(order) > 1 and darknuts(ram):
            snapshot, route = env.snapshot(), env.route
            best = None
            for candidate in order[:6]:
                lasted = outlook(probe, snapshot, route, fighter, candidate, tally.health, depth, grid)
                if best is None or lasted > best[0]:
                    best = (lasted, candidate)
                if lasted > depth:
                    break
            step = best[1]
        fighter.note(ram, step)
        if recorder is not None and ok(step):
            kind = step.split()[0]
            recorder.add(obs, step, repeat=REPEAT.get(kind, 1), any_duration=kind in ('MOVE', 'WAIT'))
        obs, _, terminated, truncated, info = env.step(step)
        tally.step(ram, step, env, info, fighter)
        if terminated or truncated:
            break
    success = info.get('ending') == 'exit-correct'
    if recorder is not None:
        (recorder.keep if success and tally.hits == 0 else recorder.drop)()
    return record(seed, start, info, tally, fighter, env)


def record(seed, start, info, tally, fighter, env) -> dict:
    return {'seed': seed, 'start': Path(start).name, 'ending': info.get('ending'),
            'success': info.get('ending') == 'exit-correct', 'hits': tally.hits,
            'hearts_lost': round(tally.lost, 2),
            'start_hearts': tally.start_health, 'frames': tally.frames, 'swings': fighter.swings,
            'bombs': len(tally.bomb_hurt), 'bomb_hurt': tally.bomb_hurt, 'bomb_pred': fighter.bomb_pred,
            'where': tally.where}


def starts(room: str, bombs: int, seed0: int, per_state: int):
    """(seed, start state) per visit exactly as `triforce evaluate --episodes-per-state per_state --seed seed0 --clips
    on` lays its episodes out on the case's task (`triforce.clips.episode_jobs`): the same entries for both."""
    from triforce._native import SpecialistConfig
    from triforce.clips import episode_jobs
    config = SpecialistConfig.load(str(task_file(room, bombs)))
    for state, seed, _ in episode_jobs(config, per_state, seed0, 'on'):
        yield seed, state


def _visits(task) -> list[dict]:
    room, bombs, visits, depth = task
    env = make_env(room, bombs)
    probe = make_env(room, bombs) if depth else None
    out = []
    for seed, start in visits:
        rec = play(env, room, seed, start, probe=probe, depth=depth)
        rec.update(room=room, bag=bombs, look=depth)
        out.append(rec)
    return out


def measure(room: str, bombs: int, seed0: int, per_state: int, depth: int = 0, progress=None, log: Path | None = None,
            workers: int = 1, skip: set | None = None) -> dict:
    """The visits' summary; `log`: each visit's record appended as a JSON line as it ends; `skip`: seeds already
    in the log (a piece run again)."""
    visits = [v for v in starts(room, bombs, seed0, per_state) if not skip or v[0] not in skip]
    tasks = [(room, bombs, visits[i:i + 2], depth) for i in range(0, len(visits), 2)]
    records = []
    with multiprocessing.get_context('spawn').Pool(workers) as pool:
        for part in pool.imap_unordered(_visits, tasks):
            records += part
            if log is not None:
                with open(log, 'a', encoding='utf-8') as out:
                    out.write(''.join(json.dumps(r) + '\n' for r in part))
            if progress:
                progress(len(records), len(visits))
    return summary(records)


def summary(records: list[dict]) -> dict:
    n = len(records)
    if not n:
        return {'visits': 0}
    won = [r for r in records if r['success']]
    bombs = [h for r in records for h in r['bomb_hurt']]
    where = collections.Counter(w['situation'] for r in records for w in r['where'])
    return {
        'visits': n, 'wins': len(won), 'win_rate': len(won) / n,
        'hits_per_visit': sum(r['hits'] for r in records) / n,
        'visits_hit': sum(r['hits'] > 0 for r in records),
        'hearts_lost_per_visit': sum(r['hearts_lost'] for r in records) / n,
        'frames_mean_won': sum(r['frames'] for r in won) / len(won) if won else None,
        'swings_per_visit': sum(r['swings'] for r in records) / n,
        'bombs_per_visit': sum(r['bombs'] for r in records) / n,
        'darknuts_hurt_per_bomb': sum(bombs) / len(bombs) if bombs else None,
        'bombs_hurting_2plus': sum(h >= 2 for h in bombs), 'bombs_hurting_1': sum(h == 1 for h in bombs),
        'bombs_hurting_0': sum(h == 0 for h in bombs),
        'endings': dict(collections.Counter(r['ending'] for r in records)),
        'hit_situations': dict(where),
    }


# --- demonstrations ----------------------------------------------------------------------------------------------------

BACK = 6                                                      # decisions the search backs up before a hit
TRIES = 400                                                   # the search's branch points tried a visit at most
LOOK_REACH = 48                                               # pixels within which a Darknut calls for the look-ahead


def near(ram: bytes, reach: int = LOOK_REACH) -> bool:
    """A Darknut out of its cloud within `reach` pixels of Link on both axes: where the look-ahead is worth its cost."""
    x, y = ram[OBJ_X], ram[OBJ_Y]
    return any(abs(d.x - x) < reach and abs(d.y - y) < reach for d in darknuts(ram) if not d.cloud)


def teach(env, room: str, seed: int, start: str, probe, depth: int, recorder=None) -> dict:
    """Building demonstrations only (moving the emulator ahead and back is allowed there, never at play): the
    fighter timed by its look-ahead (`depth`), and where Link is still hit, a backtracking search seeded by it: back
    `BACK` decisions before the hit and the next of the fighter's own ranked choices there, then the fighter on (at
    most `TRIES` branch points).  The way that leaves by the exit unhit is played again from the visit's reset as
    one episode (`replay`) and kept; none: the visit is dropped."""
    import copy
    t0 = time.time()
    obs, _ = env.reset(seed=seed, options={'start_state': start})
    fighter = Fighter(room, Floor(env.state.render_grid()))
    health = env.state.health
    trail = []                                                # per decision: (snapshot, fighter, obs, order, chosen)
    tries = 0
    info = {}
    terminated = truncated = False
    while len(trail) < LIMIT:
        ram = bytes(env.state.ram)
        grid = env.state.render_grid() if not darknuts(ram) else None
        ok = mask_ok(obs)
        order = fighter.ranked(ram, ok, grid)
        snapshot, route = env.snapshot(), env.route
        choice = 0
        if depth and len(order) > 1 and near(ram):
            best = None
            for i, candidate in enumerate(order[:6]):
                lasted = outlook(probe, snapshot, route, fighter, candidate, health, depth, grid)
                if best is None or lasted > best[0]:
                    best = (lasted, i)
                if lasted > depth:
                    break
            choice = best[1]
        trail.append([snapshot, route, copy.deepcopy(fighter), obs, order, choice])
        step = order[choice]
        fighter.note(ram, step)
        obs, _, terminated, truncated, info = env.step(step)
        if env.state.health < health:
            # back up: the latest decision before the hit within BACK with an untried choice left
            resumed = False
            while tries < TRIES:
                cut = None
                for back in range(len(trail) - 1, max(-1, len(trail) - 1 - BACK), -1):
                    if trail[back][5] + 1 < min(len(trail[back][4]), 6):
                        cut = back
                        break
                if cut is None:
                    break
                tries += 1
                del trail[cut + 1:]
                snapshot, route, saved, obs, order, choice = trail[cut]
                trail[cut][5] = choice + 1
                env.set_route(*route)
                obs, _ = env.restore(snapshot, 'search')
                fighter = copy.deepcopy(saved)
                ram = bytes(env.state.ram)
                step = order[choice + 1]
                fighter.note(ram, step)
                obs, _, terminated, truncated, info = env.step(step)
                if env.state.health >= health:
                    resumed = True
                    break
            if not resumed:
                break
        if terminated or truncated:
            if info.get('ending') == 'exit-correct':
                break
            break
    actions = [t[4][t[5]] for t in trail]
    rec = replay(env, room, seed, start, actions, recorder) if info.get('ending') == 'exit-correct' and \
        env.state.health >= health else None
    if rec is None:
        rec = {'seed': seed, 'start': Path(start).name, 'ending': info.get('ending'), 'success': False, 'hits': 1,
               'hearts_lost': 0.0, 'start_hearts': health, 'frames': 0, 'swings': 0, 'bombs': 0, 'bomb_hurt': [],
               'where': []}
    rec.update(seconds=round(time.time() - t0, 1), tries=tries, actions=len(actions))
    return rec


def replay(env, room, seed, start, actions, recorder=None) -> dict:
    """Plays `actions` from the visit's reset as one episode, as training would; with `recorder` its rows, kept only
    if it leaves by the exit unhit."""
    obs, _ = env.reset(seed=seed, options={'start_state': start})
    fighter = Fighter(room, Floor(env.state.render_grid()))
    tally = Tally(env)
    info = {}
    for text in actions:
        ram = bytes(env.state.ram)
        fighter.note(ram, text)
        if recorder is not None:
            kind = text.split()[0]
            recorder.add(obs, text, repeat=REPEAT.get(kind, 1), any_duration=kind in ('MOVE', 'WAIT'))
        obs, _, terminated, truncated, info = env.step(text)
        tally.step(ram, text, env, info, fighter)
        if terminated or truncated:
            break
    ok = info.get('ending') == 'exit-correct' and tally.hits == 0
    if recorder is not None:
        (recorder.keep if ok else recorder.drop)()
    return record(seed, start, info, tally, fighter, env)


def _taught(task):
    from triforce.demos import DemoRecorder
    room, bombs, visits, depth = task
    env, probe = make_env(room, bombs), make_env(room, bombs)
    recorder = DemoRecorder()
    out = []
    for seed, start in visits:
        rec = teach(env, room, seed, start, probe, depth, recorder)
        rec.update(room=room, bag=bombs, look=depth)
        out.append(rec)
    return out, recorder.rows


SAVE_EVERY = 20


def demos(out: Path, room: str, bombs: int, seed0: int, per_state: int, depth: int = 3, progress=None,
          log: Path | None = None, workers: int = 1) -> dict:
    """The demonstration builder over `starts`' visits (seeds from `seed0`): the clean visits' rows to `out`,
    written as the visits end (a job stopped at its time limit keeps what it found)."""
    from triforce.demos import save_demos

    def save():
        if rows:
            part = out.with_name(out.name + '.part')
            save_demos(part, [obs for obs, _ in rows], [action for _, action in rows])
            part.replace(out)

    visits = list(starts(room, bombs, seed0, per_state))
    tasks = [(room, bombs, [visit], depth) for visit in visits]
    records, rows = [], []
    saved = time.time()
    with multiprocessing.get_context('spawn').Pool(workers) as pool:
        for part, kept in pool.imap_unordered(_taught, tasks):
            records += part
            rows += kept
            if log is not None:
                with open(log, 'a', encoding='utf-8') as file:
                    file.write(''.join(json.dumps(r) + '\n' for r in part))
            if time.time() - saved > SAVE_EVERY:
                save()
                saved = time.time()
            if progress:
                progress(len(records), len(visits))
    save()
    return {'file': str(out), 'rows': len(rows), 'played': len(records),
            'kept': sum(r['success'] and r['hits'] == 0 for r in records), 'summary': summary(records)}


def trace(room: str, bombs: int, seed: int, start: str | None = None, depth: int = 0):
    env = make_env(room, bombs)
    probe = make_env(room, bombs) if depth else None
    start = start or dict(starts(room, bombs, 0, 100))[seed]
    obs, _ = env.reset(seed=seed, options={'start_state': start})
    fighter = Fighter(room, Floor(env.state.render_grid()))
    tally = Tally(env)
    for i in range(LIMIT):
        ram = bytes(env.state.ram)
        grid = env.state.render_grid() if not darknuts(ram) else None
        order = fighter.ranked(ram, mask_ok(obs), grid)
        step = order[0]
        if probe is not None and len(order) > 1 and darknuts(ram):
            snapshot, route = env.snapshot(), env.route
            best = None
            for candidate in order[:6]:
                lasted = outlook(probe, snapshot, route, fighter, candidate, tally.health, depth, grid)
                if best is None or lasted > best[0]:
                    best = (lasted, candidate)
                if lasted > depth:
                    break
            step = best[1]
        fighter.note(ram, step)
        nuts = ' '.join(f"{d.slot}:{d.x:3d},{d.y:3d}{d.dir or '-'}{'*' if d.flash else ''}{'c' if d.cloud else ''}"
                        f"h{d.hp}" for d in darknuts(ram))
        obs, _, terminated, truncated, info = env.step(step)
        hits = tally.hits
        tally.step(ram, step, env, info, fighter)
        lost = f"  HIT {tally.where[-1]['situation']}" if tally.hits > hits else ''
        print(f"{i:4d} f{tally.frames:5d} link {ram[OBJ_X]:3d},{ram[OBJ_Y]:3d} b{ram[INV_BOMBS]} t{fighter.target} "
              f"{step:12s} | {nuts}{lost}")
        if terminated or truncated:
            print(info.get('ending'), 'frames', tally.frames, 'hits', tally.hits, 'bombs', tally.bomb_hurt,
                  'predicted', fighter.bomb_pred)
            break


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    for name in ('measure', 'demos', 'trace'):
        p = sub.add_parser(name)
        if name == 'demos':
            p.add_argument('out')
        p.add_argument('--room', choices=sorted(ROOMS), required=True)
        p.add_argument('--bombs', type=int, required=True, help="the bag at every reset")
        if name == 'trace':
            p.add_argument('seed', type=int)
            p.add_argument('--start', default=None)
            p.add_argument('--look', type=int, default=0)
            continue
        p.add_argument('--per-state', type=int, default=100, help="visits from each entry state (evaluate's --episodes-per-state)")
        p.add_argument('--seed0', type=int, default=0 if name == 'measure' else 100000)
        p.add_argument('--json', default=None, help="append each visit's record to this JSON-lines file")
        p.add_argument('--workers', type=int, default=1)
        p.add_argument('--look', type=int, default=0 if name == 'measure' else 3, metavar='DEPTH')
    u = sub.add_parser('summarize')
    u.add_argument('files', nargs='+')
    g = sub.add_parser('merge')
    g.add_argument('out')
    g.add_argument('files', nargs='+')
    args = parser.parse_args(argv)
    if args.cmd == 'trace':
        trace(args.room, args.bombs, args.seed, args.start, args.look)
        return 0
    if args.cmd == 'summarize':
        records = [json.loads(line) for f in args.files for line in Path(f).read_text(encoding='utf-8').splitlines()]
        print(json.dumps(summary(records), indent=1))
        return 0
    if args.cmd == 'merge':
        parts = []
        for path in args.files:
            with np.load(path) as data:
                parts.append({k: data[k] for k in data.files})
        if any(sorted(part) != sorted(parts[0]) for part in parts):
            raise SystemExit("merge: the files' keys differ")
        with open(args.out, 'wb') as file:
            np.savez_compressed(file, **{k: np.concatenate([part[k] for part in parts]) for k in parts[0]})
        print(f"{args.out}: {sum(len(part['actions']) for part in parts)} rows of {len(parts)} files")
        return 0
    from triforce import jobs
    jobs.require_job()
    log = Path(args.json) if args.json else None
    if args.cmd == 'measure':
        skip = {json.loads(line)['seed'] for line in log.read_text().splitlines()} if log and log.exists() else None
        result = measure(args.room, args.bombs, args.seed0, args.per_state, args.look,
                         lambda done, total: jobs.progress(done, total, 'visits'), log, args.workers, skip)
        print(json.dumps(result, indent=1))
        return 0
    result = demos(Path(args.out), args.room, args.bombs, args.seed0, args.per_state, args.look,
                   lambda done, total: jobs.progress(done, total, 'visits'), log, args.workers)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
