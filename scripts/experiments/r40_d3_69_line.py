"""Round 40 (docs/experiments/round40.md, "3:0x69 crossed by a line"): the owner's line through Dungeon 3's room 0x69
(eight Red Darknuts) as a scripted rule over the environment's own actions, its measurement and its demonstrations.

    python scripts/experiments/r40_d3_69_line.py measure --pass in|out [--runs N] [--seed0 S] [--look D] [--json OUT]
    python scripts/experiments/r40_d3_69_line.py demos OUT.npz --pass in|out [--runs N] [--seed0 S] [--look D]
        [--json OUT] [--workers W] [--skip SEED...]
    python scripts/experiments/r40_d3_69_line.py trace --pass in|out SEED
    python scripts/experiments/r40_d3_69_line.py summarize RECORDS.jsonl...
    python scripts/experiments/r40_d3_69_line.py merge OUT.npz PIECE.npz...

The line (the owner, decisions 2026-11-23 "3:0x69 is crossed by a line"): "Going in the door, immediate east until
you can't, south to line up the stairs, then east to stairs. [...] If you time it right, you can sword the darknut
to safely pass through him." In by the N door: S onto the floor (Y $5D), E along the top row to the block wall (X
$90), S down that column to the stairs' row (Y $8D), E to the stairs (X $D0; tiles 26-27, 10-11).  Out: the game
puts Link back at ($60, $8D) facing S (the cellar's attribute C, asm:Warp_CheckSubroomExit), N up X $60 to the top
row, E to the door's column X $78 and N out.  game.yaml's `d3first` `path_costs` for 3:0x69 hold the objective's way
on the same tiles.

The rule decides once per environment action from what a player sees (the RAM the env exposes: positions,
facings, flash timers, knockback, spawn clouds; `Line.ranked`):

- **The line while it is safe.** MOVE along the line when no way on from it (standing, along the line, back along
  it, straight any way) is predicted to be touched within `HORIZON` frames.  The prediction: each Darknut walks its
  facing at 0.5 pixel a frame (asm:Darknut_Init, q-speed `$20`) and turns only on a grid point, toward Link when he
  is within 8 pixels of its row or column (asm:Wanderer_MoveAndTurn) or on; a knockback carries it 4 pixels a frame
  (asm:Shove_Update); touch is under 9 pixels on both axes (asm:Link_CheckMonsterTouch_NoReset), the box widening
  with the frames.
- **Sword through.** A Darknut in a strike's blade box (`BLADE`, measured in the game for each way), out of its
  cloud, not flashing, not facing the blade (its shield: a hit is refused when the weapon's direction and the
  Darknut's are opposites, asm:Damage_HandleWeaponHit@CheckDarknut) and not about to turn on a grid point may be
  struck (SWORD, BEAMS where the mask offers that instead).  A struck Darknut flashes about 32 frames
  (`ObjInvincibilityTimer` `$10`, decremented every second frame, asm:Shove_Begin) and while it flashes the game
  tests neither Link's weapons nor its touch on Link (asm:Collision_CheckMonsterVsWeaponsAndLink): Link walks
  through it.
- **Wait or step aside.** Otherwise the action (MOVE any way, WAIT, a strike) whose best way on is touched latest,
  then the shortest way left along the line.

`--look D` (training only: the owner, Nov 23, "moving the emulator ahead and back is fine in training only") times
the rule with the emulator: each decision takes the first of the rule's ranked actions that a second env, playing
it and going on the same way (a strike: then along the line) for D decisions, finds unhit; none: the one hit
latest.  The rule alone reacts to what it sees and is hit where a Darknut turns into Link at a grid point; the
look-ahead picks among the same actions the moment that works.

`measure` plays the rule from the route step's entry states (`d3first_3_69n_*` in, `d3first_3_69c_*` out) through
the darknut task's environment (`specialists/d3first/darknut.yaml`), seeds `--seed0`.. drawn as a training reset
draws them (the start's random numbers and spawn draw, the task's hp draw), with 4 or 8 bombs (the route's: none is
used here), and on the way out a third of the entries from the chains' harvested handoffs (the task's
`harvested_fraction`; their own hearts and bombs).  Each visit is put in one situation: `clear` (only the line's
moves), `sworded` (it struck a Darknut on the way) or `waited` (it held or stepped aside, no strike).  It prints and
writes the success, the hits taken a visit and the frames, by situation, and where each hit was taken.

`demos` is the demonstration builder (Line40b): every visit of one pass, drawn as `measure` draws them, is played by
a best-first search over the env's own actions for a way to the exit on which Link is never hit (`search`: moving
the emulator ahead and back, allowed when building demonstrations, never at play), the way found played again from
the visit's reset; where the search finds none, by the rule timed by the look-ahead (`teach`).  The visits that
leave by the exit unhit are written as demonstration rows (`triforce.demos`, what `train --demos` and a task's
`demos:` read): a MOVE or WAIT row is cloned without its duration, a strike `REPEAT` times (one decision among
dozens).  Each visit's record (who found it, its actions, the search's decisions and seconds) goes to `--json`;
`summarize` reads them.  The file goes under the checkout's `target/demos/` and from there into the store's demos
snapshot (the lead's `triforce-lead data install`).  The rule stays as it was, for reference and as the fallback.

It plays many runs: a job of the machine's queue (``jobs.require_job``: ``triforce job run ... --timeout 10m --
.venv/bin/python scripts/experiments/r40_d3_69_line.py ...``).
"""

import argparse
import collections
import heapq
import itertools
import json
import multiprocessing
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / 'specialists' / 'd3first' / 'darknut.yaml'
OBJ_X, OBJ_Y, OBJ_DIR, OBJ_TYPE, OBJ_META, OBJ_INVINC = 0x70, 0x84, 0x98, 0x34F, 0x405, 0x4F0
OBJ_SHOVE_DIR, OBJ_SHOVE_LEFT = 0xC0, 0xD3         # asm:ObjShoveDir, asm:ObjShoveDistance
RED_DARKNUT = 0x0B
VEC = {'N': (0, -1), 'S': (0, 1), 'W': (-1, 0), 'E': (1, 0)}
GAME_DIR = {1: 'E', 2: 'W', 4: 'S', 8: 'N'}
OPPOSITE = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}
SIDEWAYS = {'N': 'WE', 'S': 'WE', 'W': 'NS', 'E': 'NS'}
LINK_SPEED = 1.4             # pixels a frame (a MOVE 4 walks 5-6)
DARKNUT_SPEED = 0.5          # asm:Darknut_Init, red q-speed $20
TOUCH = 9                    # asm:Link_CheckMonsterTouch_NoReset: both axis distances under 9
MARGIN = 1                   # pixels added to the touch box after the first frames
HORIZON = 48                 # frames looked ahead
GROW = 0.08                  # pixels a frame the touch box widens by over the horizon
CLOUD = 8                    # frames a spawn cloud is assumed harmless
SWING_FRAMES = 17            # what a SWORD takes through the env (measured)
SWING_HIT = 3                # the frame of the swing its blade first tests the monsters (measured)
FLASH = 32                   # a struck monster's untouchable frames (timer $10 every second frame, asm:Shove_Begin)
SHOVE = 0x40                 # an unparried hit's knockback, pixels in the weapon's way (asm:Shove_Begin)
# Where a Darknut's top-left must be, relative to Link's, for a strike that way to touch it (measured in the game,
# Link on a grid point; the core of each box): (dx from, to, dy from, to).
BLADE = {'N': (-12, 6, -20, -4), 'S': (-10, 8, 2, 24), 'W': (-22, -4, -6, 6), 'E': (4, 22, -6, 6)}
REPEAT = {'SWORD': 8, 'BEAMS': 8}
STALL = 40                   # decisions in one place before the rule stops choosing to stand (the env ends at 50)

# The line's points, Link's top-left (X, Y): the entry spot, then each corner.
LINES = {
    'in': [(0x78, 0x4D), (0x78, 0x5D), (0x90, 0x5D), (0x90, 0x8D), (0xD0, 0x8D)],
    'out': [(0x60, 0x8D), (0x60, 0x5D), (0x78, 0x5D), (0x78, 0x3D)],
}
STATES = {'in': 'd3first_3_69n', 'out': 'd3first_3_69c'}


class Floor:
    """Where a 16 x 16 body (Link's top-left, or a Darknut's) may stand in the room, from the state's tile grid
    (`GameState.render_grid`: '#' is a wall or block); the N doorway is Link's alone."""

    def __init__(self, grid: str):
        rows = grid.splitlines()[1:]
        self.cells = [[c != '#' for c in row[4:]] for row in rows]

    def open(self, x: float, y: float, link: bool = True) -> bool:
        if not (0x20 <= x <= 0xD0 and 0x5D <= y <= 0xBD):
            return link and x == 0x78 and 0x3D <= y < 0x5D
        c0, r0 = int(x) // 8, (int(y) - 0x3D) // 8
        c1, r1 = (int(x) + 15) // 8, (int(y) - 0x3D + 15) // 8
        return all(self.cells[r][c] for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                   if 0 <= r < len(self.cells) and 0 <= c < len(self.cells[r]))


def on_grid(x: float, y: float) -> bool:
    """A walker's grid point: 16-pixel squares, Y offset $0D (asm:Wanderer_MoveAndTurn turns only there)."""
    return x % 16 == 0 and (y - 0x0D) % 16 == 0


class Darknut:
    """One Red Darknut as a player sees it: position, facing, flash timer, knockback, spawn cloud (the RAM)."""
    __slots__ = ('slot', 'x', 'y', 'dir', 'flash', 'cloud', 'shove_dir', 'shove')

    def __init__(self, ram: bytes, slot: int):
        self.slot = slot
        self.x, self.y = ram[OBJ_X + slot], ram[OBJ_Y + slot]
        self.dir = GAME_DIR.get(ram[OBJ_DIR + slot] & 0x0F)
        self.flash = 2 * ram[OBJ_INVINC + slot]           # frames it stays untouchable
        self.cloud = ram[OBJ_META + slot] != 0            # spawn cloud (or dying): no collision yet
        self.shove_dir = GAME_DIR.get(ram[OBJ_SHOVE_DIR + slot] & 0x0F)
        self.shove = ram[OBJ_SHOVE_LEFT + slot] if self.shove_dir else 0

    def to_grid(self) -> int:
        """Frames to the next grid point along its facing (0: on one)."""
        if self.dir is None:
            return 0
        along = self.x if self.dir in 'EW' else self.y - 0x0D
        ahead = (16 - along % 16) % 16 if self.dir in 'ES' else along % 16
        return int(ahead / DARKNUT_SPEED)

    def futures(self, floor: Floor, link: tuple[int, int], struck: str | None = None):
        """[(positions [HORIZON + 1, 2], harmless [HORIZON + 1])]: the ways it may go.  A knockback first (4 pixels
        a frame, asm:Shove_Update, until its distance is spent or a wall stops it); then it walks its facing at 0.5
        pixel a frame and turns only on a grid point: toward Link when he is within 8 pixels of its row or column
        there (asm:Wanderer_MoveAndTurn, turn rate $80), else on (its turn timer's sideways turns are rare and left
        out), any open way at a wall.  `struck` (the strike's way): Link's blade lands at `SWING_HIT`; it flashes
        `FLASH` frames and is knocked `SHOVE` pixels that way, along its own axis or from a grid point (a knockback
        across its axis between grid points is dropped, asm:Shove_Update)."""
        harmless = np.zeros(HORIZON + 1, bool)
        harmless[:min(self.flash, HORIZON + 1)] = True
        if self.cloud:
            harmless[:CLOUD] = True
        if struck:
            harmless[SWING_HIT:SWING_HIT + FLASH] = True
        heading = None if self.cloud else self.dir
        x, y = float(self.x), float(self.y)
        shove_dir, shove = self.shove_dir, self.shove
        path = [(x, y)]
        while len(path) <= HORIZON:
            if struck and len(path) == SWING_HIT and (heading in (struck, OPPOSITE[struck]) or on_grid(x, y)):
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
                if len(path) > 1 and on_grid(x, y):
                    break                                     # a grid point: where it may turn
                nx, ny = x + VEC[heading][0] * DARKNUT_SPEED, y + VEC[heading][1] * DARKNUT_SPEED
                if floor.open(nx, ny, link=False):
                    x, y = nx, ny
            path.append((x, y))
        if len(path) > HORIZON:
            return [(np.array(path), harmless)]
        ways = []
        if floor.open(x + VEC[heading][0] * 8, y + VEC[heading][1] * 8, link=False):
            ways.append(heading)
        else:
            ways += [d for d in SIDEWAYS[heading] if floor.open(x + VEC[d][0] * 8, y + VEC[d][1] * 8, link=False)]
        lx, ly = link
        toward = ('S' if ly > y else 'N') if abs(lx - x) < 8 else ('E' if lx > x else 'W') if abs(ly - y) < 8 \
            else None
        if toward and toward not in ways and floor.open(x + VEC[toward][0] * 8, y + VEC[toward][1] * 8, link=False):
            ways.append(toward)
        out = []
        for way in ways or [heading]:
            vx, vy = VEC[way]
            rest = [(x + vx * DARKNUT_SPEED * f, y + vy * DARKNUT_SPEED * f) for f in range(1, HORIZON + 2 - len(path))]
            out.append((np.array(path + rest), harmless))
        return out


def darknuts(ram: bytes) -> list[Darknut]:
    return [Darknut(ram, s) for s in range(1, 12) if ram[OBJ_TYPE + s] == RED_DARKNUT]


def togo(which: str, x: float, y: float) -> float:
    """How far Link at (x, y) is from the end of the line: along it from the nearest point of it, plus twice the
    way back onto it."""
    points = LINES[which]
    best = None
    for i, ((ax, ay), (bx, by)) in enumerate(zip(points, points[1:])):
        px, py = min(max(x, min(ax, bx)), max(ax, bx)), min(max(y, min(ay, by)), max(ay, by))
        rest = abs(bx - px) + abs(by - py) + sum(abs(cx - dx) + abs(cy - dy)
                                                 for (cx, cy), (dx, dy) in zip(points[i + 1:], points[i + 2:]))
        value = 2 * (abs(px - x) + abs(py - y)) + rest
        best = value if best is None else min(best, value)
    return best


class Line:
    """The rule for one visit: `decide(ram, ok)` -> the env action; `ok(text)` says whether the mask offers it."""

    def __init__(self, which: str, floor: Floor):
        self.which, self.floor = which, floor
        self.struck = self.waited = 0
        self.still, self.last = 0, None

    def heading(self, x: float, y: float, back: bool = False) -> str:
        """The way along the line from (x, y) (`back`: the way back along it): toward the next point of the segment
        nearest him."""
        points = LINES[self.which][::-1] if back else LINES[self.which]
        best = None
        for i, ((ax, ay), (bx, by)) in enumerate(zip(points, points[1:])):
            px, py = min(max(x, min(ax, bx)), max(ax, bx)), min(max(y, min(ay, by)), max(ay, by))
            off = abs(px - x) + abs(py - y)
            if best is None or off < best[0] or (off == best[0] and i > best[1]):
                best = (off, i, px, py, ax == bx, bx, by)
        off, _, px, py, upright, bx, by = best
        if off > 1:                                           # back onto the line, across it
            return ('E' if px > x else 'W') if abs(px - x) > 1 else ('S' if py > y else 'N')
        if upright:
            return 'S' if by > y else 'N'
        return 'E' if bx > x else 'W'

    def walk(self, x, y, plan):
        """Link's positions frame by frame (HORIZON + 1 of them) for `plan` [(way, frames)]: a direction, None
        (standing), 'line' or 'back' (along the line, either way, turning at its corners); he stands where the
        floor ends."""
        path, px, py = [(x, y)], float(x), float(y)
        for step, frames in plan:
            for _ in range(frames):
                if len(path) > HORIZON:
                    break
                way = self.heading(px, py, back=step == 'back') if step in ('line', 'back') else step
                if way:
                    dx, dy = VEC[way]
                    nx, ny = px + dx * LINK_SPEED, py + dy * LINK_SPEED
                    if self.floor.open(nx, ny):
                        px, py = nx, ny
                path.append((px, py))
        while len(path) <= HORIZON:
            path.append(path[-1])
        return np.array(path)

    @staticmethod
    def contact(paths: np.ndarray, futures) -> np.ndarray:
        """Per Link path ([P, HORIZON + 1, 2]): the first frame some way of some Darknut touches him (HORIZON + 1:
        none); the touch box widens with the frames (`GROW`), for what the model of their walk misses."""
        frames = np.arange(HORIZON + 1)
        box = np.where(frames < 4, TOUCH, TOUCH + MARGIN + GROW * frames)
        first = np.full(len(paths), HORIZON + 1)
        for ways in futures:
            for pos, harmless in ways:
                d = np.abs(paths - pos[None])
                touch = (d[..., 0] < box) & (d[..., 1] < box) & ~harmless[None]
                touch[:, 0] = False
                hit = np.where(touch, frames[None], HORIZON + 1).min(axis=1)
                first = np.minimum(first, hit)
        return first

    def strikes(self, x, y, facing, nuts, ok) -> list[tuple[str, list, Darknut]]:
        """The strikes that would hurt a Darknut, as (action, Link's frames, the Darknut): the Darknut in that way's
        blade box, out of its cloud, not flashing, not facing the blade (its shield,
        asm:Damage_HandleWeaponHit@CheckDarknut), and not reaching a grid point (where it may turn its shield
        round) before the blade lands.  A strike across the way Link faces, off the 8-pixel grid, first walks him
        onto the nearest grid point on that way (docs/architecture.md "Attacks by buttons"); the box is measured
        from there."""
        out = []
        for way, (x0, x1, y0, y1) in BLADE.items():
            text = next((f'{kind} {way}' for kind in ('SWORD', 'BEAMS') if ok(f'{kind} {way}')), None)
            if text is None:
                continue
            sx, sy, lead = x, y, []
            if facing in VEC and (way in 'NS') != (facing in 'NS'):
                if facing in 'NS':
                    off = (y - 0x0D) % 8
                    sy = y + (8 - off if off >= 4 else -off)
                    shift = sy - y
                    lead = [('S' if shift > 0 else 'N', round(abs(shift) / LINK_SPEED))] if shift else []
                else:
                    off = x % 8
                    sx = x + (8 - off if off >= 4 else -off)
                    shift = sx - x
                    lead = [('E' if shift > 0 else 'W', round(abs(shift) / LINK_SPEED))] if shift else []
            for d in nuts:
                if d.cloud or d.flash or d.dir is None or d.dir == OPPOSITE[way]:
                    continue
                if not (x0 <= d.x - sx <= x1 and y0 <= d.y - sy <= y1):
                    continue
                if 0 < d.to_grid() <= SWING_HIT + 1 + sum(n for _, n in lead) or \
                        (d.to_grid() == 0 and on_grid(d.x, d.y)):
                    continue
                out.append((text, lead + [(None, SWING_FRAMES)], d))
                break
        return out

    def ranked(self, ram: bytes, ok) -> list[str]:
        """The actions the rule would take here, best first: the line while it is safe; else the one whose best way
        on is touched latest (none within the horizon: safe), then the shortest way left, then the order MOVE along
        the line, the other moves, WAIT, the strikes."""
        x, y = ram[OBJ_X], ram[OBJ_Y]
        heading = self.heading(x, y)
        forward = f'MOVE {heading} 4'
        if not self.floor.open(x, y):                         # out through the door
            return [forward]
        nuts = darknuts(ram)
        options = []                                          # (text, Link's first frames, the Darknut struck)
        for step in [heading] + [d for d in VEC if d != heading] + [None]:
            text = f'MOVE {step} 4' if step else 'WAIT 4'
            if ok(text) and not (step is None and self.still >= STALL):
                options.append((text, [(step, 4)], None))
        options += self.strikes(x, y, GAME_DIR.get(ram[OBJ_DIR] & 0x0F), nuts, ok)
        plain = [d.futures(self.floor, (x, y)) for d in nuts]
        scored = []                                           # (safe frames, way left, rank, text)
        for rank, (text, first, struck) in enumerate(options):
            futures = plain if struck is None else \
                [d.futures(self.floor, (x, y), struck=text.split()[1]) if d is struck else f
                 for d, f in zip(nuts, plain)]
            plans = [first + [(then, HORIZON)] for then in [None, 'line', 'back'] + list(VEC)]
            paths = np.stack([self.walk(x, y, plan) for plan in plans])
            if text.startswith('MOVE') and np.allclose(paths[0, 4], (x, y)):
                continue                                      # a wall
            first_touch = self.contact(paths, futures)
            ends = [togo(self.which, *paths[i, -1]) for i in range(len(plans))]
            i = min(range(len(plans)), key=lambda i: (-first_touch[i], ends[i]))
            scored.append((int(first_touch[i]), ends[i], rank, text))
        if not scored:
            return ['WAIT 4']
        order = [s[3] for s in sorted(scored, key=lambda s: (-s[0], s[1], s[2]))]
        if any(s[3] == forward and s[0] > HORIZON for s in scored):
            order.remove(forward)
            order.insert(0, forward)                          # the line itself whenever it is safe
        return order

    def note(self, ram: bytes, text: str):
        """Keeps the visit's counts for the action taken (`ranked`'s first, or another the look-ahead chose)."""
        x, y = ram[OBJ_X], ram[OBJ_Y]
        self.still = self.still + 1 if (x, y) == self.last else 0
        self.last = (x, y)
        if text.startswith(('SWORD', 'BEAMS')):
            self.struck += 1
        elif self.floor.open(x, y) and text != f'MOVE {self.heading(x, y)} 4':
            self.waited += 1

    def decide(self, ram: bytes, ok) -> str:
        """The rule alone: `ranked`'s first."""
        text = self.ranked(ram, ok)[0]
        self.note(ram, text)
        return text


def mask_ok(obs):
    from triforce.demos import action_indices, legal
    mask = obs['action_mask']
    return lambda text: legal(mask, action_indices(text))


def make_env(bombs: int, states_dir=None):
    """The darknut task's env (`TASK`) with exactly `bombs` bombs at every reset (the route brings 4 or 8)."""
    from triforce._native import Env, SpecialistConfig
    config = SpecialistConfig.load(str(TASK))
    ram = dict(config.to_dict()['ram_randomization'], bombs=[bombs, bombs], bombs_share=1.0)
    return Env(config.replace(ram_randomization=ram), explicit_starts=True, strict_actions=False, seed=0,
               states_dir=None if states_dir is None else str(states_dir))


def harvested(which: str) -> list[str]:
    from triforce import paths
    folder = paths.data_dir('harvested') / 'darknut'
    return sorted(str(p) for p in folder.glob(f'{STATES[which]}_h*.state'))


def outlook(probe, snapshot, route, rule: Line, first: str, health: float, depth: int) -> int:
    """Training-time look-ahead (the owner, Nov 23: moving the emulator ahead and back is fine in training only):
    `probe` (a second env) plays `first` from `snapshot`, then goes on the same way (a strike: then along the line)
    for up to `depth` decisions in all; returns how many it played before Link was hit (`depth` + 1: none, or the
    step left the room by its exit)."""
    probe.set_route(*route)
    obs, _ = probe.restore(snapshot, 'look-ahead')
    step = first
    for k in range(depth):
        if k and step.startswith(('SWORD', 'BEAMS')):
            ram = bytes(probe.state.ram)
            step = f'MOVE {rule.heading(ram[OBJ_X], ram[OBJ_Y])} 4'
        obs, _, terminated, truncated, info = probe.step(step)
        if probe.state.health < health:
            return k
        if terminated or truncated:
            return depth + 1 if info.get('ending') == 'exit-correct' else k
    return depth + 1


def play(env, which: str, seed: int, start: str, recorder=None, probe=None, depth: int = 0):
    """One visit by the rule; returns its record (and, with `recorder`, adds its rows, kept or dropped).  With
    `probe` and `depth`, each decision takes the first of the rule's ranked actions whose `outlook` is clean (none:
    the one hit latest)."""
    obs, _ = env.reset(seed=seed, options={'start_state': start})
    rule = Line(which, Floor(env.state.render_grid()))
    health, hits, frames, info = env.state.health, 0, 0, {}
    swings, where, looked = 0, [], 0
    for _ in range(400):
        ram = bytes(env.state.ram)
        ok = mask_ok(obs)
        order = rule.ranked(ram, ok)
        step = order[0]
        if probe is not None and depth and len(order) > 1:
            snapshot, route = env.snapshot(), env.route
            best = None
            for candidate in order:
                looked += 1
                lasted = outlook(probe, snapshot, route, rule, candidate, health, depth)
                if best is None or lasted > best[0]:
                    best = (lasted, candidate)
                if lasted > depth:
                    break
            step = best[1]
        rule.note(ram, step)
        if recorder is not None and ok(step):
            kind = step.split()[0]
            recorder.add(obs, step, repeat=REPEAT.get(kind, 1), any_duration=kind in ('MOVE', 'WAIT'))
        obs, _, terminated, truncated, info = env.step(step)
        swings += step.startswith(('SWORD', 'BEAMS'))
        frames += info.get('frames', 0)
        now = env.state.health
        if now < health:
            hits += 1
            x, y = ram[OBJ_X], ram[OBJ_Y]
            near = min(darknuts(ram), key=lambda d: abs(d.x - x) + abs(d.y - y), default=None)
            where.append({'link': [x, y], 'action': step, 'struck_before': rule.struck,
                          'darknut': near and [near.x, near.y, near.dir, near.flash]})
        health = now
        if terminated or truncated:
            break
    success = info.get('ending') == 'exit-correct'
    situation = 'sworded' if rule.struck else 'waited' if rule.waited else 'clear'
    if recorder is not None:
        (recorder.keep if success and hits == 0 else recorder.drop)()
    return {'seed': seed, 'start': Path(start).name, 'ending': info.get('ending'), 'success': success,
            'hits': hits, 'frames': frames, 'situation': situation, 'swings': swings, 'where': where,
            'looked': looked}


SEARCH_ACTIONS = ('MOVE N 4', 'MOVE S 4', 'MOVE W 4', 'MOVE E 4', 'WAIT 4', 'SWORD N', 'SWORD S', 'SWORD W', 'SWORD E')
SEARCHES = ((2.0, 25000), (1.0, 50000))   # (the heuristic's weight, decisions tried): the search, then a wider one
REACH = 32                   # pixels (both axes) within which a Darknut may be struck: a strike farther is not tried
SAVE_EVERY = 20              # seconds between the builder's saves of its file


def search(env, which: str, obs, weight: float, budget: int):
    """Building demonstrations only (the owner, Nov 23: moving the emulator ahead and back is allowed when building
    demonstrations and in training, never at play): a best-first search over the env's own actions from the env's
    current state for a way to the pass's exit on which Link is never hit.  A state's priority is the frames spent
    plus `weight` times the frames left at Link's pace along the line (`togo`, which counts a pixel off the line
    twice); expanding it plays every action the mask offers (`SEARCH_ACTIONS`; a SWORD the mask offers as BEAMS at
    full hearts is played so; a strike only with a Darknut within `REACH`), drops every child whose hearts fell and
    every situation seen before (Link's position and facing, each Darknut's position, facing, flash and knockback).
    Returns (the actions or None, the decisions tried, at most `budget`)."""
    def near(ram: bytes, nuts) -> bool:
        return any(abs(d.x - ram[OBJ_X]) < REACH and abs(d.y - ram[OBJ_Y]) < REACH and not d.cloud for d in nuts)

    ram = bytes(env.state.ram)
    order = itertools.count()
    frontier = [(0.0, next(order), env.snapshot(), [], obs, env.state.health, 0, near(ram, darknuts(ram)))]
    seen = set()
    tried = 0
    while frontier and tried < budget:
        _, _, snapshot, done, before, health, spent, strike = heapq.heappop(frontier)
        ok = mask_ok(before)
        for text in SEARCH_ACTIONS:
            if text.startswith('SWORD'):
                if not strike:
                    continue
                if not ok(text):
                    text = text.replace('SWORD', 'BEAMS')
            if not ok(text):
                continue
            env.restore(snapshot, 'search')
            after, _, terminated, truncated, info = env.step(text)
            tried += 1
            state = env.state
            if state.health < health:
                continue
            if terminated or truncated:
                if info.get('ending') == 'exit-correct':
                    return done + [text], tried
                continue
            ram = bytes(state.ram)
            nuts = darknuts(ram)
            key = (ram[OBJ_X], ram[OBJ_Y], ram[OBJ_DIR]) + tuple((d.x, d.y, d.dir, d.flash > 0, d.shove) for d in nuts)
            if key in seen:
                continue
            seen.add(key)
            frames = spent + info.get('frames', 0)
            priority = frames + weight * togo(which, ram[OBJ_X], ram[OBJ_Y]) / LINK_SPEED
            heapq.heappush(frontier, (priority, next(order), env.snapshot(), done + [text], after, state.health, frames,
                                      near(ram, nuts)))
    return None, tried


def replay(env, which: str, seed: int, start: str, actions: list[str], recorder=None) -> dict:
    """Plays `actions` from the visit's reset (the same seed and start: the same entry) as one episode, as training
    and play would, and returns its record; with `recorder` its rows, kept only if it leaves by the exit unhit."""
    obs, _ = env.reset(seed=seed, options={'start_state': start})
    health, hits, frames, info, swings = env.state.health, 0, 0, {}, 0
    for text in actions:
        if recorder is not None:
            kind = text.split()[0]
            recorder.add(obs, text, repeat=REPEAT.get(kind, 1), any_duration=kind in ('MOVE', 'WAIT'))
        obs, _, terminated, truncated, info = env.step(text)
        swings += text.startswith(('SWORD', 'BEAMS'))
        frames += info.get('frames', 0)
        hits += env.state.health < health
        health = env.state.health
        if terminated or truncated:
            break
    success = info.get('ending') == 'exit-correct'
    if recorder is not None:
        (recorder.keep if success and hits == 0 else recorder.drop)()
    return {'seed': seed, 'start': Path(start).name, 'ending': info.get('ending'), 'success': success, 'hits': hits,
            'frames': frames, 'situation': 'sworded' if swings else 'clear', 'swings': swings, 'where': [],
            'actions': len(actions), 'waits': sum(a.startswith('WAIT') for a in actions),
            'way': ' '.join(a.replace('MOVE ', '').replace(' 4', '').replace(' ', '_') for a in actions)}


def teach(env, which: str, seed: int, start: str, recorder=None, probe=None, depth: int = 4) -> dict:
    """The demonstration builder's visit: the search (`search`, then a wider one, `SEARCHES`), the way found played
    again from the reset (`replay`); with no way found, the rule timed by the look-ahead (`play`).  Kept only when it
    leaves by the exit unhit."""
    t0 = time.time()
    tried = 0
    for weight, budget in SEARCHES:
        obs, _ = env.reset(seed=seed, options={'start_state': start})
        full = env.state.health >= env.state.max_health
        actions, n = search(env, which, obs, weight, budget)
        tried += n
        if actions is not None:
            record = replay(env, which, seed, start, actions, recorder)
            record.update(teacher='search', weight=weight)
            break
    else:
        record = play(env, which, seed, start, recorder, probe, depth)
        record.update(teacher='rule', weight=None, actions=None)
    record.update(seconds=time.time() - t0, tried=tried, full=full)
    return record


def starts(which: str, seed0: int, runs: int):
    """(seed, start state, bombs) per visit: the entry states round robin, a third of the out-pass from the
    harvested handoffs (drawn by the seed), 4 or 8 bombs by the seed's parity."""
    pool = harvested(which) if which == 'out' else []
    for seed in range(seed0, seed0 + runs):
        draw = random.Random(seed)
        if pool and draw.random() < 0.33:
            yield seed, draw.choice(pool), 4 + 4 * (seed % 2)
        else:
            yield seed, f'{STATES[which]}_{seed % 3}', 4 + 4 * (seed % 2)


def _visits(task) -> list[dict]:
    """One worker's share of `measure`: (which, [(seed, start, bombs)], depth) -> the records."""
    which, visits, depth = task
    envs = {4: make_env(4), 8: make_env(8)}
    probes = {4: make_env(4), 8: make_env(8)} if depth else {}
    out = []
    for seed, start, bombs in visits:
        record = play(envs[bombs], which, seed, start, probe=probes.get(bombs), depth=depth)
        record.update(harvested=start.endswith('.state'), bombs=bombs, look=depth, which=which)
        out.append(record)
    return out


def measure(which: str, seed0: int, runs: int, depth: int = 0, progress=None, log: Path | None = None,
            workers: int = 1) -> dict:
    """The visits' summary; `log`: each visit's record appended as a JSON line as it ends (`summarize` reads them);
    `workers` processes share the visits, 10 at a time."""
    visits = list(starts(which, seed0, runs))
    tasks = [(which, visits[i:i + 10], depth) for i in range(0, len(visits), 10)]
    records = []
    with multiprocessing.get_context('spawn').Pool(workers) as pool:
        for part in pool.imap(_visits, tasks):
            records += part
            if log is not None:
                with open(log, 'a', encoding='utf-8') as out:
                    out.write(''.join(json.dumps(record) + '\n' for record in part))
            if progress:
                progress(len(records), runs)
    return summary(which, records)


def summary(which: str, records: list[dict]) -> dict:
    def stats(rows):
        n = len(rows)
        if not n:
            return {'visits': 0}
        won = [r for r in rows if r['success']]
        return {'visits': n, 'success': sum(r['success'] for r in rows) / n,
                'hits_per_visit': sum(r['hits'] for r in rows) / n,
                'hit_visits': sum(r['hits'] > 0 for r in rows),
                'frames_mean_won': sum(r['frames'] for r in won) / len(won) if won else None,
                'endings': dict(collections.Counter(r['ending'] for r in rows))}
    out = {'pass': which, 'all': stats(records)}
    for situation in ('clear', 'sworded', 'waited'):
        out[situation] = stats([r for r in records if r['situation'] == situation])
    out['harvested'] = stats([r for r in records if r['harvested']])
    out['entry_states'] = stats([r for r in records if not r['harvested']])
    taught = [r for r in records if 'teacher' in r]
    if taught:                                              # the demonstration builder's visits (`demos`)
        found = [r for r in taught if r['teacher'] == 'search']
        out['builder'] = {
            'visits': len(taught), 'search_found': len(found),
            'clean': sum(r['success'] and r['hits'] == 0 for r in taught),
            'by_weight': dict(collections.Counter(str(r['weight']) for r in taught)),
            'by_bombs': {b: stats([r for r in taught if r['bombs'] == b]) for b in (4, 8)},
            'by_hearts': {k: stats([r for r in taught if r['full'] == full]) for k, full in (('full', True),
                                                                                         ('partial', False))},
            'actions_mean_found': sum(r['actions'] for r in found) / len(found) if found else None,
            'actions_max_found': max((r['actions'] for r in found), default=None),
            'waits_mean_found': sum(r['waits'] for r in found) / len(found) if found else None,
            'swings_mean_found': sum(r['swings'] for r in found) / len(found) if found else None,
            'seconds_mean': sum(r['seconds'] for r in taught) / len(taught),
            'seconds_max': max(r['seconds'] for r in taught),
            'tried_mean': sum(r['tried'] for r in taught) / len(taught)}
    out['hit_runs'] = [r for r in records if r['hits'] or not r['success']][:40]
    return out


def _taught(task) -> tuple[list[dict], list]:
    """One worker's share of `demos`: (which, [(seed, start, bombs)], depth) -> the records and the kept rows."""
    from triforce.demos import DemoRecorder
    which, visits, depth = task
    envs = {4: make_env(4), 8: make_env(8)}
    probes = {4: make_env(4), 8: make_env(8)}
    recorder = DemoRecorder()
    out = []
    for seed, start, bombs in visits:
        record = teach(envs[bombs], which, seed, start, recorder, probes[bombs], depth)
        record.update(harvested=start.endswith('.state'), bombs=bombs, look=depth, which=which)
        out.append(record)
    return out, recorder.rows


def demos(out: Path, which: str, seed0: int, runs: int, depth: int = 4, progress=None, log: Path | None = None,
          workers: int = 1, skip: frozenset = frozenset()) -> dict:
    """The demonstration builder: every visit of one pass from `starts` by `teach` (the search, else the rule timed
    by a `depth`-decision look-ahead), the visits that leave by the exit unhit written to `out` as rows (`triforce.
    demos`: a MOVE or WAIT without its duration, a strike `REPEAT` times); `log`: each visit's record as a JSON line;
    `workers` processes share the visits one at a time; `skip`: seeds left out (a set of visits recorded again, as
    the first build's: round40.md "The generation's tasks").  The file is written as the visits end (at most every
    `SAVE_EVERY` seconds, and at the end, through a temporary file), so a job stopped at its time limit keeps what
    it found."""
    from triforce.demos import save_demos

    def save():
        if rows:
            part = out.with_name(out.name + '.part')
            save_demos(part, [obs for obs, _ in rows], [action for _, action in rows])
            part.replace(out)

    visits = [visit for visit in starts(which, seed0, runs) if visit[0] not in skip]
    tasks = [(which, [visit], depth) for visit in visits]
    records, rows = [], []
    saved = time.time()
    with multiprocessing.get_context('spawn').Pool(workers) as pool:
        for part, kept in pool.imap_unordered(_taught, tasks):
            records += part
            rows += kept
            if log is not None:
                with open(log, 'a', encoding='utf-8') as file:
                    file.write(''.join(json.dumps(record) + '\n' for record in part))
            if time.time() - saved > SAVE_EVERY:
                save()
                saved = time.time()
            if progress:
                progress(len(records), len(visits))
    save()
    result = summary(which, records)
    return {'file': str(out), 'rows': len(rows), 'played': len(records),
            'kept': sum(r['success'] and r['hits'] == 0 for r in records), 'builder': result['builder']}


def trace(which: str, seed: int, start: str | None = None):
    env = make_env(4 + 4 * (seed % 2))
    start = start or f'{STATES[which]}_{seed % 3}'
    obs, _ = env.reset(seed=seed, options={'start_state': start})
    rule = Line(which, Floor(env.state.render_grid()))
    health = env.state.health
    for i in range(400):
        ram = bytes(env.state.ram)
        step = rule.decide(ram, mask_ok(obs))
        nuts = ' '.join(f"{d.x:02x},{d.y:02x}{d.dir or '-'}{'*' if d.flash else ''}{'c' if d.cloud else ''}"
                        for d in sorted(darknuts(ram), key=lambda d: abs(d.x - ram[OBJ_X]) + abs(d.y - ram[OBJ_Y]))[:3])
        obs, _, terminated, truncated, info = env.step(step)
        lost = '  HIT' if env.state.health < health else ''
        health = env.state.health
        print(f"{i:3d} link {ram[OBJ_X]:02x},{ram[OBJ_Y]:02x} {step:12s} near {nuts}{lost}")
        if terminated or truncated:
            print(info.get('ending'))
            break


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('measure')
    m.add_argument('--pass', dest='which', choices=('in', 'out'), required=True)
    m.add_argument('--runs', type=int, default=1000)
    m.add_argument('--seed0', type=int, default=0)
    m.add_argument('--json', default=None, help="append each visit's record to this JSON-lines file")
    m.add_argument('--workers', type=int, default=1, help="processes sharing the visits")
    m.add_argument('--look', type=int, default=0, metavar='DEPTH',
                   help="training-time look-ahead: each decision the first of the rule's ranked actions that a second "
                        "env, playing it and then the rule DEPTH decisions, finds unhit (0: the rule alone)")
    d = sub.add_parser('demos', help="the demonstration builder (search, else the rule timed by the look-ahead)")
    d.add_argument('out')
    d.add_argument('--pass', dest='which', choices=('in', 'out'), required=True)
    d.add_argument('--runs', type=int, default=100)
    d.add_argument('--seed0', type=int, default=100000)
    d.add_argument('--look', type=int, default=4, metavar='DEPTH', help="the rule's look-ahead where no way is found")
    d.add_argument('--json', default=None, help="append each visit's record to this JSON-lines file")
    d.add_argument('--workers', type=int, default=1, help="processes sharing the visits")
    d.add_argument('--skip', type=int, nargs='+', default=[], metavar='SEED', help="seeds left out")
    u = sub.add_parser('summarize', help="the summary of measure's or demos' JSON-lines files")
    u.add_argument('files', nargs='+')
    g = sub.add_parser('merge', help="one demonstration file of several (`demos` run in pieces), rows in order")
    g.add_argument('out')
    g.add_argument('files', nargs='+')
    t = sub.add_parser('trace')
    t.add_argument('--pass', dest='which', choices=('in', 'out'), required=True)
    t.add_argument('seed', type=int)
    t.add_argument('--start', default=None)
    args = parser.parse_args(argv)
    if args.cmd == 'trace':
        trace(args.which, args.seed, args.start)
        return 0
    if args.cmd == 'summarize':
        records = [json.loads(line) for f in args.files for line in Path(f).read_text(encoding='utf-8').splitlines()]
        print(json.dumps(summary(records[0]['which'], records), indent=1))
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
    if args.cmd == 'measure':
        result = measure(args.which, args.seed0, args.runs, args.look,
                         lambda done, total: jobs.progress(done, total, 'visits'), Path(args.json) if args.json else None,
                         args.workers)
        print(json.dumps({k: v for k, v in result.items() if k != 'hit_runs'}, indent=1))
        return 0
    result = demos(Path(args.out), args.which, args.seed0, args.runs, args.look,
                   lambda done, total: jobs.progress(done, total, 'visits'), Path(args.json) if args.json else None,
                   args.workers, frozenset(args.skip))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
