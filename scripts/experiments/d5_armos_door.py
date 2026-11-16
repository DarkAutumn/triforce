"""Dungeon 5's two ways in from overworld room 0x0b, timed by buttons: the open cave mouth at X $70 and the stairs
under the Armos at ($B0, $80) (zelda-asm/levels/level-5.md "At a glance"; asm:Armos_SecretRoomIds).

    .venv/bin/python scripts/experiments/d5_armos_door.py [SEEDS]

Each run starts on the first playable frame of 0x0b entered from the Lost Hills (state `0_0bs_0`, the game's random
numbers drawn per seed), goes into level 5, walks straight back out of 5:0x76's S door and ends on the first
playable frame on the overworld.  The room's Leever is removed every frame, so only the statue matters.  Printed
per way: runs that got in, total frames, hearts lost.

Ways: `mouth`; `below` (up into the statue from the row under it, Up held until the stairs take Link); `west N`
(one frame of Right from ($A0, $7D), stand still, A pressed N frames before the statue has woken, then Right onto
the stairs; `west none`: no stab)."""

import statistics
import subprocess
import sys
from pathlib import Path

from triforce._native import Game

ROOT = Path(subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True,
                           cwd=Path(__file__).parent, check=True).stdout.strip())
MODE, LEVEL, ROOM = 0x12, 0x10, 0xEB                # asm:GameMode, asm:CurLevel, asm:RoomId
OBJ_TYPE, OBJ_X, OBJ_Y = 0x34F, 0x70, 0x84          # asm:ObjType, asm:ObjX, asm:ObjY (slot 0 is Link)
PLAY_AREA_TILES = 0x6530                            # asm:PlayAreaTiles, by column, 22 rows of 8 pixels
PLAY, ARMOS, STAIRS, WAKE_FRAMES = 5, 0x1E, 0x70, 63
STATUE = (0xB0 // 8, (0x80 - 0x40) // 8)            # the secret statue's top left tile (column, row)


class Run:
    def __init__(self, seed: int):
        self.game = Game(seed=seed, render=False, states_dir=str(ROOT / 'triforce' / 'states'))
        self.game.load('0_0bs_0', randomize_rng=True, seed=seed)
        assert (self.r(LEVEL), self.r(ROOM), self.r(MODE)) == (0, 0x0B, PLAY)

    def r(self, addr: int) -> int:
        return self.game.read(addr)

    @property
    def playing(self) -> bool:
        return self.r(MODE) == PLAY

    def woken(self) -> bool:
        return self.r(PLAY_AREA_TILES + STATUE[0] * 22 + STATUE[1]) == STAIRS

    def armos(self) -> bool:
        return any(self.r(OBJ_TYPE + slot) == ARMOS for slot in range(1, 0x0C))

    def frame(self, buttons: str):
        for slot in range(1, 0x0C):                 # every monster but an Armos is removed
            if self.r(OBJ_TYPE + slot) not in (0, ARMOS):
                self.game.write(OBJ_TYPE + slot, 0)
        self.game.press(buttons, 1)

    def until(self, done, buttons: str, limit: int = 3000):
        for _ in range(limit):
            if done():
                return
            self.frame(buttons)
        raise RuntimeError(f"not reached at ({self.r(OBJ_X):#04x}, {self.r(OBJ_Y):#04x})")

    def go(self, **target):
        (axis, value), = target.items()
        addr = OBJ_X if axis == 'x' else OBJ_Y
        low, high = ('LEFT', 'RIGHT') if axis == 'x' else ('UP', 'DOWN')
        self.until(lambda: self.r(addr) == value or not self.playing,
                   high if value > self.r(addr) else low, limit=600)
        assert self.playing and self.r(addr) == value


def play(way: str, stab, seed: int) -> dict:
    run = Run(seed)
    first, hearts = run.game.frame, run.game.state.health
    run.go(x=0x70)
    if way == 'mouth':
        run.until(lambda: not run.playing, 'UP')
    elif way == 'below':
        run.go(y=0x8D)
        run.go(x=0xB0)
        run.until(lambda: not run.playing, 'UP')
    else:
        run.go(y=0x8D)
        run.go(x=0xA0)
        run.go(y=0x7D)
        run.frame('RIGHT')
        assert run.armos(), 'the statue did not wake'
        touched = run.game.frame
        while not run.woken():
            early = stab is not None and run.game.frame - touched == WAKE_FRAMES - stab
            run.frame('A' if early else 'NONE')
        run.until(lambda: not run.playing, 'RIGHT', limit=400)
    run.until(lambda: run.r(LEVEL) == 5 and run.playing, 'NONE')
    lost = hearts - run.game.state.health
    run.until(lambda: run.r(LEVEL) == 0, 'DOWN')
    run.until(lambda: run.playing, 'NONE')
    return {'frames': run.game.frame - first, 'lost': lost}


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    ways = [('mouth', None), ('below', None), ('west', None)] + [('west', n) for n in (20, 16, 15, 14, 13, 12, 8, 4)]
    for way, stab in ways:
        rows = []
        for seed in range(seeds):
            try:
                rows.append(play(way, stab, seed))
            except RuntimeError:
                pass
        name = way if way != 'west' else f"west {'none' if stab is None else stab}"
        if not rows:
            print(f"{name}: 0/{seeds} in")
            continue
        frames = [row['frames'] for row in rows]
        print(f"{name}: {len(rows)}/{seeds} in, frames {statistics.median(frames):.0f} ({min(frames)}-{max(frames)}), "
              f"hearts lost {statistics.mean(row['lost'] for row in rows):.2f}")


if __name__ == '__main__':
    main()
