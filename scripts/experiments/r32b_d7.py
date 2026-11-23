"""Round 32's walks of Dungeon 7 with the wood sword (`d3first`: 11 containers on entry, the same as d2first's, which
brings the white sword; docs/routes/d3first-walks-b.md §5): the fights that are the dungeon's dear rooms, each played
by the scripted fighter that exists (round 28's `r28_darknut.Fighter`: stab what stands in the blade's reach, or in
the beam's line at full hearts; step aside from what is predicted to touch Link; else to the nearest target's flank)
with the wood sword against the white.

    python scripts/experiments/r32b_d7.py [--seeds N]

Rooms (`r26_walks.d7_state`: the probe with the route's inventory, doors, keys and bombs before the room, the sword
written): 0x49 (3 Blue Goriyas, 3 blue Keese, 2 Bubbles), 0x1b (6 Blue Goriyas: the 4 bombs), 0x1c (the recorder, the
three small Digdoggers: one kill opens the room as d2first plays it, and all three), Aquamentus 0x2a.  Each from full
11 hearts and from 7.

It plays the game over many seeded trials: a job of the machine's queue (``jobs.require_job``: ``triforce job run
... --timeout 30m -- .venv/bin/python scripts/experiments/r32b_d7.py ...``)."""

import argparse
import collections
import statistics
import tempfile
from pathlib import Path

from triforce.jobs import require_job

from r26_walks import AQUAMENTUS, DIGDOGGER_SMALL, GORIYA_BLUE, GORIYA_RED, KEESE_BLUE, d7_state
from r27_walks import load, summary
from r28_darknut import Fighter

RECORDER_SLOT = 5
TMP = Path(tempfile.mkdtemp(prefix='r32b-d7-'))


class Fight(Fighter):
    SHIELDED = ()

    def __init__(self, w, targets, **kwargs):
        super().__init__(w, **kwargs)
        self.TARGETS = targets


ROOMS = [  # (label, room, entry, targets, prelude, done)
    ('0x49 S (3 Blue Goriyas, Keese)', 0x49, 'S', (GORIYA_BLUE, GORIYA_RED, KEESE_BLUE), None, None),
    ('0x1b W (6 Blue Goriyas)', 0x1B, 'W', (GORIYA_BLUE,), None, None),
    ('0x1c W (Digdogger: one small one)', 0x1C, 'W', (DIGDOGGER_SMALL,), 'recorder', 1),
    ('0x1c W (Digdogger: all three)', 0x1C, 'W', (DIGDOGGER_SMALL,), 'recorder', None),
    ('0x2a W (Aquamentus)', 0x2A, 'W', (AQUAMENTUS,), None, None),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--seeds', type=int, default=20)
    args = parser.parse_args()
    require_job()
    for label, room, entry, targets, prelude, kills in ROOMS:
        for sword, sname in ((1, 'wood'), (2, 'white')):
            for hearts in (11.0, 7.0):
                path = TMP / f'7_{room:02x}{entry.lower()}_{sword}_{hearts}.state'
                d7_state(path, room, entry, hearts=hearts, extra_inventory={'sword': sword})
                rows, ends, swings, frames = [], collections.Counter(), [], []
                for seed in range(args.seeds):
                    w = load(path, seed)
                    w.wait_play()
                    extra = 0
                    if prelude == 'recorder':
                        w.select_b(RECORDER_SLOT)
                        w.press('RIGHT', 6)
                        w.press('B', 1)
                        extra = 7 + w.until(lambda w: any(o[1] == DIGDOGGER_SMALL for o in w.objects()), 'NONE', 400)
                    f = Fight(w, targets)
                    start = len([o for o in w.objects() if o[1] in targets])
                    if kills:
                        end = 'time'
                        for _ in range(6000):
                            if w.hearts() <= 0:
                                end = 'dead'
                                break
                            alive = [o for o in w.objects() if o[1] in targets and o[5] < 0x10]
                            if start - len(alive) >= kills:
                                end = 'clear'
                                break
                            f.frame()
                    else:
                        end = f.fight(9000)
                    rows.append((end == 'clear', f.t.lost(), f.t.frames + extra, f.t.by))
                    ends[end] += 1
                    swings.append(f.swings)
                print(f'7:{label}, {sname} sword from {hearts} of 11: {summary(rows)}; ended {dict(ends)}; swings mean '
                      f'{statistics.mean(swings):.1f}')


if __name__ == '__main__':
    main()
