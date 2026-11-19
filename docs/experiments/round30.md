# Round 30: the optimisation round over the whole route

## The golden path to the ending, the planner's complete run, the corpus (no training)

The owner's list: "a complete run from power on to Zelda, models alone and with the honest planner; the golden
path extended to the end; a replay in the corpus". Before this, the golden path had last been run to the D3
triforce (`corpus/replays/v0-golden.tfreplay`), the models alone ended the game in 0 of 1000 single runs from the
game start (`r29land-1k-full`) and the planner's ending (`corpus/replays/v0-plan-ending.tfreplay`) started from the
game start state, not from power on. The team is main's at 1.9.0.0 (`r29-land`, 34 families on the route); every
promoted model acts by `temperature:0.5+relax:16` (its `act` file or the default), so `--act model` is the play
default and every action is drawn: no random numbers are redrawn at a checkpoint.

### The tool

Nothing stopped the race at an old end. `golden.checkpoints('d2first')` is derived from the route (a step whose
exit is `triforce` and that is not the last): `[24, 68, 105, 134, 209, 248, 281, 297]`, outside each of the eight
dungeons, and the ninth segment plays Dungeon 9 to the route's last step (exit `end`, the ending);
`tests/test_golden.py` already asserted the eight. No segment list, step or frame budget needed changing (a run's
budget is `DEFAULT_MAX_STEPS` 50,000 steps per segment run). Only the documentation was stale ("[24, 68]", "to the
route's last triforce"): `golden.py`, `zelda-headless.py` and `docs/architecture.md` now say what the tool does.
`play-zelda.py`'s and `play_grid.py`'s docstrings still say "to the route's last triforce" (left as they are).

### Models alone, from console power on (the golden race)

```
TRIFORCE_MODELS_DIR=$PWD/models nice -n 10 .venv/bin/python zelda-headless.py OUT 1 --golden --power-on \
    --clips on --act model --parallel 8 --first-seed 0
tf replay verify OUT/golden-00000000-win.tfreplay
tf replay info OUT/golden-00000000-win.tfreplay
```

The race reaches the ending. The first path (seeds from 0): 158,300 frames (44 minutes of game at 60.1 fps), 28 s
of wall time on 8 games (load average about 7), `tf replay verify`: ok, 158,300 frames, 14 edits, 9 states.
`tf replay info`: source `power-on LINK 18001` (the idle delay drawn from seed 1, the first segment's winner; the
models take over at frame 18,665, 0 edits before it), act `temperature:0.5+relax:16 (every model, 34)`, 9
segments, states 1 END + 8 CHECKPOINT, no cheated item, ending `win` ("the game ended in (level 9, room 0x32)").
The file is 143,213 bytes: the nine states are 102,506 of them (about 11.4 KB each), the inputs 25,905.

Runs finished when the segment's winner came in (8 games at once; the games still playing are dropped), for that
path and five more (`OUT 5 --golden ... --first-seed 100000`; every one ended the game):

| segment (stop) | from - to | seeds 0 | 100000 | 102880 | 105760 | 108640 | 111520 |
|---|---|---|---|---|---|---|---|
| 0 (step 24) | power on - D2 triforce | 1 | 2 | 1 | 1 | 1 | 2 |
| 1 (68) | - D1 triforce | 3 | 4 | 1 | 1 | 2 | 3 |
| 2 (105) | - D3 triforce | 3 | 2 | 2 | 2 | 1 | 2 |
| 3 (134) | - D4 triforce | 1 | 1 | 1 | 1 | 2 | 2 |
| 4 (209) | - D5 triforce | 5 | 10 | 6 | 4 | 4 | 2 |
| 5 (248) | - D7 triforce | 6 | 6 | 9 | 3 | 6 | 2 |
| 6 (281) | - D6 triforce | 7 | 7 | 7 | 5 | 8 | 5 |
| 7 (297) | - D8 triforce | 17 | 11 | 7 | 16 | 6 | 4 |
| 8 (end) | - the ending (Dungeon 9, Ganon) | 12 | 25 | 26 | 20 | 16 | 5 |
| all | | 55 | 68 | 60 | 53 | 46 | 27 |
| frames | | 158,300 | 157,949 | 150,995 | 151,932 | 158,760 | 174,937 |
| wall (s) | | 28 | 66 | 67 | 63 | 58 | 47 |

The five later paths ran while the machine's load average rose from 5 to 65 (other agents' chains), hence their
wall times. The last two segments take the most runs (4-17 and 5-26): the segment chains from full hearts (`d8`
24.8%, `d9` 31.0%) would give a winner every 3-4 runs; the race enters them with the hearts its earlier winners
kept. The default 320 attempts a segment are far from binding.

### The honest planner, one complete run from power on

```
TRIFORCE_MODELS_DIR=$PWD/models nice -n 10 .venv/bin/python zelda-headless.py OUT 1 --wins-only --max-runs 40 \
    --power-on --clips on --act temperature:0.5+relax:16+plan:16:1 --parallel 8 --first-seed 0
```

A plain run does it (no race): of the first 8 runs to finish (8 games from seed 0) 7 died and seed 3 ended the
game, 5 min 40 s of wall time (load average 30-35). `00000003-win.tfreplay`: 53,548 bytes, 149,596 frames, `tf
replay verify` ok (30 edits, 1 state), source `power-on LINK 3550` (control from frame 4,214), act
`temperature:0.5+relax:16+plan:16:1 (every model, 34)`, 8 cheated bombs (no key, no rupee), "the game ended in
(level 9, room 0x32)".

### The corpus: proposed, not committed

The models' golden replay from power on to the ending (seeds from 0, above) is the proposal for
`corpus/replays/v1-golden-ending.tfreplay`, but at 143,213 bytes it is 2.2 times the largest corpus file (49-65
KB): its nine states are 102,506 bytes of it (about 11.4 KB each; eight are the stitch's CHECKPOINT states), and
every golden path of the whole route is 143-145 KB. It is not committed; the owner keeps or drops it. Placed in `corpus/replays/`
(untracked), `cargo test --release -p triforce-tools --test cli the_replay_corpus_still_verifies` passed with it
(81 s). A corpus-sized golden path would need the stitch to keep fewer states (`crates/replay`, not changed in
this round). The planner's single run from power on (53,548 bytes, one state) is in line with the corpus sizes,
should the owner want a complete game from power on there instead.

### Watching the new route

In the worktree, offscreen (`QT_QPA_PLATFORM=offscreen`, `--no-sound`, `--parallel 2`, `--power-on`), with a
throwaway wrapper printing the title and the cells' strips every 10 s:

- `play-zelda.py 2x1 --show golden`, 150 s: the title counts the legs "racing leg N of 9"; the first path was
  made in about 65 s on 2 games and its cell played it at 60 fps from the models' take-over (`#0 L0 0x77
  ow-start`, on to 0x58, 0x5b, 0x5d, 0x4c and Dungeon 2 by 140 s); the second was made by 140 s (`#2880`). No error.
- `play-zelda.py 2x1 --live`, 100 s: the live cell played seed 1,000,000 from power on at 60 fps through the first
  overworld rooms into Dungeon 2 (0x7d, 0x7e, 0x5e by 90 s); the simulations filed 13 runs meanwhile (all deaths:
  best, died and random fill, `best` hands out once half full). No error.
- The golden file itself (`play-zelda.py FILE --fps 0`, 120 s): uncapped at 1,400-3,400 fps it played the whole
  replay through Dungeons 3, 4, 5, 7, 9 to the end, then looped. No error.

The only message on stderr was Qt's offscreen "This plugin does not support propagateSizeHints()". Nothing in the
player was changed.
