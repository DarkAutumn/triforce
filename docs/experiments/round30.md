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

### The corpus: `v1-golden-ending.tfreplay`

The models' golden replay from power on to the ending (seeds from 0, above) is in the corpus as
`corpus/replays/v1-golden-ending.tfreplay` (the lead's decision, Nov 19: the first complete game by the models
alone belongs there). At 143,213 bytes it is 2.2 times the largest of the other corpus files (49-65 KB): its nine
states are 102,506 bytes of it (about 11.4 KB each; eight are the stitch's CHECKPOINT states), and every golden path
of the whole route is 143-145 KB. `tf replay verify` in the main checkout: ok, 158300 frames, 14 edits, 9 states;
`cargo test --release -p triforce-tools --test cli the_replay_corpus_still_verifies` passed with it (81 s). A
smaller golden path would need the stitch to keep fewer states (`crates/replay`, not changed in this round). The
planner's single run from power on (53,548 bytes, one state) is not in the corpus: the planner's complete game from
the start state already is (`v0-plan-ending.tfreplay`).

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

## Wizzrobe and patra (Dungeon 6 and level 9)

**State of this section.** Complete: the cause, the arm, the selections and the decision. Promoted in this branch's
checkout: wizzrobe `r30w-wj-s89` and patra `r30w-patra`. Left for the landing: the chains of record with both.

wizzrobe plays Dungeon 6's 0x78, 0x38, 0x28, 0x29, 0x3a, 0x2c and level 9's 0x65, 0x15, 0x17, 0x07, 0x05, 0x20,
0x10, 0x31, 0x30, 0x04; patra level 9's 0x16, 0x27, 0x61, 0x52. The route's runs are `r79c-wj-s85` (round29.md
§18.1: the one joint run of three seeds that held `d6`, 314 against 316 of 1000, while lifting `d9`) and
`r79c-patra`.

### The cause

From the chains of record (`r29land-1k`, 1000 runs a segment, seeds 0-999) and the planner beside
(`r29land-plan-full`, 300 runs from the game start; the Dungeon 9 lane's `r79c-plan-d9`, 300 from Dungeon 9's
entrance). Hearts lost a visit are counted on the visits Link survives.

**Dungeon 6** (`d6`, from 12 hearts):

| room | models: entered, hearts in, lost a visit, deaths | planner (full route): entered, hearts in, lost, deaths |
|---|---|---|
| 0x38 | 1000, 10.83, 4.47, 137 | 254, 10.70, 2.56, 20 |
| 0x28 | 863, 6.40, 0.29, 24 | 234, 8.29, 1.01, 9 |
| 0x29 | 839, 6.23, 0.54, 27 | 237, 7.39, 1.02, 25 |
| 0x3a | 801, 6.16, 2.76, **377** | 204, 6.93, 1.81, 51 |
| 0x2c | 399, 4.77, 1.03, 56 | 141, 6.51, 0.95, 12 |

0x38 costs the models about two hearts a visit more than the planner, so 0x3a is entered with about six. There
they die at every heart count: 148 of 177 entering below 4 hearts, **174 of 374 at 4 to 7.9** (the planner 15 of
71), 55 of 250 at 8 or more (the planner 8 of 90). On the full route (90 runs reach Dungeon 6 with 10.79 of 12
hearts) 0x38 costs 4.16 a visit and 0x3a kills 33 of 67.

**Level 9** (`d9`, from 15 hearts; 690 of 1000 runs lost):

| room | entered | hearts in | lost a visit | deaths | below 4 hearts in: deaths | 4 to 7.9 in: lost, models / planner |
|---|---:|---:|---:|---:|---|---|
| 0x17 to its wall | 998 | 10.85 | 2.42 | 33 | 8 of 12 | 1.71 / 1.00 |
| 0x16, patra's fight | 939 | 7.68 | 0.92 | 20 | 19 of 114 | 0.92 / 0.25 |
| 0x05 | 922 | 6.91 | 1.06 | 69 | 66 of 148 | 1.08 / −0.08 |
| 0x10, fought | 815 | 5.93 | 2.16 | **210** | 156 of 218 | 2.15 / 1.80 |
| 0x30, fought | 537 | 4.20 | 1.21 | **115** | 115 of 271 | 1.47 / 0.48 |
| 0x52 (patra) | 399 | 3.52 | 0.44 | 40 | 40 of 246 | 0.69 / 0.13 |

Level 9's deaths are the runs that enter its fights low: 0x30's 115 deaths are all below 4 hearts, 0x05's 66 of 69,
0x10's 156 of 210. The planner, at the same hearts, loses a tenth (0x05) to a third (0x30) of the models' hearts.

**The draws.** wizzrobe's 60 seed states are 18 of Dungeon 6 and 42 of level 9; its demonstration rows (pooled,
sampled uniformly) are 84 thousand of Dungeon 6 (`bomb28`, `plan38-3a`) among 512 thousand, 300 thousand of them
level 9's scripted rules (`wizzrobe-05`, `-10`, `-30`), which lose more hearts a visit than the model. No
curriculum track is unfinished (106 of 106 complete; patra 24 of 24).

**At the route's hearts.** The segments start each dungeon at full hearts; the route does not. Played by the landed
team from the state after Dungeon 7 (`0_42c_0`, 600 runs, seeds 5000-5599) Dungeon 6 is entered with 10.85 of 12
hearts and its triforce taken in **15.7%** of runs (`d6` from 12 hearts: 31.4%): 0x38 kills 136 of 595 (4.53 hearts
a visit) and 0x3a 237 of 401 entered with 4.91 hearts (123 of 143 below 4, 99 of 196 at 4 to 7.9). From Dungeon 8's
triforce (`0_6dc_0`, 1000 runs, seeds 5000-5999; Dungeon 9 entered with 12.29) 127 end the game: 0x10 kills 221
(entered with 4.64), 0x05 130 (5.07), 0x17 on the way to its wall 102, 0x30 86 (3.54), patra's 0x16 fight 67, 0x52
29 (3.25).

### The arm

The recipe of decisions 2026-11-19 ("One team for the whole route"), at the route's hearts:

- **Hearts drawn as the route brings them.** `triforce/chain_hp.json` refitted from the two chains above (`python
  -m triforce.chain_hp h6.jsonl h9.jsonl`): Dungeon 6's rooms are profiled now (0x38 9.64, 0x3a 4.91, 0x28 5.38,
  0x29 5.19, 0x2c 4.05), level 9's barely move (0x05 5.28 → 5.07, 0x10 4.37 → 4.30, 0x30 3.56 → 3.54, 0x52 3.07 →
  3.25). wizzrobe and patra keep `chain_hp: 0.5`: half the room-entry resets into a profiled room draw from it.
- **Handoffs at the route's hearts as starts** (`harvested_fraction`: wizzrobe 0.33, patra 0.25; the states in
  `triforce/states/harvested/<family>`, not in git). The two chains ran with `--harvest DIR --replays DIR`; 6:0x28,
  0x29 and 9:0x10, 0x30 follow a wizzrobe room (no handoff) and are cut from the replays
  (`scripts/experiments/r29_zora_states.py REPLAYS OUT --step 271:6_28s --step 272:6_29w`, `--step 340:9_10s --step
  350:9_30e`). wizzrobe: 6:0x38 450, 0x3a 401, 0x2c 135, 0x28 100, 0x29 100, 9:0x17 150, 0x05 250, 0x10 250, 0x30
  200 (Dungeon 6 about 60% of them: with the seed states' 18 of 60, about 40% of the resets go to Dungeon 6 against
  30% before); patra: 0x16's fight 400, 0x52 174, 0x27 from W 150. Each set is the first N of its stem by seed
  (`ls | sort -t_ -k3.2n | head -N`).
- **The planner as a teacher at the route's hearts** (`scripts/experiments/r27_plan_demos.py`, the planner
  `temperature:0.5+relax:16+plan:16:1` playing the route's run from the harvested states themselves, two episodes a
  state, with a copy of the family's yaml whose `ram_randomization` is `{}` so that a state keeps the hearts and
  bombs the chain brought: an evaluation pool has no harvested dir and would draw them again).

  | File (run played) | State: won of episodes, hearts lost in the wins (the models a visit at the route's hearts) | Rows |
  |---|---|---:|
  | `plan30-d6-05` (`r79c-wj-s85`) | 6:0x38 1655 of 1800, 2.48 (4.53); 0x3a 975 of 1604, 1.23 (164 of 401 survive); 9:0x05 454 of 500, 0.23 (1.11) | 357,531 |
  | `plan30-10-30` (`r79c-wj-s85`) | 9:0x10 356 of 500, 1.25 (2.02, 62% survive); 0x30 360 of 472, 0.31 (1.12) | 100,973 |
  | `plan30-patra` (`r79c-patra`) | 0x16's fight 775 of 800, 0.16 (0.76); 0x27 from W 300 of 300, 2.01 (2.42); 0x52 332 of 348, 0.09 (0.46) | 100,107 |

  Dungeon 6's states were listed twice (four episodes a state), 0x30's twice: the rows of Dungeon 6 are about 40% of
  wizzrobe's demonstration rows (16% before: the level 9 rules' 300 thousand rows dominated them).
- **Three seeds and one** from the route's runs, 488 iterations each, `--stop-success 1.01 --sil-coeff 0.1
  --checkpoint-every 30`, clips `{enabled: true, off_share: 0}` (both yamls already), `--demo-coeff 0.05` over the
  lineage's files and the new ones: `r30w-wj-s87`, `-s88`, `-s89` from `r79c-wj-s85` (`bomb28`, `plan38-3a`,
  `wizzrobe-05`, `-10`, `-30`, `clip-07`, `clip-20n`, `plan9-wizz`, `plan30-d6-05`, `plan30-10-30`); `r30w-patra`
  (seed 90) from `r79c-patra` (`patra-16`, `-61`, `-52`, `plan9-patra`, `plan30-patra`). No curriculum track is
  unfinished in either family, so none is taken out.

### Single rooms on equal terms

Every run evaluated under this branch's yaml and `chain_hp.json` (100 episodes a state, seed 0, clips on, its act
mode; the route's runs in `eval-batch r30w-base`). Success and hearts a visit:

| Room | `r79c-wj-s85` | `r30w-wj-s88` (last) |
|---|---|---|
| 6:0x38 | 80.3%, −4.44 | 77.3%, −4.90 |
| 6:0x3a | 60.7%, −3.56 | 60.0%, −3.40 |
| 6:0x28, 0x29, 0x2c | 99.0, 98.0, 91.3% | 97.7, 96.7, 93.0% |
| 9:0x05 | 88.3%, −1.29 | 88.7%, −1.14 |
| 9:0x10 | 88.8%, −1.05 | 90.2%, −0.86 |
| 9:0x30 | 85.7%, −0.97 | 86.3%, −0.98 |
| 9:0x17 | 94.8%, −1.65 | 93.2%, −1.84 |
| all 6000 episodes | 93.1%, −1.02 | 93.0%, −1.02 |

| Room | `r79c-patra` | `r30w-patra` (last) |
|---|---|---|
| 9:0x52 | 91.0%, −0.54 | 93.7%, −0.28 |
| 9:0x16 | 98.5%, −0.72 | 99.2%, −0.54 |
| 9:0x27 | 97.7%, −1.09 | 97.3%, −1.04 |
| 9:0x61 | 99.0%, −0.12 | 98.5%, −0.11 |

With Dungeon 6's profile 0x3a is played from 4.9 hearts in half the resets: `r79c-wj-s85` takes it in 60.7% (74.3%
under round 29's yaml). The chain decides (round 25): the selections on `d9` and `d6` (screen seeds 2000-2999,
confirmation 3000-3999, `r79c-wj-s85` and `r79c-patra` pinned beside) are queued.

### patra chosen on `d9`

`select-checkpoint patra r30w-patra --segment d9 --runs 1000 --first-seed 2000 --confirm 1000 --confirm-first-seed
3000 --confirm-top 1 --only it001884,it001914,it001944,last,promoted` (the route's wizzrobe `r79c-wj-s85` in the
team; `r79c-patra` pinned beside):

| candidate | screen (2000-2999) | confirm (3000-3999) | hearts at the end | confirm against `r79c-patra` |
|---|---:|---:|---:|---:|
| **it001944** | 36.1% | **36.4%** | 3.54 | **+6.2 ± 2.0** |
| it001884 | 35.5% | - | - | - |
| last (it001952) | 35.1% | 33.1% | 3.68 | +2.9 ± 2.0 |
| it001914 | 33.7% | - | - | - |
| `r79c-patra` | 28.7% | 30.2% | 3.16 | |

**it001944 is better on `d9` beyond noise on seeds not used to choose it** (+6.2 points, three standard errors);
patra plays no other segment (`post-d8` ends on entering 9:0x76). Applied as the run's `ckpt.pt` and evaluated on
equal terms (100 episodes a state, this branch's yaml): 97.9% of 2100 episodes, −0.53 hearts (`r79c-patra` 97.3%,
−0.63); 0x52 94.7%, −0.41 (91.0%, −0.54), 0x16 98.8%, −0.53 (98.5%, −0.72), 0x27 97.5%, −1.04 (97.7%, −1.09), 0x61
99.0%, −0.08 (99.0%, −0.12): every state within a point or better. **`r30w-patra` (it001944) qualifies as the
route's patra.** Its promotion waits for the wizzrobe selections, whose chains play patra's promoted run.

### wizzrobe chosen on `d9` and `d6`

Each seed: `select-checkpoint wizzrobe RUN --segment SEG --runs 1000 --first-seed 2000 --confirm 1000
--confirm-first-seed 3000 --confirm-top 1 --only it002844,it002874,it002904,last` for SEG `d9` and `d6`; `r79c-wj-s85`
pinned beside in s88's two selections, s87 and s89 `--no-promoted` and paired against those chains (the same
weights on the same seeds; the `d9` reference reproduces the patra selection's run for run, 302 of 1000). `d6` counts
the runs that take the D6 triforce. Confirmation seeds 3000-3999, paired against `r79c-wj-s85` (`d9` 302, `d6` 321
of 1000):

| run | `d9` chosen | `d9` confirm | vs `r79c-wj-s85` | `d6` chosen | `d6` confirm | vs `r79c-wj-s85` |
|---|---|---:|---:|---|---:|---:|
| `r30w-wj-s87` | last | 30.0% | −0.2 ± 2.0 | last | 30.9% | −1.2 ± 2.1 |
| `r30w-wj-s88` | last | 25.0% | **−5.2 ± 2.0** | it002874 | 38.8% | +6.7 ± 2.1 |
| **`r30w-wj-s89`** | **it002844** | **32.9%** | +2.7 ± 2.1 | **it002844** | **38.9%** | **+6.8 ± 2.2** |

Screens (2000-2999): s87 `d9` 27.2-30.5%, `d6` 28.1-31.8%; s88 `d9` 21.3-23.8% (`r79c-wj-s85` 28.7%), `d6`
28.6-37.0% (32.6%); s89 `d9` 32.2-34.0%, `d6` 32.1-38.3%.

**`r30w-wj-s89` at it002844 is the one run that qualifies:** Dungeon 6 better beyond noise (+6.8 points, three
standard errors, on seeds not used to choose it) and level 9 not worse (+2.7 ± 2.1), the same checkpoint chosen on
both segments. s88 gains Dungeon 6 as much but loses level 9 beyond noise (the third of three seeds again: one seed
of three holds both stretches); s87 changes neither. **Promoted in this branch's checkout:** patra `r30w-patra`
(it001944, after the last wizzrobe selection ended); wizzrobe `r30w-wj-s89` (it002844).

Its evaluation on equal terms (100 episodes a state, this branch's yaml): 93.2% of 6000 episodes, −0.98 hearts
(`r79c-wj-s85` 93.1%, −1.02). Within a point or better on 12 of 16 rooms (9:0x05 91.3% and −0.94 against 88.3%
and −1.29, 6:0x2c 94.3% against 91.3%); outside the bar on success: 6:0x38 78.3% against 80.3% (hearts lost
better, −4.31 against −4.44), 9:0x30 83.7% against 85.7% (hearts equal), 0x31 97.0% against 99.0%, 0x65 98.0%
against 99.3%. On the confirmation chains, room by room (`r79c-wj-s85` / `r30w-wj-s89`, deaths): 6:0x38 121 / 88
(5.22 / 4.55 hearts a visit), 0x3a 390 / 320, 9:0x05 70 / 35; worse 9:0x31 23 / 48, 0x04 14 / 26, 6:0x2c 50 / 88
(396 / 506 entering, 1.08 / 1.41 a visit). The segments decide (round 25: single rooms don't), so the run is
promoted; the three rooms that lose more are in the backlog.

| family | run (checkpoint) | replaces | segment, seeds 3000-3999, paired |
|---|---|---|---|
| wizzrobe | `r30w-wj-s89` (it002844) | `r79c-wj-s85` | `d6` +6.8 ± 2.2, `d9` +2.7 ± 2.1 |
| patra | `r30w-patra` (it001944) | `r79c-patra` | `d9` +6.2 ± 2.0 |

Not done: the two promotions were judged one at a time (each with the other family's old run); the chains of
record with both (`d6`, `d9`, `post-d7`, `post-d8`, the full route) are the landing's.

## Darknut, boss-manhandla, gibdo: Dungeon 5 at the route's hearts, then Dungeon 8

darknut plays 3:0x5b, 0x59, 0x69 and 0x5c in Dungeon 3, 5:0x64, 0x06, the Olive Garden 0x05 and 0x37 in Dungeon 5,
8:0x5e, 0x4e, 0x3e and 0x3f in Dungeon 8. Its line: `r75-d537-c5` (round 25, `main` until round 28) →
`r78t4-darknut` → `r78-dark37-s85` (round28.md §15: chosen on `d8`, checked on `d3` +1.4 ± 2.0 and `d5` +6.8 ± 2.0
from full hearts). Every round of the lineage trained half its episodes with CLIP masked; this is its first with
`clips: {enabled: true, off_share: 0}`.

### The cause

The full route (`r29land-1k-full`, 1000 runs, `r78-dark37-s85`) against `r77b-1k-full` (the same seeds, `main`'s
round 27 team with `r75-d537-c5`) and the honest planner beside (`r29land-plan-full`, 300 runs), darknut's visits by
route step: visits, hearts on entering, hearts lost a visit (the visits Link survives), deaths.

| visit | `r78-dark37-s85` | `r75-d537-c5` | the planner (300) |
|---|---|---|---|
| 5:0x05 from E (step 192) | 540, 7.06 in, **1.69**, **141 deaths** | 434, 7.42, 1.31, 80 | 284, 9.17, 0.73, 5 |
| 5:0x64 from the cellar (197) | 398, 5.44, 1.12, 40 | 354, 6.10, 1.15, 24 | 280, 8.39, 0.93, 4 |
| 5:0x37 (203) | 285, 4.05, 0.12, 4 | 264, 4.62, 0.16, 8 | 273, 7.35, 0.07, 1 |
| 3:0x59 from N (94) | 740, 4.89, 0.58, 4 | 641, 4.92, 0.70, 10 | 297, 5.71, −0.05, 1 |
| 3:0x69 from N (95) | 725, 4.32, 0.65, 12 | 631, 4.26, 0.71, 19 | 294, 5.78, 0.49, 0 |
| 3:0x5b from W (100) | 703, 3.48, −0.05, 15 | 603, 3.39, −0.14, 13 | 294, 5.25, −0.17, 0 |

The Olive Garden by the hearts Link brings (full route): 6 to 7.9 hearts in, 63 deaths of 245 visits against 30 of
168; 8 to 9.9, 2.61 hearts lost a visit against 1.65 (32 deaths of 186 against 24 of 187). From full hearts (the
`d5` segment, `r29land-1k-d5`) it loses 2.65 a visit with sword beams and 35 deaths of 691: the segment the run was
checked on starts where the route never arrives.

**Dungeon 5 at the route's hearts.** The leg after Dungeon 4 arrives at 5:0x76 with 7.77 hearts (the full route
7.74), and no darknut room is on it, so `--from 0_45c_0 --stop-at step:209` (the leg and Dungeon 5, to the overworld
after the D5 triforce) plays Dungeon 5 as the full route does. 500 runs of it (seeds 7000-7499, the harvest chain
below): **247 of 500 take the triforce (49.4%)**, the full route 262 of 540 entering (48.5%), the `d5` segment
72.2%; 0x05 entered with 7.27 hearts, 1.87 lost a visit, 116 deaths of 499.

**Dungeon 8** (`r29land-1k-d8`, 1000 runs from its entrance at 13 of 13 hearts; the full route enters with 12.50):
0x5e 1000 visits, 12.18 in, 4.04 lost, 20 deaths; 0x4e 979, 8.15, 2.91, 88; **0x3e 891, 5.68, 1.61, 414**; 0x3f
476, 5.49, 1.98, 155. darknut's four rooms end 677 of the 752 runs `d8` loses. The planner in the same rooms (full
route): 0x5e 0.97 hearts a visit, 0x4e 1.33 (3 deaths of 129), 0x3e −0.29 (none), 0x3f 0.10 (none).

**Dungeon 3**: darknut's rooms kill few runs (0x69 12, 0x5b 15 of 1000 full runs) but lose the hearts Manhandla
needs: 3:0x4d is entered with 3.58 hearts (the planner 5.42) and 81 of its 97 deaths are entries below 4 hearts.

All 73 curriculum tracks of `r78-dark37-s85` are complete: no track idles the resets (the recipe's fourth lever has
nothing to take out).

### The arm

From `r78-dark37-s85`, three seeds (`r30-dark-s87`, `-s88`, `-s89`), 488 iterations, `--sil-coeff 0.1
--checkpoint-every 30`, `clips: {enabled: true, off_share: 0}`, and:

- **the chain's entries as harvested starts** at `harvested_fraction: 0.33` (`triforce/states/harvested/darknut`, not
  in git; 1220 states). One batch harvested the handoffs (`eval-chain --harvest`) and recorded replays on seeds
  7000-7499 no measurement uses, with the promoted team: the leg after Dungeon 4 and Dungeon 5 (`--from 0_45c_0
  --stop-at step:209 --runs 500`), Dungeon 8 (`--from 8_7es_0 --stop-at step:297 --runs 500`) and the leg after
  Dungeon 1 with Dungeon 3 (`--from 0_37c_0 --stop-at step:134 --runs 300`; Dungeon 3 is entered as the full route
  enters it, 0x5b at 5.45 hearts against 5.42). The rooms entered from another darknut room have no handoff and were
  cut out of the replays (`scripts/experiments/r29_zora_states.py REPLAYS OUT --step 192:5_05e`, `--step 290:8_4es
  --step 291:8_3es --step 292:8_3fw`). Kept, the first by seed: 0x05 from E 250 (7.27 hearts on entering), 0x64
  from the cellar 120, 0x37 60; 8:0x5e 120, 0x4e 120, 0x3e 250, 0x3f 150; 3:0x59 from N 60, 0x5b from W 60, 0x59 from
  E 30. The harvested selector weighs them by their failures and hearts lost as the seed selector does;
- **the honest planner's play as demonstrations** beside the lineage's three files (`target/demos/r28-demos/darknut-5e.npz`,
  `target/demos/r28-demos/darknut-3e.npz`: Dungeon 8's scripted rules; `target/demos/r78dk/demos37.npz`: 0x37's rule;
  untracked, gathered in `main`'s checkout), all at `--demo-coeff 0.05`:
  `scripts/experiments/r27_plan_demos.py models/darknut/r78-dark37-s85 plan-dk.npz --family darknut --states 5_05e
  5_64c 8_5es 8_4es 8_3es 8_3fw --runs 40 --max-loss 2` (the family's hp and bombs draws, full rooms): 61,067 rows.
  The planner with `r78-dark37-s85` wins 0x05 from E 114 of 120 losing 0.24-0.61 hearts, 0x3e 111 of 120 at −0.31
  to 0.28, 0x5e 112 of 120 at −0.13 to 1.18, 0x3f 118 of 120, 0x4e 118 of 120, 0x64 from the cellar 115 of 120.

**The Olive Garden's damage by bombs and by the sword** (the baseline for a darknut that fights by bombs, as a route
with the wood sword will need). 5:0x05 from E, per visit, from the room records (`bomb_damage` counts in wood-sword
hits, `hits_dealt` every hit on a monster, `bomb_hits` the hits credited to bombs); the sword's share is taken as
two wood-sword hits of damage a white-sword hit (`[INFERENCE]` from the game's sword damage, not measured here):

| chain | visits | bombs carried in | bombs placed | bomb damage | sword hits | bombs' share of the damage | kills |
|---|---:|---:|---:|---:|---:|---:|---:|
| full route `r29land-1k-full` | 540 | 1.01 | 0.99 | 3.44 | 19.4 | 8% | 4.77 |
| leg and Dungeon 5 (harvest, 500) | 499 | 0.75 | 0.75 | 2.81 | 20.4 | 6% | 4.94 |
| `d5` from full hearts `r29land-1k-d5` | 1000 | 2.64 | 2.63 | 10.02 | 18.7 | 21% | 5.59 |
| the planner, full route (300) | 284 | 0.93 | 0.93 | 3.73 | 22.1 | 8% | 5.83 |

Every bomb Link carries in is placed (bombs free down to 0 in the room) and almost every one hits (0.92 of 0.99
placed in the full route), at 3.5-3.8 wood-sword hits of damage a bomb; the route brings about one bomb to the
room, so the sword deals some nine tenths of the damage. No bomb in the room is cheated.

### The runs (single rooms), and where the choice stands

Canonical evaluations on this code (clips on, 100 episodes a state, the run's act mode):

| run | success | hearts | 5:0x05 | 8:0x3e | 8:0x5e | 8:0x4e | 8:0x3f |
|---|---:|---:|---:|---:|---:|---:|---:|
| `r78-dark37-s85` (promoted) | 96.3% | −0.91 | 92.7% | 81.7% | 84.3% | 92.7% | 95.3% |
| `r30-dark-s87` | 96.6% | −0.86 | 92.5% | 85.7% | 85.7% | 92.0% | 94.3% |
| `r30-dark-s88` | 96.4% | −0.89 | 92.0% | 86.0% | 85.7% | 92.7% | 93.7% |
| `r30-dark-s89` | 96.8% | −0.87 | 93.0% | 87.0% | 87.0% | 93.7% | 94.3% |

The secondaries, one seed each from their promoted runs with harvested starts at the route's hearts
(`harvested_fraction: 0.33`): boss-manhandla `r30-manh-s90` (280 entries of 3:0x4d from the Dungeon 3 harvest
chain) 95.7%, −0.62 (`r78t4-manhandla2` 94.0%, −0.75; 3:0x4d 91.3% against 88.3%); gibdo `r30-gibdo-s91` (the way
back's entries: 0x65 and 0x66 from W 150 each, cut out of the leg-and-Dungeon-5 replays for 0x66, 0x47 150, 0x26
50; and the planner's play of 0x65, 0x66 from W and 0x47, 13,175 rows at `--demo-coeff 0.05`) 99.5%, +0.02
(`r75-f-a` 99.3%, +0.01).

**boss-manhandla, chosen on Dungeon 3 at the route's hearts** (`--from 0_37c_0 --stop-at step:105`; the leg after
Dungeon 1 arrives with the full route's hearts): `r30-manh-s90`'s last checkpoint screens at 84.9% against
`r78t4-manhandla2`'s 80.9% (+4.0 ± 1.2 paired, seeds 2000-2999) and **confirms at 82.0% against 77.4% (+4.6 ± 1.3,
seeds 3000-3999)**. On seeds 0-999, paired with `r29land-1k` (the same team but boss-manhandla): **`d3` 85.4% against 82.7%,
+2.7 ± 1.3** (3:0x4d kills 116 of 970 entering runs against 143 of 971, at the same 4.08 hearts in), `d8` 26.2%
against 24.8%, +1.4 ± 1.9 (8:0x6e crossed by all 1000, 0.84 hearts a visit against 0.82). Single rooms: 3:0x4d
91.3% against 88.3%, 8:0x6e 100% against 99.7%. **Promoted** in this branch's checkout (`triforce promote
boss-manhandla r30-manh-s90 --act temperature:0.5+relax:16`; its `ckpt.pt` is the chosen last checkpoint). Its 280
harvested 3:0x4d starts are not in git: `triforce/states/harvested/boss-manhandla` is rebuilt from the Dungeon 3
harvest chain above (its `boss-manhandla/3_4dw_*` files, all of them).

**Darknut and gibdo, chosen on Dungeon 5 at the route's hearts** (`select-checkpoint FAMILY RUN --from 0_45c_0
--stop-at step:209 --runs 1000 --first-seed 2000 --confirm 1000 --confirm-first-seed 3000 --confirm-top 1 --only
<the last checkpoints>,last`; the promoted team measured once, in `r30-dark-s87`'s batch, and the other selections
paired with it on the same seeds: each plays the same team with one run swapped). The D5 triforce taken:

| run | candidate | screen (2000-2999) | confirm (3000-3999) | hearts on arriving | confirm vs the promoted team |
|---|---|---:|---:|---:|---:|
| promoted team | `r78-dark37-s85`, `r75-f-a` | 48.5% | 49.0% | 5.35 | |
| darknut `r30-dark-s87` | it011462 | 53.6% | 52.6% | 5.43 | +3.6 ± 2.2 |
| | last | 49.7% | 49.5% | 5.18 | +0.5 ± 2.2 |
| darknut `r30-dark-s88` | it011492 | 54.2% | 53.4% | 5.29 | +4.4 ± 2.2 |
| | last | 49.7% | 50.3% | 5.28 | |
| darknut **`r30-dark-s89`** | **last (it 11530)** | 53.7% | **54.8%** | 5.38 | **+5.8 ± 2.2** |
| | it011462 | 49.1% | 48.5% | 5.34 | |
| gibdo **`r30-gibdo-s91`** | **last (it 1368)** | 52.8% | **54.3%** | 5.60 | **+5.3 ± 2.1** |
| | it001330 | 55.4% | 53.1% | 5.52 | |

All three darknut seeds gain over the promoted run on the segment it lost (+3.6 to +5.8 on fresh seeds), and gibdo
as much. The choice: darknut `r30-dark-s89` and gibdo `r30-gibdo-s91`, their last checkpoints (each run's
`ckpt.pt`). Their other segments (darknut `d8`, `d3`, `d5`; gibdo `d5`; 1000 runs, seeds 0-999) and the segment
with both swapped together (the gains were measured one family at a time), 1000 runs each, seeds 0-999 (`d3` and
`d8` paired with this checkout's team with the new boss-manhandla, `d5` with `r29land-1k-d5`):

| segment | reference | with the run | paired | where |
|---|---:|---:|---:|---|
| darknut `d8` | 26.2% | **34.3%** | **+8.1 ± 2.0** | 8:0x3e 354 deaths of 922 against 431 of 900, 0.89 hearts a visit against 1.42; 0x5e 6 against 23 |
| darknut `d5` (full hearts) | 72.2% | 73.9% | +1.7 ± 1.9 | |
| darknut `d3` | 85.4% | 82.2% | **−3.2 ± 1.6** | 3:0x69 from N 0.76 hearts a visit against 0.55; Manhandla entered with 3.80 hearts against 4.08, 147 deaths against 116 |
| gibdo `d5` (full hearts) | 72.2% | 73.3% | +1.1 ± 2.0 | |
| both, Dungeon 5 at the route's hearts (seeds 3000-3999) | 49.0% | **58.3%** | **+9.3 ± 2.2** | 5:0x05 1.45 hearts a visit against 1.70, 206 deaths of 1000 against 239 |

**gibdo `r30-gibdo-s91` qualifies and is promoted** in this branch's checkout (`triforce promote gibdo r30-gibdo-s91
--act temperature:0.5+relax:16`).

**darknut `r30-dark-s89` does not qualify: it loses Dungeon 3.** A second 1000 runs of `d3` on seeds 1000-1999, both
teams measured (`r30dk-check-s89-d3b`): 81.6% against 86.7%, −5.1 ± 1.7; the 2000 runs pooled **−4.2 ± 1.2**.
The cause is the same on both halves: 3:0x69 from N costs 0.74-0.76 hearts a visit against 0.55-0.56, and Manhandla,
entered with 3.80-3.84 hearts against 4.08-4.12, kills 141-147 runs against 106-116. The run is better where it was
trained to be (`d8` +8.1 ± 2.0, Dungeon 5 at the route's hearts +5.8 ± 2.2, with gibdo +9.3 ± 2.2) and worse in the
one room whose chain entries the arm left out (0x69 from N: its replay cut was dropped to start the training sooner,
while 0x59 and 0x5b, which kept theirs, hold). Not promoted; `r78-dark37-s85` stays.

**The continuation with 0x69 from N's entries** (`r30-dark69-s92`, `-s93`, `-s94`): from `r30-dark-s89` (it holds
`d8` and Dungeon 5), 244 iterations, the same arguments and demonstrations, and the harvested starts above plus 120
entries of 3:0x69 from N cut out of the Dungeon 3 harvest chain's replays (`scripts/experiments/r29_zora_states.py
REPLAYS OUT --step 95:3_69n --count 120`; all enter at X 120, Y 93; 1340 states). Chosen on `d3` (`select-checkpoint
darknut RUN --from 3_7cs_0 --stop-at step:105`, screen seeds 2000-2999, confirm 3000-3999, `r78-dark37-s85` pinned),
then Dungeon 5 at the route's hearts and `d8`. Beside it, the fallback: the two other seeds' chosen checkpoints
(`r30-dark-s88` it011492, +4.4 ± 2.2; `r30-dark-s87` it011462, +3.6 ± 2.2 on Dungeon 5) measured on `d3` and `d8`.

The fallback (`r30dk-check-s88s87`, seeds 0-999, paired with this checkout's team as for `r30-dark-s89`): both other
seeds hold Dungeon 3 and gain more on `d8`. 0x69 from N costs them what it costs the promoted run, so `r30-dark-s89`'s
loss there is that seed's drift, not the arm's:

| run (checkpoint) | `d3` | paired | 3:0x69 from N, hearts a visit | Manhandla: hearts in, deaths | `d8` | paired |
|---|---:|---:|---:|---:|---:|---:|
| reference (`r78-dark37-s85`) | 85.4% | | 0.55 | 4.08, 116 | 26.2% | |
| `r30-dark-s88` (it011492) | 86.0% | +0.6 ± 1.5 | 0.55 | 3.99, 112 | **36.6%** | **+10.4 ± 2.0** |
| `r30-dark-s87` (it011462) | 85.1% | −0.3 ± 1.6 | 0.62 | 3.94, 110 | **36.4%** | **+10.2 ± 2.0** |
| `r30-dark-s89` (last) | 82.2% | −3.2 ± 1.6 | 0.76 | 3.80, 147 | 34.3% | +8.1 ± 2.0 |
