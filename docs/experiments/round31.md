# Round 31: the owner's phase IX on `d2first`

## The owner's changes (Nov 19)

The owner watched the first complete game by the models (`corpus/replays/v1-golden-ending.tfreplay`) and asked for
eight changes to `d2first`, walked before anything was built in `docs/routes/phase9-walks.md`:

1. Dungeon 1: out of 1:0x73 by its S door and straight back in, so its N key door is open (no key spent).
2. Dungeon 1: the key rooms 1:0x72 and 1:0x74 left out.
3. Dungeon 1: 1:0x53's N wall bombed into 0x43 from S (0x52 and 0x42 left out).
4. Dungeon 6: 0x78, a ringleader room: opportunistic kills for a bomb pack (no forced fight).
5. Dungeon 6: 0x29's key taken.
6. Dungeon 6: no bombs on the Vires of 0x2d (and 0x39) once Link holds the magic sword.
7. Dungeon 8: 0x4c's key left.
8. Dungeon 9: 0x66 N into 0x56, its W wall bombed into 0x55 from E (0x65 left out).

## What was built

- `triforce/routes.yaml` `d2first`: 354 steps (four fewer). Dungeon 1 is steps 46 (1:0x73 S→S), 47 (0:0x37
  `cave → cave`), 48 (1:0x73 S→N, `1_73s_v2`), 49 (0x63), 50 (0x53 S→N), 51 (0x43 from S), 63 (the triforce); 0:0x37
  out of it is step 64 (`0_37c_v2`). Every step from the old 56 on is 4 lower: the D7 triforce 243, Dungeon 6
  261-276 (0x78 = 262 `drops: first`, 0x29 = 268 takes its key, 0x39 = 269 and 0x2d = 273 `bombs: tenth`),
  Dungeon 8 283-292 (0x4c = 290 `treasure: skip`), Dungeon 9 308-351 (0x66 = 309 S→N, 0x56 = 310 like-like
  `treasure: skip`, 0x55 = 311 from E), Ganon 352, Zelda 353.
- `triforce/game.yaml`: 1:0x73's S exit to 0:0x37 (one way; the only dungeon door out of its level), its N door and
  0x63's S side `keep_out` until the step that leaves through it, `requires_bombs` on 1:0x53 and 9:0x56.
- The vocabulary `drops: first` (a step's own "take the wanted drops in the room first"), `step:N` segment targets
  (`d1` ends on step 64), `heart_value.py --measured-on` (the 1.9.0.0 table remapped onto the new steps).
- Start states: Dungeon 1 rebuilt (steps 47-64), Dungeon 6-8 rebuilt from 6:0x29 on (one key more), 9_56s and
  9_55e new, the states of the rooms that left the route deleted; audit clean, 1033 states. The curricula of the
  rebuilt states (`scripts/experiments/r31_curricula.sh`): every crossing solved; fights as rounds 27-28 (6_3aw's full
  fight 2 of 3, its staged tracks 3 of 3).

## What the engine needed

- `GameMap::exit_direction` aimed a dungeon room's door out of its level by `MapLocation::direction_to`, which reads
  any level → overworld move as N (the triforce's warp): 1:0x73 → 0:0x37 pointed N, into the kept-out key door. A
  dungeon room's game.yaml door out of the level now aims by its side (test `a_level_door_out_is_its_side`).
- The `keep_out` on 1:0x73 N: without it every run left by the N key door on step 46 (keys 3 → 2) and re-planned
  past the trip.
- Nothing else. The tests that changed with the data (step numbers, the deleted states as fixtures, the door
  tests) follow the route. Two `crates/zelda/tests/d9.rs` failures (the walk starts at `7_2bw_0` and now plays the
  new Dungeons 6, 8 and 9) were the walk's state, not phase IX's fault:
  - **0:0x05, BOMB at (`$6C`,`$A5`)** (step 307): not the tenth-kill bomb (HelpDropCount 0 on every tick: the walk
    keeps it there). The walk now comes to the rock wall W along its lower row (Y `$A5`) and the mask offers BOMB W
    beside it: its landing (`$5C`,`$A5`) is within `$10` of the wall's tile object (`$50`,`$A0`), the game's reach
    (asm:RockWall_Update). The test asserted a window under the wall; it now asserts the game's reach on every
    offered BOMB's landing.
  - **9:0x04, no CLIP** (step 348): bag, keys and entry side as before; the shorter block clip S leaves from the
    grid point X `$70` of tile (14, 4), and Link's movement fraction now took him `$75` → `$71` → `$6C` (a MOVE W 4
    goes 4 or 5 px): never on it. A walker aid (controller input, a frame at a time) lines him up when 1-3 px off
    such a point; CLIP is played at 0x04 again.
  - Also fixed: `game.rs` 1:0x23's key at (`$80`,`$70`) (level-1.md); `ported_game.rs` read the visited bit as `$80`
    (a kill-count bit) where it is `$20` (ram-catalog R.1); `objectives.rs`: a keyless Link at the D1 entrance now
    gets the route's own S exit (no key room is a route exit any more), and `start_route` is called with
    `1_73s_0`'s visit (None drew step 46 or 48 at random).
- Suites on the final code: pytest 834 passed (9 skipped), cargo `--workspace --features zelda/zcore` 1025 passed,
  zelda-asm tests 5 passed, `asm.py check` ok, `tf replay verify corpus/replays/*.tfreplay` 5 of 5.

## The trip, before and after

Chains from `1_73s_0/_1/_2` to step 51 (20 runs each, seeds 1000-1019):

| ow-octorok | trip played | stalled on 0:0x37's mouth | 0:0x37 `cave → cave` frames | trip frames |
|---|---:|---:|---|---|
| r74-leg2 (main's) | 51 of 60 | 9 | median 1036, mean 1897, max 6996 | mean 2340 |
| r31-oct-s95, s96, s97 | 60 of 60 each | 0 | 200 every run | 643 every run, 0 hearts |

The earlier measure (seeds 0-19, after the engine fix) was 50 of 60, median 1364. The cause was known: on the mouth
the model stepped S 4 (Y `$83`, `UndergroundExitType` still 2) and N back onto it; S 8 clears it
(asm:UndergroundExitType). No key is spent in 0x73 and none is cheated in either.

## What was trained

ow-octorok, three seeds continued from `r74-leg2` (244 iterations, `--sil-coeff 0.1 --checkpoint-every 30`, the
mouth demonstrations `target/demos/r31/mouth-37.npz` at `--demo-coeff 0.05`: 1800 rows of the rule that wins 900 of
900). Every checkpoint of s97 arrives at step 51 in 100 of 100 runs from `1_73s_0` against 75 of 100 for r74-leg2
(paired +25.0 ± 4.4). Chosen: `r31-oct-s97`, then paired with `r74-leg2` on 1000 runs of every segment ow-octorok
plays (seeds 0-999, the rest of the team main's, `select-checkpoint ow-octorok r31-oct-s97 --segment SEG --only
last,promoted`): `post-d2` 92.3% against 93.2% (−0.9 ± 1.2), **`d1` 99.1% against 66.5% (+32.6 ± 1.5**: `r74-leg2`
stalls on 0:0x37's mouth), `post-d1`, `post-d3`, `post-d6` 100% both, `post-d8` 99.9% against 100.0% (−0.1 ± 0.1).
None worse beyond noise: **promoted** in the landing (1.9.0.1, below).

## The team on the new route (clips on, 1000 runs a segment, seeds 0-999)

Arrivals (the runs that reached the next segment's start). Three chains of each segment, paired by seed: main's
record `r29land-1k` (1.9.0.0's team on the old route), `r31-1k` (today's team, i.e. 1.9.0.0's with `r31-oct-s97`, on
the new route: the route's effect) and the landed team `r31land-1k` (round 30's runs, `r31-oct-s97` and darknut
`r30-dark69-s94`, on the new route; round30.md "The landing"). `r31-1k`'s Dungeon 6, 8 and 9 chains had no stop step
and ran on; they are scored at the arrival and cut there.

| Segment | `r29land-1k` | `r31-1k` (route) | `r31land-1k` (landed) | landed − main | landed − `r31-1k` | where runs are lost that were not |
|---|---:|---:|---:|---:|---:|---|
| after Dungeon 2 (→ step 46) | 93.2% | 92.3% | 94.9% | +1.7 ± 1.1 | +2.6 ± 1.1 | |
| Dungeon 1 (→ step 64) | 99.8% | 99.1% | 99.3% | −0.5 ± 0.3 | +0.2 ± 0.3 | 1:0x53 2 (0), 0x44 3 (1) |
| after Dungeon 1 (→ step 83) | 100.0% | | 100.0% | 0 | | |
| `d6` (→ step 277) | 31.4% | 21.1% | 28.5% | −2.9 ± 2.0 | +7.4 ± 1.9 | 6:0x29 74 (27), 0x2c 89 (56), 0x2d 29 (6), 0x39 36 (26) |
| `d8` (→ step 293) | 24.8% | 24.8% | 36.3% | +11.5 ± 2.0 | +11.5 ± 2.1 | 8:0x4c 19 stuck (0) |
| `d9` (→ the ending) | 31.0% | 10.9% | 13.7% | −17.3 ± 1.8 | +2.8 ± 1.2 | 9:0x56 616 stuck (0), 0x55 from E 16 (0) |

The route's own cost is the `r31-1k` column: Dungeon 6 −10.3 ± 1.7 (0x29's key, 0x39 and 0x2d without bombs),
Dungeon 9 −20.1 ± 1.7 (9:0x56, a room new to like-like: 616 of 1000 runs end `stuck` there), Dungeon 1 −0.7 ± 0.3.
The landed team wins back most of Dungeon 6 (wizzrobe `r30w-wj-s89`) and adds 11.5 points on `d8` (darknut); like-like
has no new run, so Dungeon 9 stays the route's.

**The full route** (`full`, from the game start, 1000 runs; milestones on the new step numbers):

| reaches | `r29land-1k-full` | `r31-1k-full` | `r31land-1k-full` | landed − main |
|---|---:|---:|---:|---:|
| Dungeon 2 left | 87.8% | 85.6% | 89.4% | +1.6 ± 1.4 |
| Dungeon 1 entered (step 46) | 82.2% | 79.6% | 84.9% | +2.7 ± 1.6 |
| Dungeon 1 left (step 64) | 73.0% | 71.1% | 74.4% | +1.4 ± 2.0 |
| Dungeon 3 left | 59.0% | 56.4% | 62.1% | +3.1 ± 2.1 |
| Dungeon 4 left | 54.0% | 50.9% | 55.6% | +1.6 ± 2.2 |
| the D5 triforce | 26.2% | 23.5% | 30.4% | **+4.2 ± 2.0** |
| the D7 triforce | 9.1% | 8.2% | 12.5% | **+3.4 ± 1.4** |
| the D6 triforce | 1.7% | 1.7% | 3.0% | +1.3 ± 0.7 |
| the D8 triforce, Dungeon 9 entered | 0.4% | 0.5% | 1.3% | **+0.9 ± 0.4** |
| the red ring | 0.3% | 0.2% | 0.2% | −0.1 ± 0.2 |
| the game ended | 0 | 0 | 0 | |

**The honest planner beside** (`temperature:0.5+relax:16+plan:16:1`, 300 runs, the landed team; main's
`r29land-plan-full`): the D6 triforce 47.0% (43.0%), the D8 triforce 42.3% (42.0%), Dungeon 9 entered 42.3%
(42.0%), the red ring 10.3% (39.7%), Ganon reached 9.0% (30.0%), **the game ended 9.0% (27 runs; 1.9.0.0 29.7%)**:
90 of the 127 planner runs in Dungeon 9 end `stuck` in 9:0x56. The planner plays the models' own actions, so a room
the model cannot leave stops it too.

- **The trip** is played by every run that reaches it: 1000 of 1000 in `d1`, 849 of 849 in the full route; 0 keys
  spent in 1:0x73, 0 cheated.
- **6:0x78 with `drops: first`, untrained:** 0 kills in 1000 visits on `d6` (and 0 in 81 visits in `r31-1k-full`): no leader
  kill, no bomb pack, 0.01 hearts, 304 frames: the wizzrobe model walks through without a swing. `drops: first`
  only marks what to take; it gives nothing until a model is trained to make the drop. wizzrobe is owed it.
- 6:0x39 829 visits, 13 bombs; 6:0x2d 437 visits, 27 bombs (`bombs: tenth`); no bomb cheated there.
- Cheated items: no key in any segment or leg. The full route has 2 cheated keys, at 6:0x79 (seeds 463, 492; main's
  record 0): both runs left 5:0x66 (step 195) without its key, as main's gibdo does in 6 of 313 full-route exits
  (the landed one 7 of 342), and spent two keys in 7:0x1b, as goriya does in 2% of `d7` visits on both teams; more
  runs now reach Dungeon 6 (125 against 90). `r31-1k-full`'s one cheated key was 9:0x63 (seed 547: 6:0x29 left by a
  wrong exit without its key, so Dungeon 9's ledger, which has no spare key on either route, ran short). The
  planner: 6 cheated keys, all 9:0x63, for the same three causes (0x29 wrong exits, 5:0x66's key left, one 8:0x5e
  key left). No route-data fault.

## Owed (not trained here)

- **like-like, 9:0x56** (step 310): 616 of 1000 Dungeon 9 runs end `stuck` there (the room is new to the family);
  the planner too (90 of 127).
- **wizzrobe, 6:0x29** (step 268, the key now taken: 74 runs lost against 27 even with `r30w-wj-s89`) and **6:0x78**
  (`drops: first` has nothing to take: the model never swings there).
- **vire, 6:0x39 and 6:0x2d** (`bombs: tenth`: 36 and 29 runs lost against 26 and 6).
- **lanmola, 9:0x55 from E** (16 runs lost).
- **pols-voice, 8:0x4c** without its key (19 stuck).
- stalfos-keese-gel 1:0x53 and 0x43 from S: 1 and 10 deaths in the full route (main's 1:0x53 1, 0x43 4): small.

## For the owner

1. Your Dungeon 1 works as you said: out and back in, the N door is open for good, no key spent, the keys come out
   even. The overworld model needed a lesson for the step off the entrance (it used to step back onto it); after
   one round of training it does the trip in every run, in a third of a second on the overworld. It is promoted.
2. Landed as 1.9.0.1 together with the optimisation round (round 30): from the start the models now take the D5
   triforce in 30% of runs (26%), the D7 triforce in 12.5% (9.1%), and reach Dungeon 9 in 13 of 1000 (4).
3. Dungeon 6 and Dungeon 9 lose runs to the route for now: the rooms whose job changed (0x29's key, the Vires without
   bombs, Dungeon 9's new 0x56) are new to their models. Dungeon 9's 0x56 is the big one: two runs in three that
   reach it from Dungeon 9's entrance stay there, so the models end the game from Dungeon 9's entrance in 14% (31%
   before) and the planner from the start in 9% (30%). like-like is the first family to train on the new route.
4. 0x78's bomb pack does not happen yet: the model walks through without swinging. Dungeon 8 is unchanged by
   leaving 0x4c's key (19 runs of 1000 stick in 0x4c, pols-voice's to learn).

## Commands

```sh
TRIFORCE_MODELS_DIR=$PWD/models triforce train --specialist ow-octorok --load models/ow-octorok/r74-leg2/ckpt.pt \
  --run-id r31-oct-sNN --seed NN --iterations 244 --stop-success 1.01 --sil-coeff 0.1 --checkpoint-every 30 \
  --demos target/demos/r31/mouth-37.npz --demo-coeff 0.05 --parallel 8          # NN = 95, 96, 97
triforce select-checkpoint ow-octorok r31-oct-s97 --from 1_73s_0 --stop-at step:51 --runs 100 --confirm 0
triforce eval-batch r31-trip4 --chain "s97-c0: --route d2first --from 1_73s_0 --runs 20 --first-seed 1000 \
  --act model --stop-at step:51 --checkpoint ow-octorok=... --output ..."     # c0-c2, each seed and r74-leg2
triforce select-checkpoint ow-octorok r31-oct-s97 --segment SEG --only last,promoted --runs 1000 --confirm 0 --parallel 10
triforce eval-batch r31-1k-dungeons --chains r31-1k --runs 1000 --act model --chains-dir target/r31-chains \
  --segments "d6:6_79s_0 d8:8_7es_0 d9:9_76s_0" --parallel 10
triforce eval-batch r31-1k-full --chains r31-1k --runs 1000 --act model --chains-dir target/r31-chains \
  --segments full:start --chain-args "--checkpoint ow-octorok=models/ow-octorok/r31-oct-s97/ckpt.pt" --parallel 10
# the landed team's chains of record: round30.md "The landing"
```
