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
(paired +25.0 ± 4.4). Chosen: `r31-oct-s97`. **Not promoted yet**: its paired 1000-run comparisons on the other
segments it plays (post-d2, d1, post-d1, post-d3, post-d6, post-d8) were queued and had not run when this was
written (the commands are below); promote when none is worse beyond noise.

## The team on the new route (clips on, 1000 runs a segment)

Arrivals (the runs that entered the next segment's start) against main's `r29land-1k` on the same seeds. This
branch's chains have no stop step and ran on past their segment; they are scored at the arrival and their records
cut there.

| Segment | new | main | paired | where runs are lost that were not |
|---|---:|---:|---:|---|
| d6 | 21.1% | 31.4% | −10.3 ± 1.7 | 6:0x29 119 (main 27), 0x39 49 (26), 0x2d 19 (6) |
| d8 | 24.8% | 24.8% | +0.0 ± 1.9 | 8:0x4c 13 stuck (0) |
| d9 | 10.9% | 31.0% | −20.1 ± 1.7 | 9:0x56 616 stuck (0), 0x55 from E 16 (0) |
| post-d2, d1, full | queued | | | |

- Dungeon 6: 6:0x39 720 visits, 14 bombs (12 runs); 6:0x2d 316 visits, 20 bombs (19 runs); no bomb cheated there.
  6:0x78 gives nothing: 0 kills in 1000 visits (as main: the wizzrobe model crosses without a swing), so no leader
  kill, no bomb pack, 0 hearts, 305 frames. Hearts into 6:0x3a 5.52 (696 runs) against 6.16 (801). Cheated: 0 keys,
  12 bombs (6:0x28, as main).
- Dungeon 8: 0 keys and 0 bombs cheated; the key 0x4c leaves costs nothing on the doors.
- Dungeon 9: hearts into 9:0x55 13.98 (438 runs, main 14.13); 3 bombs cheated (0x04, 0x31), 0 keys.

## Owed (not trained here)

- **like-like, 9:0x56** (step 310): 616 of 1000 Dungeon 9 runs end `stuck` there (the room is new to the family).
- **wizzrobe, 6:0x29** (step 268, the key now taken: 119 runs lost against 27) and **6:0x78** (`drops: first` has
  nothing to take: the model never swings there).
- **vire, 6:0x39 and 6:0x2d** (`bombs: tenth`: 49 and 19 runs lost against 26 and 6).
- **lanmola, 9:0x55 from E** (16 runs lost).
- **pols-voice, 8:0x4c** without its key (13 stuck).
- stalfos-keese-gel 1:0x53/0x43 from S: to be read from the queued d1 and full chains.

## For the owner

1. Your Dungeon 1 works as you said: out and back in, the N door is open for good, no key spent, the keys come out
   even. The overworld model needed a lesson for the step off the entrance (it used to step back onto it); after
   one round of training it does the trip in every run, in a third of a second on the overworld.
2. Dungeon 6 and Dungeon 9 lose runs for now: the rooms whose job changed (0x29's key, the Vires without bombs,
   Dungeon 9's new 0x56) are new to their models; they are listed above for the next training round. 0x78's bomb
   pack does not happen yet: the model walks through without swinging.
3. Dungeon 8 is unchanged by leaving 0x4c's key.

## Commands

```sh
TRIFORCE_MODELS_DIR=$PWD/models triforce train --specialist ow-octorok --load models/ow-octorok/r74-leg2/ckpt.pt \
  --run-id r31-oct-sNN --seed NN --iterations 244 --stop-success 1.01 --sil-coeff 0.1 --checkpoint-every 30 \
  --demos target/demos/r31/mouth-37.npz --demo-coeff 0.05 --parallel 8          # NN = 95, 96, 97
triforce select-checkpoint ow-octorok r31-oct-s97 --from 1_73s_0 --stop-at step:51 --runs 100 --confirm 0
triforce eval-batch r31-trip4 --chain "s97-c0: --route d2first --from 1_73s_0 --runs 20 --first-seed 1000 \
  --act model --stop-at step:51 --checkpoint ow-octorok=... --output ..."     # c0-c2, each seed and r74-leg2
triforce select-checkpoint ow-octorok r31-oct-s97 --segment SEG --only last --runs 1000 --confirm 0 --parallel 10
triforce eval-batch r31-1k-dungeons --chains r31-1k --runs 1000 --act model --chains-dir target/r31-chains \
  --segments "d6:6_79s_0 d8:8_7es_0 d9:9_76s_0" --parallel 10
triforce eval-batch r31-1k-full --chains r31-1k --runs 1000 --act model --chains-dir target/r31-chains \
  --segments full:start --chain-args "--checkpoint ow-octorok=models/ow-octorok/r31-oct-s97/ckpt.pt" --parallel 10
```
