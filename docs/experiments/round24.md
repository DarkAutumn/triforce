# Round 24: the leg to Dungeon 5 and Dungeon 5 (the recorder, Digdogger); 1.5.0.0

The owner (Nov 14): "The route to d5 should have already been captured by the other agent, and checked in with d5-9
routes." (`triforce/draft_routes.yaml` `first-quest`, `docs/routes/first-quest.md`, `docs/routes/overworld.md`,
`zelda-asm/levels/level-5.md`.) "This is the new round. What we saved to the backlog will be after d5, which is
incredibly hard. We want d5 completed so we can raise its success rate with the next round's work."

His Dungeon 5 notes (verbatim essentials; §6 says how each is carried and what the models do):

- "Room 0x05 is one of the hardest rooms in the entire game. It has a special name: Olive Garden." (The flute looks
  like a breadstick, and the only reason to go there is the breadstick.)
- "Bombs shouldn't be used in this dungeon except in 0x05 and 0x24 (olive garden and the boss)."
- "Don't kill 0x66 on entry, save that for passing back through (key included unless it's convenient or in the
  way). We move quickly from 76 -> 66 -> 64 without fighting much unless we have to on dungeon entry."
- "0x65 has bombs. You should discourage taking them when headed west to 0x64, strongly encourage taking them moving
  east to 66 on the way back."
- "0x64 fighting should be discouraged, just clip the blocks to the stairs. [...] would be interested to see how this
  room does with no clip...probably brutal. Blue darknuts should be with the red darknuts model wise."
- "0x05 is the olive garden. Bombs are ok to use here down to 0, the idea is that we pick them back up in 0x65
  before moving west on the way out."
- "0x56: The bombs in this room are a non-objective. [...] for now a quick passthrough is fine." "0x57: don't stop
  for the 5 rupee room kill."
- "0x47 has a very fast path through it, walk straight north [...] hit with your sword to cause invuln frames to
  walk through." "0x37 has a similar route [...] walk left another half tile [...] step across the water (remember
  the ladder)." "0x25 is tough, don't fight the pols voices, but sword-invuln here is the way through, not killing."
- "0x24 boss room. You have to use the flute to turn him into a smaller version. [...] get near him in large form,
  drop a bomb, instantly hit menu for flute and play it. He'll turn little just in time to get hit (and die) to a
  bomb. Fire in the room being thrown by statues aren't blockable without magic shield which we never have."

His answers (Nov 14): (1) **the leg and the dungeon together**: build the overworld leg from the D4 triforce first,
so that Dungeon 5's start states are played, not made; (2) **add the kinds and convert the models**: recorder, food
and candle in one model-incompatible change, the checkpoints widened, not retrained, the proof identical evaluation
results before and after; (3) **0x65's bombs never westbound**, whatever Link holds (an empty bag gets a cheated bomb
for each wall; the Olive Garden is fought with what he holds; the four bombs are taken on the way back); (4) **run
everything overnight, then tag and push**: builds, trainings, 1000-run chains, the round document, 1.5.0.0, commits
on `main`, tag, push, once the dungeon is completed and the suites are green. Settled by the lead (told to the owner,
no objection): the draft's 23 Dungeon 5 steps with the notes as rules; with clips off only 0x64 differs (the fight
and the push); families by enemy (Blue Darknuts join `darknut`; new `gibdo`, `pols-voice`, `boss-digdogger`,
`dodongo-crossing`; 5:0x57 joins `zol`, the cellars `equipment-room`); the tenth-kill bomb stays allowed; 0x37 by
the owner's column (X `$70`); Digdogger by the fastest sequence the game allows.

Built on branches `r24-kinds`, `r24-names`, `r24-d5`, `r24-leg`, `r24-maze`, `r24-route-wip`, `r24-d5b` and
`r24-legstates`, merged into `main` at `a75e40c` (1.4.2.0); trained and measured as `r74`; versioned 1.5.0.0.
Decisions: `docs/decisions.md` 2026-11-14 ("Three item kinds and the model conversion", "State names for later
visits", "Dungeon 5 as route steps", "Digdogger's recorder and rewards", "The leg to Dungeon 5: its clip way", "A
maze loop is a room transition", "Dungeon 5's states and curricula", "The leg and Dungeon 5 in `d2first`", "The
leg's states and curricula") and 2026-11-15 ("Round 24's training and promotions", "darknut r74-darknut5 promoted
on the Olive Garden's hearts", "The played states after the Olive Garden", "1.5.0.0"). Routes: `docs/routes/d5-leg.md`
(steps 134-185), `docs/routes/d5.md` (steps 186-208).

## Summary

- **Dungeon 5 is completed by the models.** `d2first` now runs from the game start to the D5 triforce (209 steps).
  1000 runs per segment, act `temperature:0.5+relax:16`, clips on | off: **from Dungeon 5's entrance 34.9 | 22.0%
  reach the D5 triforce, from the D4 triforce (the leg and Dungeon 5) 15.2 | 2.9%, from the game start 4.8 |
  0.3%.** With darknut r74-darknut5, promoted after those chains (§9), the Dungeon 5 segment is **46.4 | 34.0%**.
- **The earlier route held or improved, but for Dungeon 4 with clips off** (§10): arrivals D2 89.9 | 89.9% (round 23
  89.9 | 89.9), D3 67.3 | 67.3% (60.1 | 60.1), D4 81.3 | 59.7% (78.6 | 64.7); from the start Link reaches the D4
  triforce in 33.3 | 19.2% of runs (round 23 28.7 | 15.3%) and Dungeon 5's entrance in 33.3 | 17.5%: **the leg loses
  almost nothing with clips on** (99.8% of its runs arrive; 92.0% with clips off).
- **What loses Dungeon 5** (§11): the Olive Garden (131 of 1000 runs of the D5 segment die there; from the D4
  triforce 398 of the 927 that reach it, arriving with 0 bombs and 5.9 hearts), then the way back at low hearts:
  0x66 W→N (140), 0x64 back E (104), 0x37 (94), 0x47 (80). With clips off 0x64's plain way (5 Blue Darknuts) kills
  261 of 1000 D5 runs, and from the D4 triforce 639 of the 815 that reach it: the leg's plain ways bring Link to
  Dungeon 5 with 3.1 hearts.
- **Built for it:** three B-item action kinds (RECORDER, FOOD, CANDLE) with the 1.4.1 models widened and proven
  bit-identical (§2); state names for a room entered the same way twice (§3); the Lost Hills as a route exit (§4);
  round the world, the overworld ladder and its clip, 0x55's wrap over its river, 0x1f's wall gap, 0x5f's heart as
  a room objective, two caves (§5); Dungeon 5's rules note by note (§6); staged starts by the game's room history
  and skipped fights for the state builds (§7); 219 new start states (the leg 150, Dungeon 5 69) and curricula.
- **Trained** (§8): 18 runs, 4 new families from scratch (gibdo, pols-voice, boss-digdogger, dodongo-crossing, all
  99-100% in single rooms), 11 continued from their converted runs; darknut five times (the Olive Garden 0% for the
  old model, 79-81% now). **The Olive Garden's curve** (§9): with 0 / 2 / 4 / 8 bombs held, r74-darknut 39 / 45 /
  50 / 59% → r74-darknut5 73 / 81 / 82 / 87%; hearts lost a fight at 2 bombs 6.1 → 3.0.
- **The played states after the Olive Garden** (§12) replace the 48 `fight_skipped` ones: harvested from chain runs
  that cleared it, refilled, audited clean (613 states), curricula explored again. The seven families' promoted runs
  play them as well as the made ones in every room but two, slightly harder: 0x66 W→N 95.0% −1.14 hearts (made
  97.7% −0.55) and 0x64 on the way back 96.7% −1.64 (97.7% −1.08).
- **Version 1.5.0.0** (§13): MINOR 5 (five dungeons), COMPAT and REV back to 0; the promoted runs (1.4.2.0, the same
  layout) are stamped.

## 1. What the scouts found

Four read-and-measure scouts ran before any building (their scripts are throwaway; their findings are in the route
documents).

- **The leg** (scout LegScout; `docs/routes/d5-leg.md`): 52 steps from the D4 triforce's warp (0x45) to 5:0x76 in
  three stretches with a clip way and a plain way each (`coast-heart`, `lake-3e`, `mountain`). Two route-load errors
  blocked it outright (0x60's wrap "round the world" into 0x5f had no `route_entry`; a step leaving 0x1b by the Lost
  Hills' sequence was refused), and five things were missing before either way could be played: the overworld ladder
  (plain and clip), 0x5f's heart container as an objective, 0x1f's wall gap, the caves of 0x2f and 0x0f, and nine
  state names colliding with `d2first`'s. The wrap exits of 0x51, 0x3e and 0x0c, the raft docks, `stretch:` blocks
  and the cave waypoints already worked.
- **Dungeon 5's engine** (D5Engine): three things needed code (the tunnel cellar 5:0x07 accepted either ladder; the
  recorder was neither route progress nor an action; nothing could write a clean entrance state); most per-step
  rules (treasure, bomb reserve, shutters, push blocks, carried items, the block clip in 0x64) needed only yaml.
- **Dungeon 5's crossings** (D5Paths, 821 hits on Link measured): **sword-through is real**: a monster flashing
  from a sword hit never hurt Link (Gibdos, Red Darknuts hit from the side or behind, Pols Voices; Dodongos ignore
  the sword, a head-on Darknut parries). A swing costs 13 frames on the floor and 0-3 on a laid ladder. Fastest safe
  paths: 0x37 at **X `$70`, swinging: 122 frames, 0 hits in 54 of 54** (X `$78`, `level-5.md`'s, is hit in 17-30 of
  54); 0x57 row 3 with the ladder (138 frames, no hit); 0x56 row 6 then column 11 (no hit, against 9 of 27 hit on row
  3); 0x47 straight up swinging (0.41 hearts); 0x25 rows 0 or 6 (rows 2 and 4, the draft's, are hit in 21-27 of 27
  even swinging: the Pols Voices come from the side). A slot-1 Gibdo's item is taken by touch whatever its timer.
- **Digdogger** (D5Boss): **one bomb never kills the small Digdogger.** The owner's bomb-then-recorder does land (a
  bomb whose START came within 46 frames blasts at the small one's first hittable frame, B+168: the fuse runs during
  the recorder's freeze, the blast waits for it to end) but only halves it (HP `$80` → `$40`). The fastest sequence
  adds a white-sword swing on that frame (A at B+157..164, no direction held: `$20` left) and one more hit when its
  invincibility ends at B+199/200: killed at B+199..243 in 11 of 12 RNG phases, 528-968 frames from entering, 0-4
  hearts lost (median 0.75), almost all to the statues' fireballs. The split comes at **B+152**, not `$98` + `$40`
  frames after B as `zelda-asm/enemies.md` said (corrected in this round): the shrink runs inside the freeze.

## 2. Action kinds and the conversion proof (1.4.2.0)

RECORDER (8), FOOD (9) and CANDLE (10) after CLIP, every older head index kept: B items chosen on the sub-screen
by buttons and used with B, no memory written. RECORDER takes no direction and is offered only where a big Digdogger
hasn't heard it (`actions::recorder_wanted`); the tune's 151 frozen frames count as menu frames. FOOD and CANDLE are
cardinal, tested in the game and masked for every family (`items:` turns them on; none does yet). The observation's
kind one-hot grows to 11 and `link` ends in `recorder_usable`, `food_usable`, `candle_usable`; with the maze's
`maze_step` (§4) 7 columns are added (observation schema 5, action schema 3, mask 143, joint 352).

**The models were widened, not retrained** (`triforce convert-model`): zero columns by feature name, the new kinds'
rows WAIT's with bias −5, embeddings from BOMB/WAIT, Adam moments zero-padded; the Rust policy drops all-zero
first-layer columns and keeps the draw's noise columns fixed, so a converted model is bit-identical. **Proof:** the
24 promoted runs, 20 episodes per state, seed 0, original on 1.4.1.2 against converted on 1.4.2.0, argmax and
`temperature:0.5+relax:16`: every pair identical per episode (outcome, steps, frames, hearts, whole records).
`boss-digdogger` has `items: {recorder: true}`; the play that starts the split pays `reward-recorder-split` (0.75)
once a room, and the small Digdogger (`$18`) is a boss type (`reward-boss-hit`, the boss kill for linger).

## 3. State names for later visits

A route may now enter a room from the same side more than once (0x59-0x5d from W on the coast's plain way, 0x2d S,
0x2c E, 0x1c S, 0x1b E, the tunnel cellar 5:0x07): the first visit keeps the bare name, a later one is
`<level>_<room><side>_v<N>_<chain>` (`0_1be_v2_0`, `5_07cave_v2_1`; `zelda::game_map::route_visits`). A state starts
the step its name spells (single-room envs, `--from`, room-stats), and the audit counts heart containers along the
stretch way a step is on (8 after Dungeon 4, 9 after 0x5f, 10 after 0x2f's cave, 11 after Digdogger). Every existing
name is a first visit; 2382 resets of every family started the same (route, step) before and after.

## 4. The Lost Hills

The mountain's plain way leaves 0x1b by the Lost Hills' sequence (N four times): a step says `exit: maze`. A maze
loop (`zelda::MazeLoop`: Link stays in the room and `MazeStep` `$52F` went up or the step ended on a room transition)
is a room transition everywhere a run is followed: a correct loop goes on and pays `reward-new-location` once per
maze step, a wrong one ends the episode `exit-wrong`, chain records count `maze_loops`/`maze_wrong_loops`, the
explorer's cell holds the maze step. The observation's `room_state` ends in `maze_step` (MazeStep / 4 in a maze
room). In the chains ow-start crosses it every time (925 of 925 visits with clips off, 1048 frames).

## 5. The leg's mechanics

- **Round the world:** `MapLocation::direction_to` follows the game's rule (asm:Room_CalcNextRoomForDoor): W out
  of column 0 is room id − 1, so 0x60 W enters 0x5f from E (rows 9-10 only).
- **0x55's wrap over its river:** game.yaml `wrap_in: [9, 10]` makes `CLIP E` from the dock a path-search jump onto
  0x55's W part, on those rows only.
- **The overworld ladder and its clip:** the room model knows overworld water and its one-square crossings in the
  game's ladder rooms; MOVE onto water is offered where the game lays the ladder; `CLIP W` from 0x5f's water arrival
  takes the heart container in 38 frames (8 → 9 hearts).
- **0x1f's wall gap** (N at X `$80` only) is in the room model and the exit tiles.
- **0x5f's heart as a room objective** (an overworld room's own item, `treasure: skip` leaves it), and the caves of
  0x2f (the take-any cave's right ware: 10 hearts) and 0x0f (100 rupees); a step whose exit is `cave` makes its cave
  the objective.
- **The clips switch** waits while Link is stranded (0x5f's water: no walkable exit, one reachable by skips), and on
  the coast holds the clip way from 0x55's W part to the rejoin 0x4f (tested in each clip-way room).
- Families by enemy: Zoras → ow-zora (0x54, 0x5f, 0x4f, 0x3f, 0x3e, 0x2e, 0x6f), Moblins → ow-moblin (0x53, 0x52,
  0x51, 0x6d, 0x6e), Lynels → ow-lynel (0x50, 0x60), Peahats → ow-peahat (0x1d, 0x1f), Tektites → ow-tektite (0x1e,
  0x0d, 0x0c), 0x0b's Leever → ow-leever, the caves, 0x2f and 0x0f → ow-start (one room per episode since).

In the chains (§10) 998 of 1000 runs cross the leg's clip way (two strayed back into Dungeon 4; 0x55's wrap 0.93
clips a visit and 91 wrong exits in 1024 visits, every other clip room 1.00 a visit); its plain ways lose 80 of
1000 runs, most in 0x1e (19), 0x1f (16), 0x2c (16) and 0x2d (13), and bring Link to Dungeon 5 with 3.1 hearts (6.6
on the clip way).

## 6. Dungeon 5's rules, note by note

| Note | How the route carries it | What the models do (D5 segment, clips on, 1000 runs) |
|---|---|---|
| 0x66 on entry: no fight, key not wanted | `treasure: skip`; W bomb wall `requires_bombs`, gibdo `required` | crossed in 560 frames, one bomb at the wall every visit, 0.09 hearts; the key taken in 27 of 1000 |
| 0x65 westbound: no bombs | `treasure: refuse` (no objective; the pickup costs `penalty-refused-treasure`) | **taken in 156 of 1000** (15.6%; 194 of 998 from the D4 triforce): the penalty is not absolute |
| 0x64: clip the blocks to the stairs | one step, the clips switch decides: on, `CLIP N` from (`$A0`,`$9D`) into the diamond, no fight | 1.00 clips a visit, 205 frames, no hearts lost; **clips off: the fight, 261 deaths of 998, 5.15 hearts a visit** |
| 0x05, the Olive Garden: bombs free down to 0 | push forces the fight; `bomb_reserve: 0` on its first visit | 131 deaths of 996 (4.19 hearts a visit, 1907 frames, 2.6 bombs placed a visit) |
| the four bombs of 0x65 taken eastbound | default treasure; the Treasure target rides on the slot-1 Gibdo | the bombs reach Digdogger: 0 bombs at 0x24 in only 57 of 369 arrivals |
| 0x66 on the way back: the fight, the key | default treasure (the carrier first), the N shutter forces the fight, N by the ladder | 140 deaths of 722 visits (1.08 hearts a visit); the key taken in 609 |
| 0x56's bombs a non-objective, 0x57's rupees not stopped for | `treasure: skip` | 0x56 0 deaths of 577, 0x57 7 |
| 0x47 straight north, sword-through | `treasure: skip` (key) | 80 deaths of 570 (0.82 hearts a visit), on arrivals at low hearts |
| 0x37: X `$70`, one ladder crossing | `treasure: skip` (compass); the objective's column | 94 deaths of 490; the probe saw the column at X `$78` (median), not `$70` |
| 0x25: no fight, sword-through | no treasure; W key door | 11 deaths of 380; crossed by row 6 (the probe: 44 of 44) |
| 0x24: bomb, recorder, kill | `items: {recorder: true}`, `reward-recorder-split`; `bomb_reserve: 0` | 20 deaths of 369; recorder once a visit, 0.94 bombs a visit, 1010 frames |
| bombs only in 0x05 and 0x24 | `bomb_reserve: 8` elsewhere (Link holds at most 8), the walls `required` | bombs placed off those rooms: the walls (one each), 107 on 0x66's way back (its fight), 27 in 0x47 |

## 7. Staged starts, states and curricula

- **Staged starts** (training envs only): `ram_randomization: {room_kills: {'(5, 0x05)': [1, 5], '(5, 0x64)': [1,
  5]}, room_kills_share: 0.5}` banks k kills in the game's room history before entry (asm:RoomHistory_ModifyObjCountUW),
  so the game itself makes k fewer Darknuts and clearing the rest clears the room. Evaluation, chains, exploration
  and builds meet the full room.
- **Skipped fights for the state builds:** `entry_states build --skip-fights 5:0x05:E` crossed the Olive Garden with
  every Darknut put into its death sparkle as it came out (the game's own kill), so the 48 states after it existed,
  marked `fight_skipped` (replaced in §12).
- **States:** the leg 150 (the clip way by `build`, chain `c` from `4_03s_<c>`; the plain ways by `path`), Dungeon 5
  69 (from the leg's 0x0b entrance); audit 613 states, 0 problems. Dungeon 5's chains 1 and 2 descend from chain 0's
  Dungeon 4 history (0 bombs at the entrance, chain 0 4).
- **Curricula:** 151 leg tracks (ow-zora, ow-moblin, ow-peahat, ow-leever on their new states; ow-lynel and
  ow-start new, on all of theirs) and every D5 state of every family, plus darknut's staged tracks: 0x64's plain way
  with 2 alive (`~k3~plain`), the Olive Garden with 1-2 alive (`~k5`, `~k4`), and during training 3 alive (3 of 3
  states solved, 8000 iterations), 4 alive (3 of 3, 12000) and 5 alive (1 of 3, 12000); 6 alive was never solved
  (16000 iterations, no bombs).

## 8. Training and promotion (`r74`)

Three streams of `scripts/train_round.py --parallel 6` (18 of the 20 threads), every run `--stop-success 0.99
--sil-coeff 0.1`, no `--promote`; continued families from their converted runs (`--load`).

| family | run | from | steps | wall | success, hearts/episode (canonical eval) |
|---|---|---|---:|---:|---|
| darknut | r74-darknut | r72-darknut (promoted) | 4.00M | 1.19 h | 93.1% −0.95 |
| darknut | r74-darknut2 | r74-darknut | 4.00M | 1.40 h | 95.5% −0.78 |
| darknut | r74-darknut3 | r74-darknut2 (+ 3 and 4 alive tracks) | 4.00M | 1.37 h | 96.8% −0.64 |
| darknut | r74-darknut4 | r74-darknut3 (+ `5_05e_0~k1`) | 4.00M | 1.13 h | 97.0% −0.62 |
| darknut | r74-darknut5 | r74-darknut4 | 4.00M | 2.08 h | 97.3% −0.55 |
| gibdo | r74-d5new | scratch | 2.60M (early stop) | 1.12 h | 99.3% −0.04 |
| pols-voice | r74-d5new | scratch | 1.64M (early stop) | 0.62 h | 99.5% −0.20 |
| boss-digdogger | r74-d5boss | scratch | 2.96M (early stop) | 1.18 h | 99.0% +0.80 (1.2 bombs an episode) |
| dodongo-crossing | r74-d5boss | scratch | 0.35M (early stop) | 0.16 h | 100% +0.00 |
| zol | r74-d5cont | r71-zol-cont | 0.24M (early stop) | 0.09 h | 99.9% +0.14 |
| equipment-room | r74-d5cont | r71-e | 0.25M (early stop) | 0.09 h | 99.9% −0.13 |
| ow-zora | r74-leg | r72-zora-off | 0.77M (early stop) | 0.32 h | 99.6% −0.12 |
| ow-moblin | r74-leg | r72-moblin-off | 4.00M | 1.49 h | 98.9% −0.14 |
| ow-peahat | r74-leg | r71-ow | 4.00M | 2.05 h | 97.0% −0.17 |
| ow-lynel | r74-lynel | r71-ow | 1.45M (early stop) | 0.60 h | 99.0% −0.13 |
| ow-tektite | r74-leg2 | r72-ow | 0.51M (early stop) | 0.16 h | 99.7% −0.00 |
| ow-leever | r74-leg2 | r72-ow | 0.95M (early stop) | 0.24 h | 100% +0.05 |
| ow-start | r74-leg2 | r72-ow | 4.00M | 1.08 h | 99.9% +0.08 |
| ow-octorok | r74-leg2 | r72-octorok | 0.27M (early stop) | 0.13 h | 99.3% −0.08 |

Stream C's loop was stopped by hand to move ow-lynel to a free stream; its orphaned `train_round` also trained
`ow-tektite/r74-leg` (an unused duplicate) and was stopped (SIGTERM) in `ow-leever/r74-leg` (interrupted, unused).
Checked in every run: the expected curriculum tracks, episodes finishing from the first iteration, the clips share
(`clips/enabled_share` 0.24-0.71 by family's `off_share`), darknut's staged kills (Olive Garden 45% unstaged, 1-5 in
the rest; 0x64 50%; 3:0x59 never staged).

**The previous run on today's states** (each continued family's promoted run re-evaluated; success and hearts an
episode over its old rooms → over the new rooms): every continued family is equal or better on its old rooms and
far better on the new ones (darknut old run 53.9% −2.60 on the new rooms, 0.0% −8.04 in the Olive Garden; ow-leever
0.2% on its new rooms; ow-peahat 28.1%, ow-lynel 48.0%). One old room is more than 2 points worse: ow-moblin `0_3ds`
with clips off, 94.3% −0.26 against 97.7% −0.23 (300 episodes, about 1.4 standard errors), accepted.

**Rooms below 90% in single rooms:** the Olive Garden (80.3 | 80.7% clips on | off, r74-darknut5), 0x64's plain way
(69.7%), ow-peahat `0_1ds_2` with clips off (0%: a state the plain way can't finish, §14), ow-zora `0_55n` with clips
on 94.3% (16 wrong exits in 300).

**Promoted** (act `temperature:0.5+relax:16`; the run it replaced in brackets): gibdo, pols-voice r74-d5new (new);
dodongo-crossing, boss-digdogger r74-d5boss (new); zol r74-d5cont (r71-zol-cont); equipment-room r74-d5cont (r71-e);
ow-zora r74-leg (r72-zora-off); ow-moblin r74-leg (r72-moblin-off); ow-peahat r74-leg (r71-ow); ow-lynel r74-lynel
(r71-ow); ow-tektite, ow-leever, ow-start r74-leg2 (r72-ow); ow-octorok r74-leg2 (r72-octorok); darknut r74-darknut4
(r72-darknut) for the chains, then r74-darknut5 (§9). The other 10 families keep their converted round 21-22 runs.

## 9. The Olive Garden's curve

The Olive Garden alone (`5_05e_0..2`, all six Blue Darknuts, bombs held on entry fixed, hp drawn as in training,
clips on, 300 episodes each):

| run | 0 bombs | 2 | 4 | 8 | won at < 3.5 / 3.5-6 / ≥ 6 hearts (4 bombs) | hearts lost a fight (2 bombs) |
|---|---:|---:|---:|---:|---|---:|
| r72-darknut (round 22) | 0.0 (canonical, −8.04 hearts) | | | | | |
| r74-darknut (4M) | 39.3 | 44.7 | 50.0 | 58.7 | 6 / 23 / 61 | 6.09 |
| r74-darknut2 (8M) | 55.3 | 65.7 | 69.0 | 78.3 | 12 / 40 / 83 | 4.63 |
| r74-darknut3 (12M) | 67.3 | 72.0 | 78.7 | 80.0 | 28 / 51 / 91 | 4.04 |
| r74-darknut4 (16M) | 72.3 | 78.3 | 81.7 | 82.0 | 19 / 70 / 93 | 3.63 |
| r74-darknut5 (20M) | 73.0 | 81.3 | 82.3 | 87.0 | 28 / 56 / 95 | 3.04 |

Pooled over the four bomb counts (1200 episodes): r74-darknut4 78.6 ± 1.2%, −3.54 ± 0.08 hearts; **r74-darknut5
80.9 ± 1.1%, −2.92 ± 0.08 hearts**: success within noise, 0.6 hearts a fight less lost (about 5 standard errors),
and no worse anywhere else (all states 97.3% −0.55 against 97.0% −0.62; old rooms 99.5% −0.20 against −0.21; 0x64's
plain way 69.7% against 67.0%). Promoted (decisions 2026-11-15): by the comparison rule (success, then hearts) and
because the hearts Link carries out of the Garden are what the way back lacks (§11).

**The Dungeon 5 segment under it** (`r74b-1k`, `r74b-1k-noclips`: `SEGMENTS="d5:5_76s_0"`, 1000 runs each, the
rest of the team unchanged) against the same segment with r74-darknut4 (`r74-1k`, §10):

| D5 segment (from `5_76s_0`), clips on \| off | r74-darknut4 | **r74-darknut5** |
|---|---|---|
| reach the D5 triforce | 34.9 ± 1.5 \| 22.0 ± 1.3 | **46.4 ± 1.6 \| 34.0 ± 1.5** |
| 0x64's plain way (off): deaths | – \| 261 of 998 | – \| 195 of 996 |
| the Olive Garden: won | 865 of 996 (87%) \| 383 of 734 (52%) | 904 of 996 (91%) \| 504 of 800 (63%) |
| the Olive Garden: hearts lost a visit | 4.19 \| 3.02 | 3.34 \| 2.67 |
| 0x64 on the way back: hearts on entry, deaths | 5.59, 104 \| 5.01, 0 | 6.33, 68 \| 5.68, 0 |
| 0x66 W→N deaths | 140 \| 43 | 109 \| 49 |
| 0x47 / 0x37 deaths | 80 / 94 \| 29 / 32 | 62 / 94 \| 32 / 37 |
| Digdogger: fights, hearts on entry, deaths | 369, 3.36, 20 \| 228, 3.64, 8 | 492, 3.35, 28 \| 352, 3.95, 12 |

Eleven to twelve points more of the segment, far more than the 2 points in the Garden alone: the Garden is won more
often and left with 0.7 more hearts, and the way back (0x64's live Darknuts, 0x66's fight) takes fewer runs. The
full route and the leg segment were not measured again with r74-darknut5; the Garden is the first Dungeon 5 room
their runs fight in, so they likely gain too (an inference, not a measurement).

## 10. Chains (1000 runs per segment; clips on | off; round 23 alongside)

`RUNS=1000 ACT=model TAG=r74-1k scripts/experiments/r56_chains.sh` and `CLIPS=off TAG=r74-1k-noclips`; team as
promoted at 02:34 Nov 15 (darknut r74-darknut4). Every segment now plays on to the D5 triforce. Round 23 (`r73-1k`,
`r73-1k-noclips`, route ending on the D4 triforce) ran D2, D3, D4 and the full route only.

Arrivals at each segment's end (± one standard error); round 23's D4 segment ended on the D4 triforce, round 24's
on the leg's first room 0:0x45 (the triforce's warp):

| segment (from → arrives at) | `r74-1k` (on) | `r74-1k-noclips` (off) | `r73-1k` (on) | `r73-1k-noclips` (off) |
|---|---:|---:|---:|---:|
| d2 (`2_7ds_0` → 0:0x3c) | 89.9 ± 1.0 | 89.9 ± 1.0 | 89.9 ± 1.0 | 89.9 ± 1.0 |
| post-d2 (`0_3cc_0` → 1:0x73) | 95.2 ± 0.7 | 92.3 ± 0.8 | – | – |
| d1 (`1_73s_0` → 0:0x37) | 99.7 ± 0.2 | 99.4 ± 0.2 | – | – |
| post-d1 (`0_37c_0` → 3:0x7c) | 100.0 | 99.9 ± 0.1 | – | – |
| d3 (`3_7cs_0` → 0:0x74) | 67.3 ± 1.5 | 67.3 ± 1.5 | 60.1 ± 1.5 | 60.1 ± 1.5 |
| post-d3 (`0_74c_0` → 4:0x71) | 100.0 | 100.0 | – | – |
| d4 (`4_71s_0` → 0:0x45) | 81.3 ± 1.2 | 59.7 ± 1.6 | 78.6 ± 1.3 | 64.7 ± 1.5 |
| post-d4, the leg (`0_45c_0` → 5:0x76) | 99.8 ± 0.1 | 92.0 ± 0.9 | – | – |
| d5 (`5_76s_0` → the D5 triforce) | **34.9 ± 1.5** | **22.0 ± 1.3** | – | – |

From the game start (`full`, 1000 runs; round 23's route ended on the D4 triforce):

| Link reaches | `r74-1k` (on) | `r74-1k-noclips` (off) | `r73-1k` (on) | `r73-1k-noclips` (off) |
|---|---:|---:|---:|---:|
| Dungeon 1 | 80.7 ± 1.2 | 64.6 ± 1.5 | 81.4 ± 1.2 | 66.5 ± 1.5 |
| Dungeon 3 | 67.0 ± 1.5 | 47.4 ± 1.6 | 63.8 ± 1.5 | 48.1 ± 1.6 |
| the D3 triforce (and Dungeon 4) | 41.8 ± 1.6 | 27.6 ± 1.4 | 33.5 ± 1.5 | 22.7 ± 1.3 |
| the D4 triforce | 33.3 ± 1.5 | 19.2 ± 1.2 | 28.7 ± 1.4 | 15.3 ± 1.1 |
| Dungeon 5 | 33.3 ± 1.5 | 17.5 ± 1.2 | – | – |
| **the D5 triforce** | **4.8 ± 0.7** | **0.3 ± 0.2** | – | – |

Each segment's runs reaching the D5 triforce (clips on | off): D2 5.6 | 0.6%, after D2 6.5 | 0.7%, D1 8.9 | 1.0%,
after D1 8.5 | 0.6%, D3 7.6 | 0.8%, after D3 9.8 | 0.8%, D4 13.0 | 1.0%, the leg 15.2 | 2.9%, D5 34.9 | 22.0%.

- **Dungeon 3 rose 7 points** (60.1 → 67.3%, either switch state): darknut's five runs also trained 3:0x5b/0x59/
  0x69/0x5c; Manhandla (3:0x4d, unchanged model) still kills 282 of 1000 D3 runs.
- **Dungeon 4 with clips off fell 5 points** (64.7 → 59.7%, about 2.3 standard errors) with the same D4 models but
  zol: Link reaches 4:0x30 with 3.40 hearts (round 23 3.57) and Gleeok with 2.90 (3.04), and Gleeok kills 269 of
  1000 runs (236). Clips on it rose (78.6 → 81.3%). Not explained; in the backlog.
- **Where full runs end**, clips on: 3:0x4d (Manhandla) 185, the Olive Garden 150, Gleeok 70, 2:0x1e 48, 2:0x5e 44,
  5:0x64 on the way back 37, 1:0x23 32, 5:0x66 W→N 30; clips off: 5:0x64's plain way 135, Manhandla 133, 2:0x5e 97,
  2:0x1e 53, 2:0x3e 51, Gleeok 48.

The yardstick report (`r56_chains.sh report`) refused the tag: the ten families kept on round 21-22 runs carry the
`eval.json` of their source checkpoints (the conversion re-binds `act` to the widened sha1 but keeps the source's
evaluation), so it found no evaluation of the current weights. The tables above are read from the chain files
(`RoomRecord`s); `room-stats r74-1k --noclips` ran (`r74-1k-both-room-stats.json`). In the backlog.

## 11. What loses runs

**The Dungeon 5 segment** (from `5_76s_0`: 10 hearts, 4 bombs, chain 0; 1000 runs; each cell clips on \| off):

| step | room | entry → exit | visits | deaths | wrong exits | hearts lost a visit | frames a visit | clips a visit | bombs placed (cheated) |
|---:|---|---|---|---|---|---|---|---|---|
| 186 | 5:0x76 | S → N | 1000 \| 1000 | 0 \| 0 | 0 \| 0 | 0.00 \| 0.00 | 307 \| 307 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 187 | 5:0x66 | S → W | 1000 \| 1000 | 0 \| 0 | 0 \| 0 | 0.09 \| 0.09 | 560 \| 560 | 0.00 \| 0.00 | 1000 (0) \| 1000 (0) |
| 188 | 5:0x65 | E → W | 1000 \| 1000 | 4 \| 4 | 0 \| 0 | 0.68 \| 0.68 | 782 \| 782 | 0.00 \| 0.00 | 1000 (0) \| 1000 (0) |
| 189 | 5:0x64 | E → cave | 996 \| 998 | 0 \| 261 | 0 \| 2 | 0.00 \| 5.15 | 205 \| 1614 | 1.00 \| 0.00 | 0 (0) \| 2 (0) |
| 190 | 5:0x07 cave | S → E | 996 \| 735 | 0 \| 1 | 0 \| 0 | 0.05 \| 0.08 | 440 \| 432 | 0.00 \| 0.00 | 0 (0) \| 1 (0) |
| 191 | 5:0x06 | cave → W | 996 \| 734 | 0 \| 0 | 0 \| 0 | 0.00 \| 0.00 | 285 \| 285 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 192 | 5:0x05 Olive Garden | E → cave | 996 \| 734 | 131 \| 351 | 0 \| 0 | 4.19 \| 3.02 | 1907 \| 1616 | 0.00 \| 0.00 | 2594 (0) \| 1779 (0) |
| 193 | 5:0x04 cave | S → N | 865 \| 383 | 0 \| 1 | 0 \| 0 | 0.14 \| 0.13 | 813 \| 811 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 194 | 5:0x05 | cave → E | 865 \| 382 | 0 \| 0 | 0 \| 0 | 0.00 \| 0.00 | 256 \| 256 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 195 | 5:0x06 | W → cave | 865 \| 382 | 0 \| 0 | 0 \| 0 | 0.00 \| 0.00 | 262 \| 262 | 0.01 \| 0.00 | 0 (0) \| 0 (0) |
| 196 | 5:0x07 cave | S → W | 865 \| 382 | 2 \| 2 | 0 \| 0 | 0.12 \| 0.11 | 416 \| 414 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 197 | 5:0x64 | cave → E | 863 \| 380 | 104 \| 0 | 0 \| 0 | 1.60 \| 0.00 | 360 \| 366 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 198 | 5:0x65 | W → E | 759 \| 381 | 39 \| 32 | 2 \| 1 | 0.42 \| 0.72 | 487 \| 486 | 0.00 \| 0.00 | 3 (0) \| 7 (0) |
| 199 | 5:0x66 | W → N | 722 \| 350 | 140 \| 43 | 5 \| 3 | 1.08 \| 0.83 | 978 \| 1160 | 0.00 \| 0.00 | 107 (0) \| 126 (0) |
| 200 | 5:0x56 | S → E | 577 \| 304 | 0 \| 0 | 0 \| 0 | 0.00 \| 0.00 | 376 \| 375 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 201 | 5:0x57 | W → N | 577 \| 304 | 7 \| 1 | 0 \| 0 | 0.06 \| 0.03 | 385 \| 377 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 202 | 5:0x47 | S → N | 570 \| 303 | 80 \| 29 | 0 \| 0 | 0.82 \| 0.69 | 333 \| 346 | 0.00 \| 0.00 | 27 (0) \| 24 (0) |
| 203 | 5:0x37 | S → N | 490 \| 274 | 94 \| 32 | 0 \| 0 | 1.16 \| 0.95 | 251 \| 266 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 204 | 5:0x27 | S → W | 396 \| 242 | 12 \| 5 | 0 \| 0 | 0.04 \| 0.10 | 377 \| 383 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 205 | 5:0x26 | E → W | 384 \| 237 | 4 \| 2 | 0 \| 0 | 0.05 \| 0.08 | 385 \| 387 | 0.00 \| 0.00 | 0 (0) \| 1 (0) |
| 206 | 5:0x25 | E → W | 380 \| 235 | 11 \| 7 | 0 \| 0 | 0.15 \| 0.19 | 456 \| 458 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 207 | 5:0x24 Digdogger | E → N | 369 \| 228 | 20 \| 8 | 0 \| 0 | 0.07 \| 0.07 | 1010 \| 1003 | 0.00 \| 0.00 | 346 (0) \| 205 (0) |
| 208 | 5:0x14 | S → triforce | 349 \| 220 | 0 \| 0 | 0 \| 0 | 0.00 \| 0.00 | 921 \| 909 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |

**Arrivals in the D5 segment** (hearts; bombs followed through the room records: + cheated, − placed, + 4 a pickup,
capped at 8):

| | clips on | clips off |
|---|---|---|
| the Olive Garden: hearts (mean; < 3.5 / 3.5-6 / ≥ 6) | 9.22 (4 / 26 / 966) | 5.53 (168 / 213 / 353) |
| the Olive Garden: bombs held (won) | 2: 843 (731 won, 87%); 6 (took 0x65's): 153 (134, 88%) | 2: 630 (332, 53%); 6: 102 (49, 48%) |
| 0x64 on the way back: hearts | 5.59 (187 / 284 / 392) | 5.01 (103 / 134 / 143) |
| Digdogger: hearts | 3.36 (208 / 112 / 49) | 3.64 (115 / 71 / 42) |
| Digdogger: bombs held | 0: 57, 1-3: 32, 4: 182, 5-7: 19, 8: 79 (of 369) | 0: 38, 1-3: 30, 4: 126, 5-7: 6, 8: 28 (of 228) |

**The leg and Dungeon 5 from the D4 triforce** (`post-d4`, from `0_45c_0`: 8 hearts, 0 bombs; 1000 runs; each cell
clips on \| off; `way` is the stretch way a step is on):

| step | room | entry → exit | way | visits | deaths | wrong exits | hearts lost a visit | frames a visit | clips a visit | bombs placed (cheated) |
|---:|---|---|---|---|---|---|---|---|---|---|
| 134 | 0:0x45 | cave → S | – | 1000 \| 1000 | 0 \| 0 | 0 \| 0 | 0.00 \| 0.00 | 178 \| 178 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 135 | 0:0x55 | N → W | clips | 1024 \| – | 0 \| – | 91 \| – | 0.12 \| – | 314 \| – | 0.93 \| – | 0 (0) \| – |
| 136 | 0:0x54 | E → W | clips | 933 \| – | 0 \| – | 0 \| – | 0.00 \| – | 141 \| – | 1.00 \| – | 0 (0) \| – |
| 137 | 0:0x53 | E → W | clips | 933 \| – | 0 \| – | 0 \| – | 0.25 \| – | 356 \| – | 0.00 \| – | 0 (0) \| – |
| 138 | 0:0x52 | E → W | clips | 933 \| – | 0 \| – | 0 \| – | 0.32 \| – | 346 \| – | 0.00 \| – | 0 (0) \| – |
| 139 | 0:0x51 | E → W | clips | 933 \| – | 0 \| – | 0 \| – | 0.00 \| – | 140 \| – | 1.00 \| – | 0 (0) \| – |
| 140 | 0:0x50 | E → S | clips | 933 \| – | 0 \| – | 0 \| – | 0.07 \| – | 205 \| – | 0.00 \| – | 0 (0) \| – |
| 141 | 0:0x60 | N → W | clips | 933 \| – | 0 \| – | 0 \| – | 0.00 \| – | 205 \| – | 1.00 \| – | 0 (0) \| – |
| 142 | 0:0x5f | E → N | clips | 933 \| – | 0 \| – | 0 \| – | 0.01 \| – | 253 \| – | 0.98 \| – | 0 (0) \| – |
| 143 | 0:0x55 | N → E | plain | – \| 1026 | – \| 0 | – \| 26 | – \| 0.15 | – \| 295 | – \| 0.00 | – \| 0 (0) |
| 144 | 0:0x56 | W → E | plain | – \| 1000 | – \| 0 | – \| 0 | – \| 0.28 | – \| 339 | – \| 0.00 | – \| 0 (0) |
| 145 | 0:0x57 | W → E | plain | – \| 1000 | – \| 0 | – \| 0 | – \| 0.14 | – \| 328 | – \| 0.00 | – \| 0 (0) |
| 146 | 0:0x58 | W → E | plain | – \| 1000 | – \| 0 | – \| 0 | – \| 0.09 | – \| 307 | – \| 0.00 | – \| 0 (0) |
| 147 | 0:0x59 | W → E | plain | – \| 1000 | – \| 0 | – \| 0 | – \| 0.29 | – \| 331 | – \| 0.00 | – \| 0 (0) |
| 148 | 0:0x5a | W → E | plain | – \| 1000 | – \| 0 | – \| 0 | – \| 0.10 | – \| 337 | – \| 0.00 | – \| 0 (0) |
| 149 | 0:0x5b | W → E | plain | – \| 1000 | – \| 0 | – \| 0 | – \| 0.17 | – \| 332 | – \| 0.00 | – \| 0 (0) |
| 150 | 0:0x5c | W → E | plain | – \| 1000 | – \| 0 | – \| 0 | – \| 0.29 | – \| 341 | – \| 0.00 | – \| 0 (0) |
| 151 | 0:0x5d | W → S | plain | – \| 1000 | – \| 0 | – \| 3 | – \| 0.44 | – \| 357 | – \| 0.00 | – \| 0 (0) |
| 152 | 0:0x6d | N → E | plain | – \| 997 | – \| 0 | – \| 0 | – \| 0.04 | – \| 196 | – \| 0.00 | – \| 0 (0) |
| 153 | 0:0x6e | W → E | plain | – \| 997 | – \| 0 | – \| 0 | – \| 0.84 | – \| 375 | – \| 0.00 | – \| 0 (0) |
| 154 | 0:0x6f | W → N | plain | – \| 997 | – \| 0 | – \| 6 | – \| 0.09 | – \| 234 | – \| 0.00 | – \| 0 (0) |
| 155 | 0:0x5f | S → N | plain | – \| 991 | – \| 2 | – \| 0 | – \| 0.21 | – \| 477 | – \| 0.00 | – \| 0 (0) |
| 156 | 0:0x4f | S → N | – | 933 \| 989 | 0 \| 0 | 0 \| 0 | 0.18 \| 0.16 | 246 \| 245 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 157 | 0:0x3f | S → N | – | 933 \| 989 | 0 \| 0 | 0 \| 0 | 0.20 \| 0.20 | 295 \| 297 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 158 | 0:0x2f | S → cave | – | 937 \| 1001 | 0 \| 0 | 0 \| 0 | 0.00 \| 0.00 | 334 \| 333 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 159 | 0:0x2f cave | S → S | – | 937 \| 1001 | 0 \| 0 | 0 \| 0 | 0.00 \| 0.00 | 382 \| 378 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 160 | 0:0x2f | cave → S | – | 937 \| 1001 | 0 \| 0 | 0 \| 0 | 0.00 \| 0.00 | 178 \| 178 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 161 | 0:0x3f | N → W | – | 937 \| 1001 | 0 \| 0 | 4 \| 12 | 0.04 \| 0.07 | 257 \| 254 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 162 | 0:0x3e | E → W | clips | 933 \| – | 0 \| – | 0 \| – | 0.00 \| – | 140 \| – | 1.00 \| – | 0 (0) \| – |
| 163 | 0:0x3d | E → N | clips | 933 \| – | 0 \| – | 0 \| – | 0.03 \| – | 245 \| – | 0.00 \| – | 0 (0) \| – |
| 164 | 0:0x2d | S → N | clips | 998 \| – | 0 \| – | 0 \| – | 0.63 \| – | 255 \| – | 0.00 \| – | 0 (0) \| – |
| 165 | 0:0x3e | E → N | plain | – \| 989 | – \| 0 | – \| 0 | – \| 0.34 | – \| 282 | – \| 0.00 | – \| 0 (0) |
| 166 | 0:0x2e | S → W | plain | – \| 989 | – \| 0 | – \| 0 | – \| 0.01 | – \| 237 | – \| 0.00 | – \| 0 (0) |
| 167 | 0:0x2d | E → N | plain | – \| 989 | – \| 1 | – \| 0 | – \| 0.57 | – \| 338 | – \| 0.00 | – \| 0 (0) |
| 168 | 0:0x1d | S → E | – | 998 \| 988 | 0 \| 0 | 0 \| 0 | 0.14 \| 0.05 | 306 \| 310 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 169 | 0:0x1e | W → E | – | 998 \| 989 | 0 \| 1 | 0 \| 0 | 0.00 \| 0.40 | 142 \| 320 | 1.00 \| 0.00 | 0 (0) \| 0 (0) |
| 170 | 0:0x1f | W → N | – | 998 \| 988 | 0 \| 16 | 0 \| 0 | 1.02 \| 0.96 | 321 \| 315 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 171 | 0:0x0f | S → cave | – | 998 \| 972 | 0 \| 0 | 0 \| 0 | 0.00 \| 0.00 | 358 \| 358 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 172 | 0:0x0f cave | S → S | – | 998 \| 972 | 0 \| 0 | 0 \| 0 | 0.00 \| 0.00 | 460 \| 442 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 173 | 0:0x0f | cave → S | – | 998 \| 972 | 0 \| 0 | 0 \| 0 | 0.00 \| 0.00 | 178 \| 178 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 174 | 0:0x1f | N → W | – | 998 \| 972 | 0 \| 0 | 0 \| 0 | 0.00 \| 0.00 | 226 \| 226 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 175 | 0:0x1e | E → W | – | 998 \| 972 | 0 \| 18 | 0 \| 0 | 0.00 \| 0.39 | 185 \| 349 | 1.01 \| 0.00 | 0 (0) \| 0 (0) |
| 176 | 0:0x1d | E → N | clips | 998 \| – | 0 \| – | 0 \| – | 0.00 \| – | 214 \| – | 0.00 \| – | 0 (0) \| – |
| 177 | 0:0x0d | S → W | clips | 998 \| – | 0 \| – | 0 \| – | 0.27 \| – | 371 \| – | 0.00 \| – | 0 (0) \| – |
| 178 | 0:0x0c | E → W | clips | 998 \| – | 0 \| – | 0 \| – | 0.00 \| – | 140 \| – | 1.00 \| – | 0 (0) \| – |
| 179 | 0:0x0b | E → cave | clips | 998 \| – | 0 \| – | 0 \| – | 0.01 \| – | 424 \| – | 0.00 \| – | 0 (0) \| – |
| 180 | 0:0x1d | E → S | plain | – \| 954 | – \| 0 | – \| 1 | – \| 0.00 | – \| 260 | – \| 0.00 | – \| 0 (0) |
| 181 | 0:0x2d | N → W | plain | – \| 953 | – \| 12 | – \| 0 | – \| 0.22 | – \| 266 | – \| 0.00 | – \| 0 (0) |
| 182 | 0:0x2c | E → N | plain | – \| 941 | – \| 16 | – \| 8 | – \| 0.15 | – \| 346 | – \| 0.00 | – \| 0 (0) |
| 183 | 0:0x1c | S → W | plain | – \| 925 | – \| 0 | – \| 0 | – \| 0.00 | – \| 229 | – \| 0.00 | – \| 0 (0) |
| 184 | 0:0x1b | E → maze | plain | – \| 925 | – \| 0 | – \| 0 | – \| 0.00 | – \| 1048 | – \| 0.00 | – \| 0 (0) |
| 185 | 0:0x0b | S → cave | plain | – \| 925 | – \| 5 | – \| 0 | – \| 0.02 | – \| 321 | – \| 0.00 | – \| 0 (0) |
| 186 | 5:0x76 | S → N | – | 998 \| 920 | 0 \| 0 | 0 \| 0 | 0.00 \| 0.00 | 307 \| 307 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 187 | 5:0x66 | S → W | – | 998 \| 920 | 1 \| 1 | 0 \| 0 | 0.06 \| 0.00 | 549 \| 552 | 0.00 \| 0.00 | 998 (985) \| 920 (792) |
| 188 | 5:0x65 | E → W | – | 998 \| 919 | 4 \| 104 | 0 \| 0 | 0.69 \| 0.53 | 605 \| 593 | 0.00 \| 0.00 | 995 (982) \| 872 (747) |
| 189 | 5:0x64 | E → cave | – | 997 \| 815 | 65 \| 639 | 4 \| 0 | 0.32 \| 2.27 | 220 \| 931 | 0.93 \| 0.00 | 0 (0) \| 0 (0) |
| 190 | 5:0x07 cave | S → E | – | 928 \| 176 | 1 \| 2 | 0 \| 0 | 0.09 \| 0.07 | 426 \| 426 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 191 | 5:0x06 | cave → W | – | 927 \| 174 | 0 \| 0 | 0 \| 0 | 0.00 \| 0.00 | 284 \| 286 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 192 | 5:0x05 Olive Garden | E → cave | – | 927 \| 174 | 398 \| 124 | 0 \| 0 | 3.46 \| 2.59 | 1800 \| 1279 | 0.00 \| 0.00 | 722 (0) \| 191 (0) |
| 193 | 5:0x04 cave | S → N | – | 529 \| 50 | 7 \| 2 | 0 \| 0 | 0.19 \| 0.20 | 809 \| 801 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 194 | 5:0x05 | cave → E | – | 522 \| 48 | 0 \| 0 | 0 \| 0 | 0.00 \| 0.00 | 256 \| 256 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 195 | 5:0x06 | W → cave | – | 522 \| 48 | 0 \| 0 | 0 \| 0 | 0.00 \| 0.00 | 262 \| 262 | 0.01 \| 0.00 | 0 (0) \| 0 (0) |
| 196 | 5:0x07 cave | S → W | – | 522 \| 48 | 4 \| 1 | 0 \| 0 | 0.10 \| 0.12 | 414 \| 416 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 197 | 5:0x64 | cave → E | – | 518 \| 47 | 101 \| 0 | 0 \| 0 | 1.46 \| 0.00 | 340 \| 366 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 198 | 5:0x65 | W → E | – | 417 \| 47 | 46 \| 4 | 1 \| 0 | 0.48 \| 0.81 | 472 \| 442 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 199 | 5:0x66 | W → N | – | 371 \| 43 | 81 \| 6 | 2 \| 0 | 0.96 \| 0.65 | 1004 \| 1013 | 0.00 \| 0.00 | 66 (0) \| 14 (0) |
| 200 | 5:0x56 | S → E | – | 288 \| 37 | 0 \| 0 | 0 \| 0 | 0.00 \| 0.00 | 377 \| 374 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 201 | 5:0x57 | W → N | – | 288 \| 37 | 0 \| 0 | 0 \| 0 | 0.02 \| 0.05 | 381 \| 416 | 0.00 \| 0.00 | 0 (0) \| 1 (0) |
| 202 | 5:0x47 | S → N | – | 288 \| 37 | 56 \| 1 | 0 \| 0 | 0.85 \| 0.58 | 307 \| 368 | 0.00 \| 0.00 | 2 (0) \| 2 (0) |
| 203 | 5:0x37 | S → N | – | 232 \| 36 | 51 \| 4 | 0 \| 0 | 1.14 \| 1.17 | 245 \| 275 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 204 | 5:0x27 | S → W | – | 181 \| 32 | 4 \| 0 | 0 \| 0 | 0.01 \| 0.00 | 379 \| 381 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 205 | 5:0x26 | E → W | – | 177 \| 32 | 3 \| 1 | 0 \| 0 | 0.07 \| 0.12 | 385 \| 380 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 206 | 5:0x25 | E → W | – | 174 \| 31 | 15 \| 1 | 0 \| 0 | 0.18 \| 0.39 | 449 \| 460 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |
| 207 | 5:0x24 Digdogger | E → N | – | 159 \| 30 | 7 \| 1 | 0 \| 0 | 0.05 \| 0.05 | 1002 \| 1023 | 0.00 \| 0.00 | 135 (0) \| 30 (0) |
| 208 | 5:0x14 | S → triforce | – | 152 \| 29 | 0 \| 0 | 0 \| 0 | 0.00 \| 0.00 | 946 \| 925 | 0.00 \| 0.00 | 0 (0) \| 0 (0) |

Off the route (clips on): 65 runs took the coast's plain way after a wrong exit out of 0x55 (0x56 .. 0x5d, 0x4d,
0x3d) and rejoined; 2 strayed back into Dungeon 4 (one stuck in 4:0x71, one dead in 4:0x50).

**Arrivals from the D4 triforce:** at Dungeon 5 6.57 hearts with clips on (7 under 3.5) and 3.10 with clips off (518
of 920 under 3.5); at the Olive Garden 5.85 | 3.70 hearts, with 0 bombs in 736 of 927 arrivals with clips on (won
412, 56%) and 4 (0x65's, taken westbound) in 178 (110, 62%); at Digdogger 2.69 | 3.35 hearts.

**What loses runs, in order:**

1. **The Olive Garden.** From Dungeon 5's own entrance (2 bombs, 9.2 hearts) it is won 87% of the time with clips on,
   close to the single-room 80% at every hp; from the D4 triforce Link brings 0 bombs (both walls cheated) and 5.9
   hearts and wins 57%. With clips off he comes from the plain 0x64 with 5.5 hearts and wins 52%.
2. **The way back at low hearts** (clips on, per 1000 D5 runs): 0x66 W→N 140 (the forced fight; 1.08 hearts a visit),
   0x64 cave→E 104 (the Darknuts the clip left alive), 0x37 94, 0x47 80, 0x65 W→E 39. Single rooms win these 99-100%:
   the arrivals are below training's hp floor (1.5 hearts at 10 containers); the probe saw 0x37's deaths as one hit
   at 0.5-1 heart within about 40 frames of entry. With clips off the Darknuts of 0x64 are dead on the way back (0
   deaths there).
3. **0x64's plain way** (clips off): 261 of 998 D5 visits die (5.15 hearts a visit); from the D4 triforce 639 of 815.
4. **The leg's plain ways** cost 3.4 more hearts than the clip way (Dungeon 5 reached with 3.1 against 6.6), and 0x65
   westbound then kills 104 of 919 (clips off) against 4.
5. **Digdogger** is not the problem: 20 deaths in 369 fights (clips on), 0.07 hearts lost a visit, the recorder
   played every visit, 0.94 bombs a fight; Link arrives with 3.4 hearts and mostly 4 or 8 bombs.
6. 0x64 with clips on from the D4 triforce: 65 deaths in 997 visits (none in the D5 segment): 7% of those visits
   don't clip (0.93 clips a visit) and fight with 6.6 hearts or less.

## 12. The played states after the Olive Garden

The 48 states after the Olive Garden (`5_04cave` .. `5_14s`, chains 0-2) were built with its fight skipped (§7) and
are now played:

1. **Harvest:** 40 chain runs a chain from `5_76s_<c>` (`eval-chain --stop-at step:194 --harvest`, darknut
   r74-darknut3) gave 99 harvested `5_04cave` states (Link climbing out of the Garden's cellar with the recorder).
   He left the Garden with a median of 5 / 6 / 6 hearts (chains 0 / 1 / 2).
2. **Sources:** one a chain, about the median: chain 0 seed 2 (5 hearts), chain 1 seed 11 (6), chain 2 seed 17 (7;
   the median run, seed 10, failed the build at 0x66 W→N's fight: 10 attempts, 596k env steps).
3. **Build:** `entry_states build --route d2first --start-step 193 --source <harvest> --chains <c>`: 15 states a
   chain, every crossing found; staged with the three sources as `5_04cave_<c>`.
4. **Hearts:** the three `5_04cave_<c>` kept the harvest's true hearts (5 / 6 / 7 of 10), which the audit refuses
   (HP); they were written again the way every entry state is (`entry_states.write_entry_state`: `refill_hearts`,
   `make_snapshot_fresh`, the load check; `MapLocation` from `triforce.zelda_enums`), now 10 of 10. The other 45
   were written that way by the build.
5. **Install** with nothing running (`entry_states install`): the 48 manifest entries are the played builds' (no
   `fight_skipped` left); audit 613 states, 0 problems, 0 directory problems.
6. **Curricula** explored again on the 48 states (3000 iterations, patience 300): every state solved in darknut
   (12), equipment-room (6), dodongo-crossing, zol, boss-digdogger (3 each), pols-voice (6), gibdo 12 of 15; gibdo's
   `5_66w_*` (0x66 on the way back) needed 8000 iterations with seed 1, as before: solved in 281 / 236 / 244 steps
   (4309-6009 iterations). Track lengths: 0x05 back 47-71 steps, 0x06 26, 0x64 back 51-73, 0x37 12-19, 0x04's
   cellar 121-132, the tunnel's second visit 55-68, 0x65 W 56-82, 0x47 41-113, 0x26 25-85, 0x14 2, 0x56 20-44,
   0x57 37-51, 0x27 63-130, 0x25 93-137, Digdogger 100-122.

**The promoted runs on the played states against the made ones** (single rooms: the 16 entries, 100 episodes a
state, seed 0, the standing act, hp drawn as in training, `--clips both` for darknut; made = the replaced states,
evaluated from a copy of the old states directory):

| family | entry | played: success, hearts an episode | made: success, hearts an episode |
|---|---|---|---|
| darknut r74-darknut5 | 5_05c | 100.0% +0.00 | 100.0% +0.00 |
| | 5_06w | 100.0% +0.00 | 100.0% +0.00 |
| | 5_37s | 99.7% −0.69 | 99.8% −0.65 |
| | 5_64c | **96.7% −1.64** | 97.7% −1.08 |
| equipment-room | 5_04cave | 100.0% −0.13 | 100.0% −0.12 |
| | 5_07cave_v2 | 100.0% −0.10 | 100.0% −0.10 |
| gibdo | 5_65w | 98.0% −0.67 | 98.7% −0.81 |
| | 5_66w | **95.0% −1.14** | 97.7% −0.55 |
| | 5_47s | 98.7% −0.58 | 99.3% −0.71 |
| | 5_26e | 99.3% −0.11 | 99.7% −0.15 |
| | 5_14s | 100.0% +2.37 | 100.0% +2.37 |
| dodongo-crossing | 5_56s | 100.0% +0.00 | 100.0% +0.00 |
| zol | 5_57w | 100.0% −0.06 | 100.0% −0.07 |
| pols-voice | 5_27s | 100.0% −0.09 | 100.0% −0.13 |
| | 5_25e | 99.0% −0.44 | 99.0% −0.27 |
| boss-digdogger | 5_24e | 99.7% +0.59 | 99.0% +0.80 |
| **all** | | darknut 99.1% −0.58, gibdo 98.2% −0.02, pols-voice 99.5% −0.27, boss-digdogger 99.7% +0.59, the rest 100% | 99.4% −0.43, 99.1% +0.03, 99.5% −0.20, 99.0% +0.80, 100% |

The empty crossings (0x05 back, 0x06, 0x14, 0x56, 0x57, the cellars) play the same from either state: the room
and Link's way in are the same, and the run's natural RNG is drawn from the seed. The rooms with a fight differ:
0x66 W→N and 0x64 on the way back cost about 0.6 hearts more from the played states, which carry the rest of a real
run's state (what Link holds, the dungeon's room history, the kill streak); why these two cost more was not
investigated. No family was retrained on them this round (the next round trains on them).

## 13. Version 1.5.0.0

MINOR 5: the route completes five dungeons. It carries round 24's model-incompatible change (the action kinds,
1.4.2.0), so COMPAT and REV are 0. `VERSION`, `pyproject.toml` and the Cargo workspace (1.5.0) moved;
`tests/test_convert.py` matches the converter's refusal against the code's version. The 28 promoted runs were
stamped 1.4.2.0 with the layout 1.5.0.0 has and are stamped 1.5.0.0 (`triforce stamp --force`: the stamp refuses to
overwrite another version without it). A 20-run chain `--from 5_76s_0` (eval-batch `r74-v150-check`) loads and plays
them on the rebuilt module: 8 of 20 reach the D5 triforce.

## 14. Problems and next steps

Next, in the order `backlog.md` "Next projects" keeps (the owner's hearts item first):

1. **Leave rooms with more hearts** (the owner, Nov 14): Dungeon 5 shows it again; most of its deaths after the
   Garden come at 0.5-3 hearts in rooms the single-room models win 99-100%.
2. **The way back after the Olive Garden at low hearts:** train the post-Garden rooms (0x64 back, 0x65 E, 0x66 W→N,
   0x47, 0x37) below today's hp floor (1.5 hearts at 10 containers), from harvested starts with their real hearts
   (the played states now exist), or with a chain hp profile.
3. **The Olive Garden:** 81% (single rooms) and still gaining; tracks from the models' own clears, since the explorer
   never cleared all six Darknuts; the chain's arrivals (0 bombs, 5.9 hearts from the D4 triforce) in training.
4. **0x65's westbound bombs:** taken in 16% of runs despite `treasure: refuse`.
5. **0x37's column:** X `$78` in play, `$70` per the owner and the scout's measurement.
6. **The plain 0x64** (clips off): 67-70% in single rooms, 195-261 of 1000 D5 runs.

Problems found (all in `backlog.md` "Left open by round 24"):

- **ow-peahat `0_1ds_2` with clips off can't be finished** (0 of 100): chain 2's clip way enters 0x1d at X `$20`, a
  pocket only a CLIP leaves, and the step (0x1d S→E) is shared by both ways, so the switch is drawn there. Fixing it
  on for that one state is not a YAML or data change: the switch comes from the route step
  (`objectives::start_clips`), so it needs a per-state override in code, or the state rebuilt from the plain way.
- **The yardstick refuses converted runs** (their `eval.json` is of the source checkpoint).
- **Dungeon 4 with clips off fell 5 points** (64.7 → 59.7%) on unchanged D4 models; not explained.
- **ow-zora `0_55n` with clips on** (94.3%, wrong exits), **ow-moblin `0_3ds` with clips off** (94.3% against 97.7%).
- **Dungeon 5's chains 1 and 2** descend from chain 0's Dungeon 4 history (0 bombs at the entrance).
- **The full route and the leg** were not measured with r74-darknut5.
