# Round 22: natural RNG, the clips switch, the new route (1.4.1.1)

The owner's list for round 22 (Nov 13, verbatim in `backlog.md` "Done (Nov 14): round 22"): fix the RNG so every
start plays on a state the game can hold, and record it in the replay; a clips switch he can flip at any moment
("ok rest of the run no clips"), models trained with clips on and with CLIP masked, room statistics with and without
clips, the clip counters fixed; the wrap into a wrong exit; after Dungeon 1 a new way to Dungeon 3 by 0x67's bomb
wall and rupees, then by a screen wrap through 0x75 with clips or along row 6 without; in Dungeon 3 0x5b (bomb) →
0x5c → 0x4c, skipping 0x4b; in Dungeon 4 the 0x21 bombs fixed and a detour from 0x11 for a key; bomb fishing in
4:0x12 as an aside, not a directive. Built on branches `r22-rng`, `r22` (the switch), `r22-wx` (the wrap exit) and
`r22-route`, merged into `main` as `bfef4b3` (1.4.1.1), then one retrain (`r72`) measured with 1000-run chains
with clips on and off. Decisions: `docs/decisions.md` 2026-11-13 ("Natural RNG at every start", "The run-time
clips switch and route stretches", "The wrap exit", "Clips in training, evaluation and watching", "Round 22 route",
"Round 22 merged") and 2026-11-14 ("The r72 retrain").

## Summary

Built (version 1.4.1.1, one REV over round 21's 1.4.1.0; observation, actions and network unchanged, so round 21's
models load and could be continued):

- **Natural RNG** (§1): every start from a saved state writes the RNG register as a state of the game's own cycle
  (a phase 88..=32854 after reset), before any frame runs; the edge spawn cell and the fresh-room writes are values
  the game can hold. Replays keep the bytes as frame-0 edits and now the phase.
- **The clips switch** (§2): `clips_allowed`, flipped at any decision; the route holds both ways through every
  stretch where they differ (`d3-approach`, `d4-island`) and re-plans from where Link stands. Clip families train
  half their episodes with CLIP masked (`clips: {enabled, off_share}`), evaluate both (`--clips both`), and chains,
  room statistics, `play-zelda.py` (`--clips`, key `c`, a `--live` cell) and `zelda-headless.py` take the switch.
  The clip counters and the wrap into a wrong exit are fixed.
- **The wrap exit** (§3): `CLIP E` in 0x75 takes Link out through its solid W edge into 0x74, on the ten safe rows
  only (four rows are a softlock).
- **The route** (§4): after Dungeon 1 S to 0x67 (bomb the wall, 30 rupees), then with clips 0x77 → 0x76 → 0x75
  (wrap exit) → 0x74, without clips 0x66 → 0x65 → 0x64 → 0x63 → 0x73 → 0x74; Dungeon 3 0x5b E (bomb) → 0x5c → 0x4c
  from S, 0x4b off the route, `bomb_reserve: 6`; Dungeon 4 0x11 N (bomb) → 0x01 (a key) → 0x02 → 0x12 from N.
- **The BOMB mask for wall families** (§5): a `required` family is offered BOMB in a bomb-wall room only where the
  bomb opens the wall the objective leaves through.
- **States and curricula** rebuilt from 0x58 to the D4 triforce, both ways; two explorer fixes (§6).

**The retrain (`r72`, §7):** 12 families retrained and promoted (ow-octorok from scratch, the rest continued from
round 21), 12 kept on their round 21 runs after a re-evaluation on the new states and RNG (within noise). 17 runs,
36.4M env steps, 10.7 h of training. Every new task is learned in single rooms (99-100% but 3:0x5b's E wall 97%);
both switch states are at or above 98% in every family, but two rooms stay weaker without clips: ow-moblin's 0x5b
W→E (91.7% against 99.3%) and ow-zora's 0x5a (96.3% against 100%).

Chains, 1000 runs per segment, clips on (`r72-1k`) | clips off (`r72-1k-noclips`); round 21 (`r71-1k-dks`) and
round 20 (`r70-1k`) in brackets (§8):

- D2 87.2% | 87.2% (85.6%, 68.6%), after D2 91.0% | 89.5% (88.6%), D1 98.9% | 99.5% (99.1%), after D1 99.9% |
  98.9% (99.4%), D3 50.4% | 50.4% (46.5%, 38.6%), after D3 100% | 100% (100%), **D4 73.6%** | 56.8% (56.4%,
  21.1%).
- From the start: to Dungeon 3 59.5% | 41.0% (56.7%), to the D3 triforce 19.5% | 10.2% (24.8%), **to the D4
  triforce 13.6% | 4.8%** (12.8%, 1.8%).

Within noise at 1000 runs (two standard errors of a difference between two tags: about 4.5 points near 50%, 3 near
13%, under 1 near 99%): D2, after D2, D3 and the full route with clips on against round 21. Beyond it: D4 with clips
on (+17.2 points), the full route to the D3 triforce (−5.3 points), and with clips off D4 and every full-route
arrival.

The findings that matter:

1. **Dungeon 4 with clips is the round's gain** (56.4% → 73.6%): Link reaches Gleeok with 3.27 hearts (2.68),
   Gleeok kills 192 runs per 1000 (245) and 4:0x12, now entered from N, 27 (95). The models were changed at the same
   time as the route, so none of this is an isolated measurement.
2. **The full route loses it again in Dungeon 3.** 3:0x5b's last pass (darknut, W → E through the new bomb wall)
   kills 145 of 1000 full runs (round 21's W → N pass: 39). With 6 or more bombs the route skips the room's
   treasure (the step is only "bomb the E wall and leave"): 41% of those visits die in the D3 segment, against 5-14%
   with 1-5 bombs, when Link kills the Darknuts and takes the treasure first. In the full route Link brings real
   bombs, so the case is common, and the full route reaches the D3 triforce in 19.5% of runs against 24.8% (§8, §9).
3. **Without clips the full route falls to 4.8%.** The plain overworld before Dungeon 2 costs hearts (Link reaches
   Dungeon 2 with 2.24 hearts against 2.63) and 2:0x5e then kills 125 runs (42 with clips); in Dungeon 4 the plain
   way (0x31 W, 0x30, 0x20) costs 0.9 hearts and 62 deaths before Gleeok (§8).
4. **One bomb per wall**: 0x2c 1.00, 0x67 1.01, 3:0x4c 1.00 (round 21 about 2), 4:0x11 1.00 per chain visit;
   4:0x21 1.6 (round 21 3.8). The rest in 4:0x21 is a second BOMB placed inside the first bomb's fuse (§5).

## 1. Natural RNG states

**The cycle** (`zelda-asm/mechanics.md` "Random numbers"): the 13-byte register `$18-$24` is stepped once per frame
by the game's rule from the reset seed (`$40`, then 12 zero bytes; `asm:Reset_ClearRam`). After an 88-step lead-in
it cycles with period 32,767; step 88 is `9C 8B B2 A5 C0 8B 0A 1C 08 30 20 40 00`, and step 87 is not on the cycle.
Round 21 and before wrote 13 independent bytes 1-255 at a start, which the game never holds.

**Built** (`crates/zelda/src/rng.rs`; `146d5e9` on `r22-rng`): the register as one 128-bit number, `natural(n)` (n
steps after reset), `draw_phase` (n uniform in 88..=32854 from the caller's seeded generator), `phase_of`,
`on_cycle`. No table: the longest `natural` takes about 44 µs against about 1.05 ms for a start state's load. All
three write sites use it: `Session::load` with `randomize_rng` (training resets, evaluation, a chain's start state),
the chain checkpoint redraw (`Session::write_rng`) and the search's `perturb_rng` (phase from the search future's
key). `Session::rng_phase()` reports the phase written. Power-on starts write nothing, as before.

**Where the write happens:** right after the restore, before `gain_control` runs any entry frame; a chain
checkpoint and a curriculum frame run no frame before it either. So the entry frames, the spawn clouds (the "second
and a half where they are spawning") and the first monster decisions run on natural values. A test logs every
restore, every write to `$18-$24` and the register at the start of every frame for `0_3cc_0`, `0_58n_0` (edge
spawns), `2_7ds_0` and `2_0de_0`: all 13 writes come before the first frame, the first frame runs on
`natural(phase)`, and every frame to 120 frames into play is on the cycle, one step after the last (none on a lag
frame). From power on the function equals the game's register on every frame: steps 0, 87, 88, 32854 and 32855 fall
on frames 10, 119, 120, 32886 and 32887 (32855 is 88 again; about 22 boot frames have no step).

**Two other writes the game could not hold, fixed:**

- **The edge spawn cell `$525`** was drawn from all 50 ring cells, including the corners `$E0` and `$4F`, where
  `asm:EdgeSpawn_FindNextCell` never stops; from one of them the search leaves the ring (`$F0`, `$00`, ..., `$3F`),
  and the next edge monster appears off the screen or the search never ends. It is now drawn from the 41 values the
  game holds (`$40`, its value after reset, and every ring cell open in at least one of the 20 edge-spawn rooms; a
  test derives the list from `zelda-asm/data`), in every level, since it carries through dungeons.
- **A fresh room** wrote `$FF` into the room history, which no room id is. Once the saved kills are cleared the
  history decides nothing, so the write is gone; `make_room_fresh` (Rust and Python) writes only the saved-kill bits.

The spawn cycle `$524` (0-8) was already holdable, and `ram_randomization` (hearts, bombs, kill streak) is not game
randomness: both unchanged.

**Replays:** the 13 bytes stay frame-0 EDITS records (each with the byte it replaced), so a replay reproduces the
start exactly; META `rng_phase` now keeps the phase (a stitch keeps each segment's) and `tf replay info` reads it
back from the edits ("a value the game never holds" for an older replay). The corpus replays verify unchanged
(61,551, 1,747 and 756 frames); `REPLAY_VERSION` is not bumped. Pool tests record a run from a start state and one
from a redrawn checkpoint: both verify bit-exactly and the phase read back matches META.

**What it changed for the models:** every seeded result differs (the same seed draws a different register, edge cell
and history), so per-seed layouts and replays from before round 22 don't compare one to one. Evaluated with the
round 21 models (seed 0, 100 episodes per state, each in its act mode), old → new:

| family (act) | success | hearts per episode | two standard errors |
|---|---|---|---|
| goriya (sample) | 95.2% → 95.8% | −0.15 → −0.19 | 1.5 points, 0.05 |
| ow-moblin (argmax) | 99.0% → 98.8% | −0.07 → −0.07 | 0.6 points, 0.01 |
| boss-gleeok (argmax) | 99.7% → 99.7% | +0.29 → +0.11 | 0.9 points, 0.19 |

Every shift is within noise; boss-gleeok's hearts (300 episodes) are the closest to the edge. The retrain's
re-evaluation of the 12 families kept on round 21 runs (on the rebuilt states as well) is within noise too (§7).

## 2. The clips switch

Decisions 2026-11-13 "The run-time clips switch and route stretches" and "Clips in training, evaluation and
watching"; `docs/architecture.md` (the clips switch, route stretches).

**The switch.** `clips_allowed`, default on, beside the family's `clips`: CLIP is offered only when both are on
(`Session::set_clips_allowed`, `GameCompletion`, `SpecialistEnv::set_clips_allowed(Option<bool>)`,
`ChainEnv::set_clips_allowed`; Python `Game.set_clips_allowed`, `Env.set_clips_allowed`). A flip takes effect at the
next decision the env or chain builds, so the observation and mask already handed out stay valid; the mask, the
`clip_usable_*` bits (now read from the mask, so they include the switch and the wrap veto; no layout change), the
objective, PBRS, `target` and linger follow it. Tests: a flip mid-room gives exactly the decisions of a switch fixed
from the start (0x77's wrap zone, 1:0x22 before and after its clip, 4:0x50 before and after its clip). Within a room
the switch can change the objective too: 1:0x22 without clips goes for the push spots instead of the stairs.

**Route stretches.** Where the clip way and the plain way go through different rooms, routes.yaml holds both:
`- stretch: NAME` with `parts`, `rejoins`, `clips:` and `plain:` step lists, flattened into one numbered route (the
clip way's steps first, each tagged with its `way` and `stretch`). Load checks: both ways leave `parts` by different
exits, each is a walk of game.yaml exits into the next step, both reach `rejoins` with the same progress and bomb
reserve, no waypoint or triforce inside. Two stretches: `d3-approach` (steps 75-79 clips, 80-86 plain, rejoining at
3:0x7c, step 87) and `d4-island` (122-123 clips: 0x31 E→N, 0x21 S→N; 124-127 plain: 0x31 E→W, 0x30 E→N, 0x20 S→E,
0x21 W→N; rejoining at 0x11, step 128). `eval-chain --from` picks the step on the run's way first; a `--stop-at`
the run's way never reaches is refused.

**Re-planning.** Switched off on the clip way, the plain step of Link's room for the side he entered takes over (0x31
from E: W). Where no plain step fits, the objective leads through the walkable exits toward the nearest room of the
plain way or the rejoin (0x21's island: back through its S door into 0x31, then W; 0x76 or 0x75's E part: back E and
N into 0x66), and adopts a plain step as soon as Link is in its room and can walk to its exit. Switched on while on
the plain way, the plain way holds to the rejoin (no turning back mid-stretch); anywhere else the clip way is taken
at once. A start state the route enters on one way only fixes the switch (`objectives::start_clips`; e.g. `4_21s`,
`0_75e` on, `4_30e`, `4_20s`, `4_21w`, `0_63e`, `0_73n` off). Chain records carry `clips_allowed` per visit (at its
first decision), `clips_flips` (`{step, clips}`, step 0 = at the entry) and the run's `clips`.

**The training mix.** A clip family's `clips: true` is `{enabled: true, off_share: 0.5}` (`env::config::ClipsConfig`):
each episode draws its switch from the env's seeded generator, off with probability `off_share`, unless its start
state fixes it or `Env.set_clips_allowed` does; curriculum starts and exploration keep the state's switch (on unless
the state fixes it off). Metrics `clips/enabled_share` and `clips/on|off/{episodes, success_rate, health_change,
hearts_at_exit}`; the early stop counts `<state>:clips-off` as a start state of its own, so both switch states must
clear the bar. The retrain moved ow-moblin and ow-zora to `off_share: 0.75` (§7). Measured `clips/enabled_share`
over the r72 runs: ow-octorok 0.44, ow-boulder 0.50, trap-room 0.50, stalfos-keese-gel 0.51, ow-tektite 0.56,
ow-start 0.60, ow-leever 0.76 (0x75 fixes it on), ow-moblin and vire 0.40 (their plain-only states), ow-moblin at
0.75: 0.20.

**The evaluation.** `triforce evaluate --clips on|off|both`, `both` the default for a clip family and the canonical
`eval.json`. Its headline (`success_rate`, `mean_health_change`, every group) is **one figure over both switch
states together**, so a model weak without clips can't hide behind its clip runs; `by_clips` keeps the full summary
per switch state (with `by_room`, `by_state`), and the switch-on half is exactly the `--clips on` evaluation (same
seeds). Each start state plays only the switch states it allows. `clips` is part of what makes two evaluations the
same measurement in `eval-batch`, so evaluations from before it are measured again.

**Chains and statistics.** `eval-chain --clips on|off` and `--clips-off-from ROOM|step:N`; `CLIPS=off
TAG=<tag>-noclips scripts/experiments/r56_chains.sh`. `triforce room-stats TAG --noclips` reads both tags into
`<TAG>-both-room-stats.json` (format 2: `by_clips` per room and per scope, run, arrival and ending counts per switch
state). `assign_steps` now uses the recorded `step` (`RoomRecord.step`, with `hits` and `hits_dealt`, round 21's
open item), inferring it only for older files; chain hits feed `zelda-map.py`'s Hits overlay, and the map has a Clips
selector (Both / On / Off) that draws both route ways.

**Watching.** `play-zelda.py --clips on|off` sets the switch its background simulations start with; key `c` turns it
off and on for every run started afterwards (queued runs keep theirs; the title says where it stands, each queue
counts its runs without clips, a cell's strip says "no clips"). `--live` makes the top left cell a live game the
models play while it is watched (a one-env chain pool paced at the cells' frame rate, `ChainTask(clips_live=True)`):
`c` acts on that run from its next decision and the route re-plans from where Link stands ("no clips from L4
0x31"); its finished runs are filed into the queues. It has no sound and no controller drawing.
`zelda-headless.py --clips on|off`; golden paths always play with clips. The training GUI shows success per switch
state (chart, progress line, the Eval tab's "Clips switch").

**The clip counters** (round 21's open item): a block or ladder clip whose crossing runs on into the next room now
counts as done (4:0x21's `CLIP N` through the opened wall into 0x11); an aborted clip that a later MOVE along its line
carries past the clip's end counts once in `clips/finished_by_move` (kept in `aborted`, not in `done`; tested in the
game with a ladder clip cut by a hit, then `MOVE N 16`).

**The wrap veto** (round 21's "wrap into a wrong exit"): `objectives::wraps_into_wrong_exits` drops a wrap when the
objective leaves by the landing edge only within its game.yaml `span` and the landing row is outside it. In 0x5b it
vetoes rows `$B5` and `$BD` (the lower E opening) and keeps `$55` and `$5D`; wraps toward other uses stay.

**Before the retrain** (round 21 models, which never trained with the switch off): vire `--clips both` 20 episodes
per state: on 99.4%, off 96.0% (4:0x30 88%); D4 chains of 20 runs on the route before §4's changes: clips on 13/20,
off 2/20, `--clips-off-from 4:0x31` 8/20.

## 3. The wrap exit 0x75 → 0x74

The owner's "77 → 76 → 75 (wrap) → 74" is real, but the scout (`r22-route-facts`, played) found it is an **E-edge**
wrap: Link wraps from 0x75's E edge to its own W edge, lands inside 0x75's tree wall (columns 0-1 are never read on
that wrap), and Left at X `$00` fires the W screen edge test even with an unwalkable tile ahead
(`asm:Link_StopAtUnwalkableTile`): the room changes to 0x74 and Link arrives at X `$F0` inside its tree columns.
Walking, 0x74 can't be reached from 0x75's E part (a river cuts the room; its W edge is solid on every row). Round
21's CLIP refused it, because the landing columns must be walkable. Built as a fourth skip kind,
`SkipKind::WrapExit` (`4f21368`; decisions 2026-11-13 "The wrap exit"; `zelda-asm/mechanics.md` "After the wrap, edge
by edge").

**The rule per row of 0x75** (played in the game):

| Y (Link's tile row) | E wrap, then Left | offered |
|---|---|---|
| `$3D`-`$4D`, `$C5`-`$DD` | no wrap: the run-up at the E edge is trees | no |
| `$55`, `$5D`, `$B5`, `$BD` | **softlock**: at X `$F0` in 0x74 only Right moves Link (back to 0x75 at X `$00`), and there only Left (back to 0x74) | no |
| `$65`-`$AD` (rows 5-14) | arrives in 0x74 at X `$F0` and walks on W: 0x74's column 29 is open on these rows | yes |

0x76's straight E-W lanes deliver Link on exactly the softlock rows, so he must change rows in 0x75 first (X
`$E0`-`$F0` is open on all of them). Left must come on the landing frame, or within the 5 frames after it while X ≤
`$07`; from X `$08` Link stands in 0x75's W part (not a softlock: N leads to 0x65). The macro: Right 8 frames (X
`$EB`), Up 1, Right 15 (X `$00`), Left 1; 0x74 plays 104 frames after the Left, 130 frames for the whole CLIP. The
room change counts as done.

**Hits:** the skip aborts on a hit like the ordinary wrap, and no hit can leave Link stuck (a Peahat placed on every
frame of the macro, from both sides, on three rows): before the bound he is knocked back into the zone (`CLIP E`
again), at X `$F8`-`$FD` knocked right he stays on the row (MOVE W walks back), on the landing frame the left bound
stops him at X `$00` (MOVE W leaves at once and counts as finishing the cut skip).

**A whitelist, not a derived rule:** the zelda crate reads tiles only from RAM and has no overworld layout decoder,
so 0x74's column 29 can't be seen from 0x75. game.yaml's 0x75 has a W exit to 0x74 with `wrap: true` and `span: [5,
14]`; the parser refuses `wrap` on anything but an overworld W exit with a span, and `crates/zelda/tests/wrap_exit.rs`
plays every whitelisted row (both sub-pixel cadences) and the softlock rows by buttons. The path search gets jumps
heading E from X `$E0`, `$E8`, `$F0` on the safe rows, so PBRS and `target` point at the E edge on a safe row when
the step leaves W, and only with clips on. With clips off nothing offers it and the W exit is unreachable, so the
stretch's plain way avoids 0x75. The mirrored shape (a W wrap onto a solid E edge, 0x60 into 0x5f) is not offered.

**Measured:** ow-leever `0_75e` 100% in single rooms (one CLIP per visit); in the after-D1 chain 1000 visits, 1.00
clips per visit, no abort, 0 hearts, 162 frames per visit, 1 run stuck. Before the retrain, round 21's ow-leever
never found it (8 of 20 runs stuck bouncing between 0x75 and 0x76).

## 4. The route changes

`triforce/routes.yaml` `d2first`, `docs/routes/overworld.md`, `d3.md`, `d4.md`; decisions 2026-11-13 "Round 22
route". The steps that changed (list them with `_native.route_steps('d2first')`; they move whenever the route does):

| steps | rooms (entry → exit) | family |
|---|---|---|
| 71-73 | 0x58 N → **S** (was W), 0x68 N → W, 0x67 E → cave (bomb the rock wall) | ow-octorok |
| 74 | 0x67's cave S → S, `waypoint: rupees` | ow-octorok |
| 75-79 (clips) | 0x67 cave → S, 0x77 N → W, 0x76 E → W, 0x75 E → W (wrap exit), 0x74 E → cave | ow-octorok, ow-start, ow-tektite, ow-leever, ow-tektite |
| 80-86 (plain) | 0x67 cave → W, 0x66 E → W, 0x65 E → W (its ford), 0x64 E → W, 0x63 E → S, 0x73 N → E, 0x74 W → cave | ow-octorok ×2, ow-zora, ow-octorok, ow-moblin ×2, ow-tektite |
| 87 | 3:0x7c S, both ways rejoin; `bomb_reserve: 6` | zol |
| 100-103 | 3:0x5b W → **E** (bomb wall; `treasure: {if_bombs_below: 6}`), 0x5c W → N, 0x4c **S** → E (bomb wall), 0x4d | darknut, darknut, trap-room, boss-manhandla |
| 122-127 | the `d4-island` stretch (§2) | vire, stalfos-keese-gel |
| 128-131 | 4:0x11 S → **N** (bomb wall), 0x01 S → E (the key), 0x02 W → S, 0x12 **N** → E | trap-room, stalfos-keese-gel, trap-room, vire |
| 132-133 | Gleeok, the D4 triforce | boss-gleeok, vire |

Round 21's steps 0x58 W, 0x57, 0x56, 0x55 S and 0x65 from N are gone (their states `0_57e`, `0_56e`, `0_55e`, `0_65n`
removed), and 3:0x4b left the route; 0x56, 0x57 and 3:0x4b stay tagged, without start states (0x55 stays on the
route as the raft's landing after Dungeon 3). New rooms in families: 0x66 and 0x67's cave (ow-octorok, now a
`required` bomb family), 0x75 (ow-leever), 0x76 (ow-tektite), 3:0x5c (darknut), 4:0x01 (stalfos-keese-gel), 4:0x02
(trap-room).

**0x67: the wall, the cave, the rupees.** levelblock `$67`: a rock wall (tile object `$63`, quest 1 only) over cave
`$21`, whose person gives 30 rupees ("IT'S A SECRET TO EVERYBODY"). game.yaml gained `bomb_secret: [14, 2]`,
`requires_bombs`, the cave exit and the in-cave room. The wall opens from facing N on row Y `$5D` at X `$68`, `$70`
or `$78` (not `$60` or `$80`), about 190 frames after `BOMB N`. In the cave the rupee ware pays only a Link at X
exactly `$78` and Y within 6 of `$98` (`asm:Cave_State2TalkOrShopOrDoorCharge`): walking straight up from the mouth
(X `$70`) takes nothing. The rupees are credited one every second frame (24 → 54), once per game; the opened wall
stays open. Leaving 0x67 without them is `missed`, like the 0x2c heart. They buy nothing on the route: arrows are 80
rupees in shops off the route, 0x66's shop sells a shield (160), a key (100) and a blue candle (60), and Link holds
42-54 after the cave. Route test (`crates/zelda/tests/route_ways.rs`, monsters removed): from 0x67's E entry the wall
opens at 300 frames, Link is out of the cave with the rupees at 903.

**After Dungeon 1, both ways** (from 0x67's cave mouth to Dungeon 3's entrance):

| | clips (0x77, 0x76, 0x75, 0x74) | plain (0x66, 0x65, 0x64, 0x63, 0x73, 0x74) |
|---|---|---|
| scout's blind walk (fixed RNG) | 1168 frames, 0 hearts | 1896 frames, 1.5 hearts (7 of 10 random-RNG walks arrive) |
| route test, monsters removed | 1167 frames | 1787 frames |
| r72 chains, per run | 5 rooms, 1,269 frames, 0.35 hearts | 7 rooms, 2,073 frames, 1.20 hearts |

From 0x58's entry the r72 chains take 2,537 frames with clips and 3,341 without (the sums of the per-visit means,
§8); round 21's way (0x58 W, 0x57, 0x56, 0x55, 0x65, 0x64, 0x63, 0x73, 0x74) took 1,976 frames (median) and 0.25
hearts in its chains. 0x67 from E to the cave takes 582 frames per visit and the cave 229 (the scout: the bomb and the
cave cost about 660 frames over walking through).
The plain way is not the owner's "line": 0x65's river is crossed only by its ford (rows Y `$85`/`$8D`), and 0x63 turns
S to 0x73; Link must keep off the mouths of 0x66's shop and 0x64's potion shop.

**Dungeon 3 by 0x5b, 0x5c, 0x4c.** 0x5b's E wall from facing E on Y `$8D` at X `$C0`; 0x5c's N door is a plain open
doorway (no shutter); 0x4c from S, then its E wall from the same spot. The scout's walk from `3_5bw_0` (2 bombs set):
1,267 frames to 0x4d, 1 heart to a Darknut in 0x5c; round 21's 0x5b (W) .. 0x4c in the r71 chains: 2,184 frames
(median), 0.2 hearts. Route test: 0x5c at 561, 0x4c at 829, 0x4d at 1,237 frames. Entered from S, 0x4c's two bottom
traps charge along the bottom row: a Link who lingers at the door is hit.

**Dungeon 4 by 0x11 N, 0x01, 0x02, 0x12.** The owner's "from 0x11 go north to 0x10 → 0x20 → 0x12" can't be walked as
written: 0x10 lies W of 0x11 and 0x20 SW, neither borders 0x12, and the only key in that corner is 0x01's, north of
0x11 (level-4.md's "without bombs" way, 0x20 → 0x10 → 0x00 → 0x01 → 0x02 → 0x12, is probably where the names came
from). The lead read it as 0x11 N → 0x01 → 0x02 → 0x12, and the scout confirmed it by play: bomb 0x11's N wall from
(`$78`, `$5D`) facing up (the line X `$78` takes the rupee stash, 32 → 40); in dark 0x01 (8 blue Keese) the ladder
crosses the one water square above the S door, the key lies at (`$80`, `$80`) (keys 5 → 6), and a second single
square leads to the E door; dark 0x02 (6 traps, no monsters) by the doc's timing; 0x12 from N, where the Vires must
still all die before the block (`$60`, `$90`) slides S and the E shutter opens, as from W. Route test: 0x01 at 505,
0x02 at 858, 0x12 at 1,165, shutter open at 1,272, 0x13 at 1,594 frames from 0x11's entry.

**Keys, bombs, rupees along the route:**

| | keys | bombs | rupees |
|---|---|---|---|
| after Dungeon 1 | as before | 0x67's wall takes one (the post-D1 states hold 1, 4 and 0) | +30 in 0x67's cave |
| Dungeon 3 | 0x4b's key and its E key door both leave: none on balance | `bomb_reserve: 6` from the entrance (was 5): 0x5b's and 0x4c's walls, a spare each, 2 for Manhandla; 0x5b's last pass takes its treasure (4 bombs, after its 3 Darknuts) only below 6 | |
| Dungeon 4 | 5 on entry, 6 after 0x51, 7 after 0x40, 6 after 0x30 E, 5 after 0x31 N, **6 after 0x01** (round 21: 5); D4 itself needs none | `bomb_reserve: 2` (0x21 N, 0x11 N; 0x11 E left the route) | about +8 in 0x11's stash |

**What skipping 0x4b loses:** its key (spent at once on its own E door before: none on balance), a fight with 3 Zols
(about 1.8 drops when killed outright with the white sword) and 3 kills toward the help streak; it saves about 900
frames and costs a bomb (0x5b's wall). In the chains round 21's 0x4b took 975 frames, 0.17 hearts and 38 deaths in
647 visits; round 22's 0x5c 280 frames, 0.16 hearts and 27 deaths in 702.

**What the 0x01 key costs:** two room entries and about 780 frames in the scout's walk (0x11 → 0x13 1,710 frames with
2 hearts lost, against 934 the old way, 0x11 E → 0x12 W). In the D4 chains, 0x11 + 0x01 + 0x02 + 0x12 take 2,290
frames per run (clips on) against round 21's 0x11 + 0x12 1,739; 0x01 kills 6 runs per 1000 (9 without clips), 0x02
none, and 4:0x12 entered from N kills 27 (round 21 from W: 95; vire was retrained too). The key is a spare for later
dungeons.

## 5. The BOMB mask for wall families

Lee: "Try to fix the 0x21 bombs thing, that's pretty bad" (round 21: 3.8 bombs per 4:0x21 visit, one needed).
Built on `r22-route` (`1d1d787`; decisions 2026-11-13 "Round 22 route").

**The rule:** in a `requires_bombs` room, a `required` family (ow-tektite, ow-octorok, trap-room,
stalfos-keese-gel) is offered BOMB in a direction only if the bomb would land within the game's reach of a wall the
current objective leaves through; otherwise only on the 10th damage-free kill. `free` families (darknut, vire and the
bosses) are untouched. Where the bomb lands: 16 px ahead of Link on the press frame, after any turn
(`zelda::bombs::landing`, `attack::press_link`); a dungeon wall opens within `$18` px of its hotspot on both axes
(`asm:Bomb_TryOpenWall`), an overworld rock wall within `$10` (`asm:RockWall_Update`,
`asm:Collision_CheckTileObjWeapon`). Code: `zelda::bombs::{BombTarget, landing}`, `objectives::route_bomb_targets`,
`Session::set_bomb_targets`. One fix it needed: a bombed rock wall leaves black tiles, not a mouth tile, so
`ow_bomb_secret_closed` now also reads the secret-found flag.

**Proven in the game** (`crates/zelda/tests/bomb_mask.rs`): in 0x2c, 0x67, 3:0x4c, 4:0x11, 4:0x21 from W and 4:0x21
from the island, every grid point within 48 px of the wall in all four directions: every offered BOMB opened the
wall with one bomb, and every BOMB the game would place but the mask refuses left it shut. BOMB N at the ladder
clip's end in 4:0x21 is no longer offered.

**Bombs per wall, chain visits** (§8; round 21 = `r71-1k-dks`):

| wall | family | round 21 | round 22 |
|---|---|---:|---:|
| 0:0x2c | ow-tektite | – | 1.00 |
| 0:0x67 (new) | ow-octorok | – | 1.01 |
| 3:0x4c | trap-room | 1.98 (2 bombs in 434 of 609 visits) | 1.00 |
| 4:0x11 | trap-room | 1.00 | 1.00 |
| 4:0x21 | stalfos-keese-gel | 3.83 (3522 in 919) | 1.61 clips on (1558 in 967), 1.65 off (1478 in 895) |

In single rooms (sample | argmax): 0x67 1.22 | 1.14, 0x2c 1.09-1.14 | 1.07-1.08, 3:0x4c 1.15-1.20 | 1.05-1.09,
4:0x11 1.13-1.18 | 1.09-1.10, 4:0x21 from S 1.73 | 1.83, from W 1.76 | 1.51 (round 21: 3.4-3.8 from S).

**The second bomb in 4:0x21** (from stalfos-keese-gel's argmax replays, episodes 4501, 4509, 9618, 9626): the first
bomb is in reach and does open the wall, but a second is placed while the first is still in its fuse or blast (the
game allows a new bomb once the first is in state `$13`), before `CurOpenedDoors` changes. From S: BOMB N from the
island (Y `$7D`-`$82`), `CLIP N`, then BOMB S at Y `$57` 50-70 frames later; from W: BOMB at Y `$5D` twice. A rule "no
BOMB while a live bomb (state `$12`/`$13`) is in reach of the same wall" would remove it; not built (§9). With
Dungeon 4's reserve of 2 a real-bomb run can't afford it. Also seen: BOMB at the island's entry (Y `$B6`) where the
S door's hotspot is the first in range (no wall opens).

## 6. States and curricula

**States, rebuilt from 0x58 (step 71) to the D4 triforce** (`7e2a865`; `scripts/experiments/r22_states_curricula.sh
states|plain|install`): the clip way by `entry_states build` (3 chains, `--iterations 30000 --patience 400`), the plain
ways by `entry_states path` (from `0_67c_N` through 0x66-0x74, from `4_31e_N` through 4:0x30, 0x20, 0x21); every
crossing on its first attempt. Removed: `0_57e`, `0_56e`, `0_55e`, `0_65n`, `3_4bs`, `3_4cw`, `4_12w`. The audit: 397
states, 0 problems (364 before the round, 373 after the switch's D4 plain-way states). The whole route was not
rebuilt: round 21's builds took about 5.9 CPU-hours, and with the machine loaded (load average about 28 on 20
threads) the rebuild from 0x58 (about 3.7 CPU-hours) was what fit. `4_11s_0` now starts deeper in the doorway (`$78`,
`$C4`), a valid state; it exposed a mask bug (a diagonal boomerang throw walked Link back out of the room), fixed in
`attack::press_leaves`.

**Explorer fixes the build needed:**

- **The alternate 0x78 at 0x68.** `step_next_rooms` accepted any alternate of the step's next room that borders the
  current room; 0x78 (meant for leaving 0x77 at the game start) borders 0x68 too, so leaving 0x68 S counted as
  correct, and the explorer took it in two of three chains. An alternate now counts only from the room its first
  step is entered from.
- **The cave's rupees in Go-Explore's cells.** `env::explore::cell_key` didn't change when the rupees were taken, so
  the shorter path without them kept the cell: 52 cells, no success in 3000 iterations. The key now holds the room's
  item-taken flag and the pickup counts as progress: solved in 436 iterations (27 s).

**Curricula**, explored again on the merged tree: 267 tracks in 18 families, 156 new and 111 copied from round 21
(unchanged state files), every state solved. A state the route enters on one way only is explored with that way's
switch: 15 tracks with the switch off (ow-moblin `0_63e`, `0_73n`, ow-zora `0_65e`, vire `4_30e`, `4_20s`), none
clipping. Tracks that clip: ow-leever `0_75e_*` (the wrap exit), ow-moblin `0_63s_0`, ow-zora `0_65w_*` (wraps),
trap-room's six 1:0x22 tracks and vire `4_50e_*` (block clips).

**Old states that keep a flaw** (listed in the backlog, not rebuilt): 71 chain-2 states from before 0x58 (and a few
off-route `path` states such as `0_58s_2`, `0_59w_2`, `0_5cw_2`) hold an edge cell the game can't hold (`$E0` in
most), and 93 hold `$FF` in the room history (harmless). A randomized load replaces the cell; a load without
randomization and every curriculum frame keep it.

## 7. The retrain (r72)

Code `bfef4b3` (1.4.1.1) plus `5efba1c` (YAML: ow-moblin and ow-zora `clips: {enabled: true, off_share: 0.75}`).
All training through `scripts/train_round.py --parallel 8 --eval-parallel 3 --chain-runs 0 --train-args "[--load
<run>] --stop-success X --sil-coeff 0.1"`, X = 0.99, and 0.995 for the second passes. The round's own evaluation is
sample with `--clips both` (`eval-sample.json`); argmax with replays; the promoted act's evaluation is `eval.json`.
17 runs, 36.4M env steps, 10.7 h of training in up to two streams (Nov 13 20:21 → Nov 14 02:31; the first 3.5 h next
to another job's chains, about 475 instead of 1100 steps/s per stream). Every run: the expected `curriculum_tracks`,
episodes finishing from iteration 1, success logged per switch state.

**Which families:** R22Merge's list (every family whose rooms, entries, exits or masks changed: ow-octorok from
scratch, its task changed most and round 21's model never bombs; the others continued from their round 21 runs) plus
like-like, whose round 21 run lost 0.90 hearts per episode on the rebuilt states and natural RNG (round 21's own
evaluation: 0.53).

Per family: success and hearts per episode over both switch states in the canonical evaluation (the promoted act),
then with clips on | off; watched rooms "on; off" (`–` = the state allows one switch state only):

| family | run (act) | from, steps | both | on \| off | watched rooms | round 21 |
|---|---|---|---|---|---|---|
| ow-octorok | r72-octorok (argmax) | scratch, 2.08M | 99.2 −0.08 | 99.1 −0.07 \| 99.3 −0.10 | 0x67 E 100; 100 (1.14 bombs), 0x67 cave 100; 100, 0x66 –; 100, 0x64 –; 99.3 | r71-d 99.0 −0.06 |
| ow-start | r72-ow (argmax) | r71-ow, 0.23M | 100.0 +0.00 | 100.0 \| 100.0 | 0x77 N 100; – | r71-ow 100.0 +0.00 |
| ow-leever | r72-ow (argmax) | r71-ow, 0.52M | 100.0 +0.09 | 100.0 +0.07 \| 100.0 +0.13 | 0x75 E 100; – (0.33 clips per episode) | r71-ow 99.7 +0.11 |
| ow-tektite | r72-ow (argmax) | r71-ow, 0.44M | 99.4 +0.06 | 99.7 +0.04 \| 99.2 +0.07 | 0x76 E 100; –, 0x74 E 100; –, 0x2c 99.7; 98.0 | r71-ow 99.6 +0.06 |
| ow-boulder | r72-ow (argmax) | r71-ow, 0.44M | 99.9 −0.07 | 100.0 −0.00 \| 99.9 −0.13 | 0x18 100; 100 (−0.17 off), 0x19 100; 99.7 | r71-ow 100.0 −0.01 |
| ow-zora | r72-zora-off (argmax) | r71-ow 0.55M, then 1.82M at 0.75 | 99.6 −0.13 | 99.9 −0.11 \| 99.4 −0.16 | **0x5a 100; 96.3**, 0x59 100; 100, 0x65 E –; 98.7 | r71-ow 99.9 −0.12 |
| ow-moblin | r72-moblin-off (argmax) | r71-d 4M, then 4M at 0.75 | 98.5 −0.08 | 98.7 −0.08 \| 98.3 −0.08 | **0x5b W 99.3; 91.7**, 0x63 E –; 100, 0x73 N –; 99.7 | r71-d 99.0 −0.07 |
| trap-room | r72-trap (argmax) | r71-a, 0.93M | 99.9 −0.04 | 100.0 −0.03 \| 99.9 −0.04 | 3:0x4c S 100; 99.3, 4:0x11 100; 100, 4:0x02 100; 100, 1:0x22 100; 100 | r71-a 99.8 −0.16 |
| stalfos-keese-gel | r72-skg (sample) | r71-b, 0.25M | 99.8 +0.02 | 99.9 +0.01 \| 99.8 +0.03 | 4:0x01 100; 100, 4:0x21 S 100; –, 4:0x21 W –; 99.7 | r71-b 99.7 +0.02 |
| darknut | r72-darknut (sample) | r71-darknut-cont2, 4M | 98.4 −0.42 | (no clips) | 3:0x5b W→E 97.0 −0.72 (4.3 bombs), 3:0x5c 100, 3:0x59 91.7 −1.52 | r71-darknut-cont2 97.8 −0.39 |
| vire | r72-vire-cont (sample) | r71-vire-cont 0.56M, then 4M at 0.995 | 99.3 −0.01 | 99.3 +0.03 \| 99.4 −0.04 | 4:0x12 N 99.3; 99.7, 4:0x50 99.0; 99.0, 4:0x31 E 99.0; 98.7, 4:0x30 –; 99.7, 4:0x20 –; 100 | r71-vire-cont 99.0 −0.06 |
| like-like | r72-likelike2 (argmax) | r71-c 0.83M, then 3.75M at 0.995 | 99.5 −0.48 | (no clips) | 4:0x32 W 99.0 −0.89 | r71-c 98.3 −0.53 (sample) |

Round 21's numbers are its own evaluations (switch always on, the states and RNG before round 22). Against them,
room by room (beyond about 2 points or 0.15 hearts): better in trap-room 3:0x5a from E (−0.21 against −0.82 hearts),
vire 4:0x50 with clips (−0.25 against −0.44) and 4:0x31 W (97.0 −0.42 against 95.7 −0.71), like-like 4_32c (−0.07
against −0.21); worse in none with the same task. The plain way, new: ow-boulder 0x18/0x19 lose 0.17-0.22 hearts per
crossing (wrap: 0.00), vire 4:0x31 E→W 0.64-0.66 (round 20: 0.65), 4:0x50 without clips 0.41-0.46 (with 0.23-0.25).

**The 12 families kept on round 21 runs**, re-evaluated on the rebuilt states and natural RNG in both act modes
(`target/r72/reeval/`, outside their run dirs), act modes unchanged: ow-lynel 98.9% −0.15 (round 21 98.7% −0.15),
ow-peahat 100.0 −0.04, goriya 95.8 −0.18 (95.2 −0.15; 2:0x1e 93.0, 2:0x5e 87.7), d2-rope 98.7 +0.24, moldorm 92.0
−0.24 (89.3), boss-dodongo 99.0 +0.89, wallmaster 100.0, boss-aquamentus 100.0 +0.96, equipment-room 99.7 −0.17, zol
99.6 +0.14, boss-manhandla 94.7 −0.33 (96.7 −0.31), boss-gleeok 99.7 +0.16 (+0.29; 300 episodes, within two standard
errors). Within noise of round 21 throughout, moldorm and goriya a little better.

**Act modes, on evidence** (success first, then hearts within about 0.5 points, argmax avoided where it loops;
decisions 2026-11-14 "The r72 retrain"). Sample | argmax, with loop endings (stuck, no-damage, timeout, max-steps)
where they decided:

- argmax: ow-octorok (99.1 | 99.2), ow-start, ow-leever, ow-boulder (ties), ow-tektite (99.6 +0.05 | 99.4 +0.06:
  within 0.5, hearts), ow-zora (99.5 | 99.6), ow-moblin (98.4 −0.10, 0 loops | 98.5 −0.08, 4), trap-room (99.7
  −0.10 | 99.9 −0.04), like-like (99.5 −0.48 both; linger worst 539 by sample, 98 by argmax: **from sample to
  argmax**).
- sample: stalfos-keese-gel (99.8, 1 loop | 99.7, 16), darknut (98.4 −0.42, 0 | 97.7 −0.39, 16), vire (99.3, 6 |
  98.8, 38, 19 of them in 4:0x12 from N).

**Remedies:**

- **vire:** r72-vire early-stopped at 0.56M with 4:0x50 without clips at 95.7% −0.63 (with: 98.0% −0.29) and 4:0x12
  from N at 96.7-98.7%. Continued 4M at `--stop-success 0.995` (r72-vire-cont): 0x50 off 99.0% −0.41, 0x12 N
  99.3-99.7%.
- **like-like** (r71-c on the rebuilt states: 97.8% −0.90; 4_32c −0.55, 4_32w −1.26..−1.35): continued twice (0.83M,
  then 3.75M at 0.995): 99.5% −0.48, 4_32w −0.89 (round 21's own figure −0.85).
- **ow-moblin:** r72-moblin (4M) crossed 0x5b W→E without clips at 90.7/89.3% (with the wrap 100%). With `off_share:
  0.75` and 4M more (r72-moblin-off): 92.3/91.7% −0.23. From scratch with the same share (r72-moblin-new, 8M): 92.3%,
  −0.35..−0.37, no better overall, not promoted. The plain 0x5b stays about 3 points under round 20's pre-clip
  model (94-96%, −0.20..−0.25).
- **ow-zora:** r72-ow crossed 0x5a without clips at 96.0/98.0% and 0x59 at 98.3/97.7%; with `off_share: 0.75`
  (r72-zora-off, 1.82M at 0.995) 0x59 off 100%, 0x5a off unchanged (96.0/96.3%; pre-clip model 97-98%).

**The two rooms weaker without clips** are ow-moblin's 0x5b W→E (91.7% −0.22 against 99.3% −0.02) and ow-zora's 0x5a
(96.3% −0.16 against 100.0% +0.00): both are crossed before Dungeon 2, where the chains show the cost (§8).

Sub-screen frames per episode, from the argmax replays (no evaluation records `menu_frames`): vire 4:0x12 from N
2.6-6.4 opens (320-790 frames), 4:0x31 W 1.9-5.1, stalfos-keese-gel 4:0x21 1.7-1.9 (round 21: 3.5), trap-room
3:0x4c 1.8-1.9, darknut 3:0x5b 2.0-3.2, ow-octorok 0x67 1.1.

## 8. Chains, 1000 runs per segment

`RUNS=1000 ACT=model CLIPS=on TAG=r72-1k scripts/experiments/r56_chains.sh run` and `report` (45 minutes), then
`CLIPS=off TAG=r72-1k-noclips` (42 minutes), nothing else running; `triforce room-stats r72-1k --noclips` →
`target/r56-chains/r72-1k-both-room-stats.json`. For the entry measurements (bombs, streak) the D3 segment (clips on)
and the D4 segment (on and off) were run again with replays (`r72-chain-replays`); every run matched the measured
segment's seed, result and frames (all 3000).

**Arrivals** (hearts at arrival in brackets):

| segment | `r72-1k` (clips on) | `r72-1k-noclips` | `r71-1k-dks` | `r70-1k` |
|---|---:|---:|---:|---:|
| D2 | 87.2% | 87.2% | 85.6% | 68.6% |
| after D2 | 91.0% (3.03) | 89.5% (2.66) | 88.6% (2.81) | 83.5% |
| D1 | 98.9% | 99.5% | 99.1% | 96.7% |
| after D1 (to 3:0x7c) | 99.9% (4.97) | 98.9% (4.17) | 99.4% (5.05) | 100.0% |
| D3 (to 0:0x74) | 50.4% | 50.4% | 46.5% | 38.6% |
| after D3 | 100.0% (6.66) | 100.0% (6.40) | 100.0% (6.66) | 99.8% |
| D4 | **73.6%** | 56.8% | 56.4% | 21.1% |
| full → D2 | 100.0% (2.63) | 98.2% (2.24) | 100.0% (2.62) | 98.1% |
| full → after D2 | 86.7% | 68.3% | 86.9% | 56.3% |
| full → D1 | 79.3% | 62.0% | 75.6% | 45.8% |
| full → D3 | 59.5% (5.08) | 41.0% (4.15) | 56.7% (5.18) | 26.7% |
| full → after D3 | 19.5% | 10.2% | 24.8% | 8.0% |
| full route (D4 triforce) | **13.6%** | **4.8%** | 12.8% | 1.8% |

The D2 and D3 segments are the same in both settings (no clip family plays in Dungeon 2; none in Dungeon 3 or on
its start state's way). Noise: at 1000 runs one standard error is 1.1-1.6 points between 13% and 50% (under 0.4
near 99%), so a difference between two tags needs about 3-4.5 points to clear two standard errors (under 1 near
99%). Clips on against round 21, D2 (+1.6), after D2 (+2.4), D3 (+3.9), full → D3 (+2.8) and the full route (+0.8)
are within it, full → D1 (+3.7) at its edge; D4 (+17.2) and full → after D3 (−5.3) are beyond it. Clips off against
on: after D1 (−1.0: 8 deaths in 0x74 from W, 3 wrong exits), D4 (−16.8) and every full-route arrival (from full → D2,
−1.8) are beyond it; after D2 (−1.5) and D1 (+0.6) are within it.

**Where runs end** (deaths and stuck runs per 1000):

| segment | clips on | clips off |
|---|---|---|
| full | 3:0x5b 145, 3:0x59 119, 3:0x4d 64, 4:0x13 44, 2:0x5e 42 | 2:0x5e 125, 3:0x59 99, 3:0x5b 99, 2:0x3e 61, 2:0x1e 58 |
| D2 | 2:0x1e 49, 2:0x3e 27, 2:0x5e 24, 2:0x0e 16 | the same |
| after D2 | 0:0x0a 37, 0:0x1a 22, 0:0x27 16, 0:0x38 10 | 0:0x0a 35, 0:0x1a 25, 0:0x27 16, 0:0x18 11, 0:0x19 10 |
| D1 | 1:0x35 6, 1:0x44 4 | 1:0x35 4 |
| after D1 | 0:0x75 1 stuck | 0:0x74 W 8 deaths, 0:0x66 3 wrong exits (into its shop) |
| D3 | 3:0x4d 168, 3:0x5b 149, 3:0x59 80, 3:0x69 44, 3:0x5c 27 | the same |
| D4 | 4:0x13 192, 4:0x12 27, 4:0x31 20, 4:0x01 6 | 4:0x13 244, 4:0x12 73, 4:0x31 44, 4:0x30 25, 4:0x32 16 |

Round 21 (`r71-1k-dks`): full 3:0x59 108, 4:0x13 71, 3:0x4d 62, 0:0x0a 47; D3 3:0x4d 132, 3:0x5b 126, 3:0x69 87,
3:0x59 86; D4 4:0x13 245, 4:0x12 95, 4:0x32 35, 4:0x31 31.

**After Dungeon 1** (after-D1 segment; visits, deaths, stuck, hearts lost per visit, frames per visit):

| room | entry → exit | clips on | clips off |
|---|---|---|---|
| 0:0x58 | N → S | 1000, 0, 0, 0.03, 215 | the same |
| 0:0x68 | N → W | 1000, 0, 0, 0.03, 242 | the same |
| 0:0x67 | E → cave | 1000, 0, 0, 0.04, 582; bombs 1015 (15 cheated) | the same |
| 0:0x67 cave | S → S (rupees) | 1000, 0, 0, 0.00, 229 | the same |
| 0:0x67 | cave → S \| cave → W | 1000, 0, 0, 0.15, 196 | 1000, 0, 0, 0.00, 226 |
| 0:0x77 | N → W | 1000, 0, 0, 0.00, 257 | – |
| 0:0x76 | E → W | 1000, 0, 0, 0.13, 330 | – |
| 0:0x75 | E → W (wrap exit) | 1000, 0, 1, 0.00, 162; 1.00 clips, 0 aborted | – |
| 0:0x74 | E → cave | 999, 0, 0, 0.07, 324 | – |
| 0:0x66 | E → W | – | 1000, 0, 0, 0.28, 360 |
| 0:0x65 | E → W | – | 997, 0, 0, 0.40, 423 |
| 0:0x64 | E → W | – | 997, 0, 0, 0.18, 336 |
| 0:0x63 | E → S | – | 997, 0, 0, 0.00, 177 |
| 0:0x73 | N → E | – | 997, 0, 0, 0.05, 181 |
| 0:0x74 | W → cave | – | 997, 8, 0, 0.29, 370 |

The 3 wrong exits without clips go into 0x66's shop by its open mouth; round 21's 6 went into 0x64's cave, none now.

**Dungeon 3** (D3 segment; the same in both settings; visits, deaths, hearts lost per visit, frames per visit, bombs
placed (cheated)):

| room | entry → exit | r72 | r71-1k-dks |
|---|---|---|---|
| 3:0x5b | S → W | 998, 0, 0.06, 329 | both passes: 1773 visits, 126 deaths, 0.28, 388 |
| 3:0x5b | W → E (bomb wall) | 851, 149, 0.86, 1006, 3578 (1097) | (W → N) |
| 3:0x5c | W → N | 702, 27, 0.16, 280, 3 | – |
| 3:0x4b | – | – | 647, 38, 0.17, 975 |
| 3:0x4c | S → E (r71 W → E) | 675, 1, 0.00, 577, 678 (0) | 609, 4 (+8 stuck), 0.10, 976, 1205 (0) |
| 3:0x4d | W → N | 674, 168 (+2 stuck), 0.46, 754, 2474 (610) | 597, 132, 0.48, 748, 2525 (992) |
| 3:0x59 | N (kill all) | 1001, 80, 1.80, 863 | 86 deaths + 24 stuck (all passes) |
| 3:0x69 | N | 917, 44, 0.64, 280 | 87 deaths (both passes) |

**Dungeon 4** (D4 segment; visits, deaths, stuck, hearts lost per visit, frames per visit; the segment starts with no
bombs, so every wall bomb in it is cheated):

| room | entry → exit | clips on | clips off | r71-1k-dks |
|---|---|---|---|---|
| 4:0x50 | E → N | 1000, 0, 0, 0.39, 376; 1.01 clips | 1000, 1, 0, 0.82, 557 | 1 death, 0.46 |
| 4:0x40 | S → N | 1000, 0, 0, 0.30, 498 | 999, 2, 0, 0.32, 497 | 1, 0.29 |
| 4:0x31 | W → E | 1000, 5, 0, 0.82, 1193 | 997, 7, 0, 0.82, 1196 | 31 (both passes), 0.61 |
| 4:0x32 | W → cave | 995, 6, 1, 0.84, 1197 | 990, 16, 2, 0.82, 1241 | 35 (both passes), 0.64 |
| 4:0x60 cellar | S → N | 988, 6, 0, 0.15, 801 | 972, 9, 0, 0.18, 802 | 13, 0.15 |
| 4:0x31 | E → N \| E → W | 982, 15, 0, 0.20, 392 | 963, 37, 0, 0.70, 403 | |
| 4:0x30 | E → N | – | 926, 25, 0, 0.21, 360 | |
| 4:0x20 | S → E | – | 901, 6, 0, 0.06, 393 | |
| 4:0x21 | S → N \| W → N | 967, 5, 1, 0.26, 598; 0.85 clips (258 aborted); bombs 1558 | 895, 1, 0, 0.06, 773; bombs 1478 | 919, 10 + 2 stuck, 0.28, 956; 0.77 clips (311 aborted); 3522 |
| 4:0x11 | S → N (r71 S → E) | 961, 0, 0, 0.00, 501; bombs 975 | 894, 0, 0, 0.00, 499; 899 | 908, 0, 0.00, 526; 907 |
| 4:0x01 | S → E | 961, 6, 0, 0.12, 406; key in 961 | 894, 9, 0, 0.12, 407; key in 892 | – |
| 4:0x02 | W → S | 955, 0, 0, 0.00, 348 | 885, 0, 0, 0.00, 348 | – |
| 4:0x12 | N → E (r71 W → E) | 955, 27, 0, 0.49, 1035 | 885, 73, 0, 0.49, 988 | 908, 95, 0.81, 1213 |
| 4:0x13 | W → N | 928, 192, 0, 0.59, 771 | 812, 244, 0, 0.61, 734 | 809, 245, 0.61, 728 |
| 4:0x03 | S → triforce | 736 | 568 | 564 |

Every overworld wrap is taken on every visit with clips on (0x59, 0x5a, 0x5b, 0x1b, 0x19, 0x18; 0x64 W and 0x65 W on
the way back; 1.00 per visit, 0 hearts); without clips no clip is made anywhere.

**Gleeok by arrival health:**

| | visits | mean arrival (hearts) | ≥ 6: won | 3.5-6: won (died) | < 3.5: visits, won, died |
|---|---:|---:|---|---|---|
| D4, clips on | 928 | 3.27 | 51, 100% | 403, 98% (8) | 474, 61%, 184 |
| D4, clips off | 812 | 2.65 | 21, 100% | 232, 98% (5) | 559, 57%, 239 |
| D4, r71 | 809 | 2.68 | 25, 100% | 246, 98% (6) | 538, 56%, 239 |
| full, clips on | 180 | 3.10 | 10, 100% | 70, 94% (4) | 100, 60%, 40 |
| full, clips off | 78 | 2.69 | 7, 100% | 20, 95% (1) | 51, 43%, 29 |
| full, r71 | 199 | 2.47 | 3, 100% | 49, 100% | 147, 52%, 71 |

Below 3.5 hearts Gleeok still wins about 40% of his fights; 184 of his 192 kills (clips on) come there.

**Bomb walls opened with a real bomb** (visits; opened with real bombs only; with a cheated bomb; bombs per visit):

| wall | segment | visits | real only | cheated | per visit |
|---|---|---:|---:|---:|---:|
| 0:0x2c | after D2 (on = off) | 1000 | 1000 | 0 | 1.00 |
| 0:0x67 | after D1 (on = off) | 1000 | 987 | 13 | 1.01 |
| 0:0x67 | full on / off | 597 / 412 | 506 / 344 | 91 / 68 | 1.02 |
| 3:0x5b E | D3 / full on / full off | 851 / 404 / 246 | 443 / 167 / 113 | 375 / 168 / 84 | 4.2 / 3.6 / 3.2 (darknut bombs as a weapon) |
| 3:0x4c | D3 / full on / full off | 675 / 259 / 147 | 674 / 259 / 147 | 0 | 1.00 |
| 4:0x21 / 4:0x11 | D3 segment on into D4 | 488 / 479 | 286 / 169 | 195 / 310 | 1.62 / 1.00 |
| 4:0x21 | full on / off | 189 / 89 | 102 / 55 | 87 / 33 | 1.64 / 1.58 |
| 4:0x11 | full on / off | 189 / 88 | 58 / 23 | 131 / 65 | 1.01 / 1.00 |
| 4:0x21 / 4:0x11 | full, r71 | 229 / 223 | 28 / 20 | 201 / 203 | 3.65 / 1.00 |

Real bombs now reach Dungeon 4: in the full route 102 of 189 visits to 4:0x21 open it with real bombs only (round
21: 28 of 229), 58 of 189 at 4:0x11 (20 of 223).

**The kill streak and bombs on entering 4:0x12** (D4 segment, from the replays; measure only):

| | entries | mean streak (`$50`) | streak 9 or 10 | bombs held |
|---|---:|---:|---:|---|
| clips on | 955 | 5.37 | 165 (17%) | 0 in every run |
| clips off | 885 | 4.64 | 48 (5%) | 0 in every run |

Streak histogram (0..10), on: 102 58 28 38 89 136 159 117 63 62 103; off: 110 45 15 36 149 197 158 82 45 18 30. The
new route raises the streak (round 21's route, scout's 182 entries: median 3), 0x01's eight Keese coming just
before 0x02. But the D4 segment starts with no bombs, so no Link can bomb his tenth kill there; the full route was not
recorded with replays.

## 9. Problems found and what was not done

Each is in `docs/experiments/backlog.md` "Left open by round 22" unless said otherwise. In the order the chains lose
runs to it:

- **3:0x5b's last pass at 6 or more bombs** (darknut). D3 segment, last visit, by bombs held on entry: 1 → 28 deaths
  in 261 visits (11%), 4 → 3 in 57 (5%), 5 → 48 in 348 (14%), 6 or more → **70 in 170 (41%)**; below 2 hearts with
  6+ bombs 36 deaths in 41. Below 6 the route takes the room's treasure (`if_bombs_below: 6`: kill the Darknuts,
  then take their bombs) and Link survives more often; at 6 or more the step is only "bomb the E wall and leave".
  In the full route Link brings real bombs (85% of its 0x67 visits open the wall with real bombs only), so the case
  is common: 145 deaths in 404 visits (round 21's W → N pass: 39 in 373). This is what costs the full route's
  Dungeon 3 exit: full → after D3 19.5% against round 21's 24.8%, while full → D3 held (59.5% against 56.7%, within
  noise). The single room hides it (3_5bw 97.0% −0.72 over the whole bomb mix, `bombs: [0, 8]`, half the
  resets on the state's own bombs). Next: darknut trained with more resets at 6-8 bombs on 3_5bw, or the treasure
  taken only above a health floor.
- **Gleeok below 3.5 hearts:** 184 of his 192 kills per 1000 D4 runs (clips on); Link wins 61% there. boss-gleeok
  was not retrained (still `hp: [0.5, max]`).
- **Manhandla** (168 deaths per 1000 D3 runs, 25% of visits; 64 per 1000 full runs), **3:0x59** (80 in the D3
  segment, 119 in the full route), 3:0x69 (44): not worked on this round.
- **The plain way's heart cost.** Clips off, Link reaches Dungeon 2 with 2.24 hearts (on 2.63): 0x5b W→E loses 0.40
  hearts per visit (on 0.00), 0x5a 0.10, 0x18/0x19 0.18-0.24, 3-7 deaths each; 2:0x5e then kills 125 full runs (on
  42), full → after D2 68.3% against 86.7%. In Dungeon 4, 0x31 E→W (37 deaths, 0.70 hearts per visit) and 0x30 (25)
  bring Link to Gleeok with 2.65 hearts (on 3.27): 244 Gleeok deaths against 192, 4:0x12 73 against 27. ow-moblin's
  plain 0x5b plateaus at about 92% in single rooms after `off_share: 0.75` (4M) and a retrain from scratch (8M).
- **The second BOMB inside the first bomb's fuse** in 4:0x21 (§5): 1.6 bombs per visit where one opens the wall. A
  mask rule is described; not built.
- **vire loops under argmax in 4:0x12 from N** (single room: 21 + 16 no-damage endings in 600 episodes, 11 + 8 after
  the continuation), so vire samples; it also opens the sub-screen 2.6-6.4 times per episode there.
- **The balanced early stop passes a weak room:** vire (4:0x50 without clips 95.7%) and ow-zora (0x5a without clips
  96%) stopped at 0.99 after about 0.55M steps; the remedies used `--stop-success 0.995`.
- **like-like 4_32w on the rebuilt states and natural RNG:** round 21's model lost 1.26-1.35 hearts there (its own
  evaluation 0.85); 4.6M steps of continuation brought it to 0.89.
- **Small:** 0:0x75 1 stuck run in 1000 (clips on); 0:0x66 3 wrong exits into its shop in 1000 (clips off); 4:0x01
  (dark Keese, the key on water) 6-9 deaths per 1000, the key taken on every visit that left the room.
- **Bomb fishing in 4:0x12** (the owner's aside, "not a directive"): **measured, not built.** The mechanics are
  proven (`asm:HelpDropCount` `$50`, `asm:IsHelpDropBomb` `$51`; one Vire left, `$50` = 9, one bomb, `BOMB S` on it:
  the forced drop was bombs, +4), and round 22's route raises the streak on entry (above: 17% of D4-segment entries
  at 9 or 10 with clips on), but every Link enters 0x12 with 0 bombs: the D4 segment starts without any, and the
  route's reserve of 2 is spent on the two walls. The proposal `bomb_reserve: 3` from Dungeon 4's entrance (one real
  bomb kept for 0x12) is not applied; measure the streak and bombs on the full route with replays first.
- **The full-route figure stays noisy at 1000 runs:** only 180-199 runs reach Gleeok (clips on), so 13.6% against
  12.8% is no difference; compare by segments.

Not done this round: the whole route's states before 0x58 were not rebuilt (§6); mid-stretch start states for a
switch flipped in 0x76 or 0x75's E part (0x66 from S, 0x76 from W) do not exist; the live cell has no sound;
`menu_frames` is still not recorded by the evaluation; the vire YAML's `notes:` still call 0x20 off the route (left
so evaluation configs don't change). The 30 rupees buy nothing on the route.

## 10. Promoted

With `triforce promote F R --act A` after the canonical evaluation in that act mode; each promoted run's `act`
matches its `ckpt.pt` sha1 (checked). New runs carry `version` 1.4.1.1, kept ones 1.4.1.0 (same first three
numbers). No round 21 run dir was changed but for `promoted` and `act`.

| family | promoted (act) | previously (act) |
|---|---|---|
| ow-start | r72-ow (argmax) | r71-ow (argmax) |
| ow-octorok | r72-octorok (argmax) | r71-d (argmax) |
| ow-zora | r72-zora-off (argmax) | r71-ow (argmax) |
| ow-moblin | r72-moblin-off (argmax) | r71-d (argmax) |
| ow-leever | r72-ow (argmax) | r71-ow (argmax) |
| ow-tektite | r72-ow (argmax) | r71-ow (argmax) |
| ow-lynel | r71-ow (argmax) | unchanged |
| ow-boulder | r72-ow (argmax) | r71-ow (argmax) |
| ow-peahat | r71-ow (argmax) | unchanged |
| goriya | r71-goriya-cont (sample) | unchanged |
| d2-rope | r71-a (sample) | unchanged |
| moldorm | r71-c (argmax) | unchanged |
| boss-dodongo | r71-e (sample) | unchanged |
| stalfos-keese-gel | r72-skg (sample) | r71-b (sample) |
| wallmaster | r71-e (argmax) | unchanged |
| boss-aquamentus | r71-e (sample) | unchanged |
| trap-room | r72-trap (argmax) | r71-a (argmax) |
| equipment-room | r71-e (argmax) | unchanged |
| zol | r71-zol-cont (sample) | unchanged |
| darknut | r72-darknut (sample) | r71-darknut-cont2 (sample) |
| boss-manhandla | r71-e (argmax) | unchanged |
| vire | r72-vire-cont (sample) | r71-vire-cont (sample) |
| like-like | r72-likelike2 (argmax) | r71-c (sample) |
| boss-gleeok | r71-d (argmax) | unchanged |

Not promoted: r72-moblin, r72-moblin-new (ow-moblin), r72-vire (vire), r72-likelike (like-like), and r72-ow for
ow-zora (replaced by r72-zora-off).
