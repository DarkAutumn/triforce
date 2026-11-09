# Experiment backlog

Ideas and problems found but not acted on yet, with the round that found them. (Named `todo.md` until Nov 8; older
round docs still call it that.)

## Now (Lee, Nov 8): baseline retrain on d2first

Goal until further notice: a set of specialists that carries the `d2first` chain from the game start to the Dungeon 1
triforce (Dungeon 2 and Dodongo first, the 0x2c heart cave, the white sword; `docs/routes/d2first.md`). The reset
onto it is done (below, "d2-first route" item 2). In order:

1. [x] **Baseline** (round 13, Nov 9): every family retrained from scratch on d2first with the current setup
   (r47; boss-dodongo r49 after fixing its success to the W exit). Full route **25/120 (21%)**; from D2's exit
   29%; D1 from a full-health entry 96%. Losses: D2's Goriya rooms (0x5e, 0x1e) and Moldorm (one-room 91%/85%),
   and health attrition from 0x2c to D1 (no healing on the way). Lynel room ~98% one-room for every spawn.
2. **Next: Lee's training ideas** (to come), measured against that baseline.

Findings from round 13 worth a look (not started): health attrition after D2 (Link reaches D1 with ~2 of 5
hearts); the Goriya/Moldorm rooms; Lynel deaths on the way back out with the white sword; the near-cave Lynel
spawn (95%).

## Next up (history before the d2-first reset)

Goal then: a set of specialists that carries the milestone-1 chain to the Dungeon 2 boss (Dodongo).

- **Where it stands (Nov 6):** the Rust port, coverage and performance steps are done (Nov 5); round 10
  (`round10.md`) has run on the new stack. §1 hardware-core chain baseline (matches FCEUmm at 120 runs);
  §2 item 8, boomerang usable vs masked: no effect, kept usable; §3 swordless departures fixed (ow-start
  r38b promoted: 10/120 → 0/120 per route); §4 key-carriage audit (72/74 Dodongo arrivals keep all keys); §5
  arrival health decides the D2 Goriya room, chain-hp Goriya retested properly: no effect; §6 finite stall
  escape in training and the player; launcher hardening. Every "Work in this order" item is done. Open items
  left need Lee: power-on RAM fill, the direction for arrival health, and whether to port observation schema 2
  to Rust for the scratch-team question.
- **Promoted now:** ow-zora r24-fairy, ow-start r38b-rebalance (round 10 §3), ow-octorok r32-nearby-off, ow-moblin r26c-moblin,
  ow-leever-peahat r23-more, boss-aquamentus r30-exit, stalfos-keese-gel r25-skg-plain, goriya r31-block1,
  d2-rope r35-east-curriculum, wallmaster r10 copy. Overworld families use both routes' rooms; D2 rooms have 3 states each.
- **Lee's order (Nov 5):** first finish the Rust port (plan steps 17–18: `round8.md` "Open", the watch-mode
  restore below, then retiring the FCEUmm oracle and quirks). Then **raise test coverage** sensibly across the
  emulation core, the Rust crates and the Python wrappers: a robust unit suite plus a few end-to-end tests, no
  PPO training from scratch, suite stays fast. Then a **performance and optimization** step:
  baseline steps/s from the training runs in `logs/` (what `training-gui.py` shows), profile training on the
  new stack over the long haul (not startup), make it faster for the same number of envs, and write
  `docs/performance.md`. **Then continue with the experiments below** (the "Work in this order" list, then
  "Other open items"), as before: log every result, commit each finished item, keep going. Side experiments
  found during the port or the performance work go into "Other open items" instead of derailing the work.
  (Port, coverage and performance steps: done Nov 5.)
- **Long jobs (Lee, Nov 5):** a state build, explore or chain hunt can hang (one hung on Aquamentus before).
  Check any such job's progress at least every couple of hours and intervene early when a single state or room
  makes no progress (retry with another seed or source, record it, move on); **12 hours on any single task is a
  hard stop**: kill it, log it in the round doc, and continue with other work.
- **Lee's order (Nov 7):** first **optimize the Rust emulation core** (done Nov 7, below); then go straight back
  to the **heart experiments** (below), a series, not one experiment.

### Done: zcore speed (Lee, Nov 7)

Make emulation faster with **perfect accuracy to the game**: nothing may change, checked before/after with the
corpus goldens (`tf ladder-golden` / `ladder-diff`, 0 divergences), the start states and trajectories walked with
varied actions, and the port gate. Free rein: profiling tools may be installed, and the architecture of the port
(a piece-by-piece cut of TetaNES) may be rethought and redone. Profile the agent too (where training spends its
time end to end), not just the emulator. Iterate until further gains aren't worth the effort; every bit counts
because the real runs will be 6–12M steps per family (the old monolithic model took ~30M). Log it in
`docs/performance.md`.

**Exit criterion (Lee, Nov 7): the right `--parallel` for training.** Find where adding envs stops paying, from
two sides: (1) pure throughput (steps/s against env count); (2) convergence: more envs can waste steps, so
measure the steps and the wall time to reach a working level (around 90% success, not 99%) against env count.
Recommend a setting (per specialist if it differs).

- [x] **Done (Nov 7; `docs/performance.md` "zcore speed" and "How many envs").** Idle-loop skip, per-opcode
  dispatch, sprite-range cache, byte-lane tile rendering, smaller fixes (`bd71234`…`76272ea`), every one through
  the accuracy gate (goldens 0/63, 481-state walk 0 differ, seeded training digest identical). Env benchmark
  2,832 → 4,526 frames/s (1.60×; per family 1.38–1.94×); training steps/s +30–33% over Nov 5 at the same env
  counts. The zcore profile is now flat (nothing above ~5%); three further attempts gained nothing.
  **`--parallel 16`** for a run alone (`train_round.py`'s default), 8 each for two rounds side by side:
  throughput rises near-linearly to 10 envs and flattens at 16–20, and the steps to reach 90% success from scratch
  don't depend on the env count (ow-leever-peahat, ow-moblin; 4–24 envs, 3 seeds, 30/30 reached), so wall time
  is lowest at 12–16. Same for every specialist measured; no per-specialist setting needed.

### Done (stopped): power-on RAM $00, regeneration and a fresh retrain (Lee, Nov 8; round 11)

- [x] $00 fill, state version 6, states and trajectories regenerated (round 11 §1).
- [x] Moldorm room 2:0x3e split out of d2-rope into its own specialist (`moldorm`).
- [x] Per-room-visit evaluation (format 2, 100 episodes per start state) and `zelda-map.py`.
- [x] Fresh retrain `r44-fresh` **stopped by Lee** during wallmaster: the next full retrain comes after Dodongo,
  on the d2-first route below. Finished runs (ow-start, ow-octorok, ow-zora, stalfos-keese-gel, goriya) kept,
  not promoted.

### Done (paused by Lee, Nov 8): beat the Dungeon 2 boss, Dodongo (round 12)

Two ways to kill Dodongo: (1) make him swallow two bombs (works, wasteful); (2) **set a bomb in his path but just
next to him, so he doesn't eat it and the smoke stuns him, then hit him with the sword while stunned (much
better)**. Train a model for (2).

- Bombs (Lee, Nov 8, replaces "infinite bombs"): counts stay natural, but Dodongo's room is `requires_bombs`
  (game.yaml): whenever Link has none there he gets one *cheated* bomb (start, entry, after every action); placing
  it counts as a cheated bomb used, leaving with it unused takes it back. Training, eval (per visit) and chains
  report `bombs used` and `bombs cheated`; the watch player's footer shows a run's cheated bombs (not beaten
  naturally). `crates/env/src/bombs.rs`, docs/architecture.md "Bombs".
- A whole task of its own: expect plenty of experimentation and trial and error.
- **The curriculum matters most:** it has to show how to do it against Dodongo's various movement patterns.
- Mechanics (ASM-backed): `target/dodongo/README.md` (gitignored notes) and round 12.
- [x] **Solved:** r45-dodongo (promoted) kills him in 300/300 eval fights, 0.13 hearts lost per fight, 86% by
  bomb-stun + sword (67% with no swallow at all), 52% with one bomb, 2.5 bombs per fight. r46 (2M more steps;
  swallow penalty 0.5 → 1.0) no better. Details: round12.md §4–§5.
- **Paused by Lee** (Nov 8): the training changes coming next will change how this is trained. What I would have
  done next to cut bombs per fight (the metric to minimize; 1 = perfect), in order:
  1. **One-bomb curriculum.** Explore many more seeds (`python -m triforce.explore --specialist boss-dodongo
     --seed S --variant --out <root>` for S in 10..40+), keep only tracks with exactly one BOMB and no swallow
     (`target/dodongo/how_killed.py <root>/boss-dodongo` prints BOMB count and max BombHits per track), and
     train from scratch on those against r45 (same budget). Today's 25 tracks spend 1–13 bombs (random
     placement), so the demonstrations themselves are wasteful. Go-Explore could also rank successes by bombs
     used (fewest) before health/steps for this specialist.
  2. **The bomb in the observation.** Link's block has only `bomb_active`/`bomb_exploding`; the bomb's position
     relative to Dodongo and its fuse time aren't visible, so the model can't judge "about to be eaten" vs "about
     to stun" once placed, nor see that a second bomb (slot $11) is useless to Dodongo. Needs an observation
     schema change (bomb x/y relative to Link, fuse fraction, which slot).
  3. **A per-bomb penalty knob** (`penalty-bomb-used` is a fixed −0.25), raised for this room only. Doubling the
     swallow penalty didn't help (r46-eat1), so reward size may not be the limit; try after 1–2.
  4. **Measure the ceiling:** the fuse (48 frames) spans about one of Dodongo's partly random direction decisions
     (every 16 px at 0.5 px/frame), so some misses are luck. A scripted placer using the probe geometry
     (`target/dodongo/geometry.py`) over many seeds would show the best achievable one-bomb rate.
  5. Low-health deaths: r46's few failures were deaths at 0.5–1.5 starting hearts; watch this in d2-first chains.
  6. Small: a stun that ends and restarts inside one step is missed by the counter (2 of 300 records).

### Done (Lee, Nov 8): the d2-first route, mapped and reset onto

1. Prove Dodongo can be beaten (above): r45 kills him 300/300 (round 12).
2. [x] **The d2-first route**, mapped first so Lee could review it on the map (merged `ac03202`). Start → 0x58,
   east along row 5 → at 0x5d north → 0x3c (Dungeon 2); never the fairy. D2 as before (3 max hearts), skipping
   the magical boomerang (2:0x4f); kill Dodongo, take his heart (4), take the triforce in 2:0x0d (warps out to
   0x3c). Then round to 0x2c, bomb the secret rock for the heart (5), and `1c -> 1a -> 0a (white sword) -> 1a -> 17
   -> 27 -> 28 -> 38 -> 37` into Dungeon 1 (both key rooms, wooden boomerang pickup kept) with the white sword;
   the route ends on taking the D1 triforce (6 hearts).
   - [x] **Reset done (Nov 8, `docs/decisions.md`):** d2first is the only route; milestone1/milestone1-fairy, the
     fairy waypoint and ow-leever-peahat removed; 0x2c to ow-tektite; route-step objectives (D1 key rooms in
     order); `retreat` at 0x0a; every state rebuilt (chains 0–2) and every curriculum re-explored.
   - [x] **Baseline retrain done** (round 13): full route 25/120. Lee's training ideas next.
3. Lee's ideas for improving training (to come, after the baseline).

### Done (report to Lee): heart experiments (Lee, Nov 7; a series)

Goal: Link picks up dropped items, hearts above all, far more than the current models do. Lee's original models
didn't take every item, but prioritized them far more. **This is solvable**: iterate until it works by the
measure below, and only when out of reasonable ideas stop and report.

- **Measure:** real health gained from pickups (`health_change` and hearts healed per visit/episode, not `total`,
  which counts overhealing) is the main metric of this work, not overall success. Current health dominates
  everything (Lee, speedrunner): on average +1 heart wins more often in the long run, even though 2M steps and a
  few evaluations can't show that statistically in success rates. Report pickups taken/left/healed alongside.
- **Starting ideas (Lee, Nov 7; try them all, none mandated, combine or replace as results suggest):**
  - A fixed, rigid pickup reward (the current max step reward, `REWARD_MAXIMUM`) instead of today's 0.25–0.75
    urgency-scaled `reward-gained-health`.
  - Direction and distance to the nearest target (exit or item) in the observation: "a little bit cheating, but
    almost certainly a big difference". A new observation schema, so every model retrains.
  - Every dropped item is a shaping target, as in Lee's original (hearts, clocks, rupees, bombs, keys, fairies:
    no dropped item is worthless), with hearts **not** a target while at max health. The model already sees
    `hp_full` and each dropped item's kind and position (round 10 Q&A, Nov 7).
- **What failed before and why it may have** (round 6 §10 `collect_hearts`; round 7 §9 r32 `nearby_items`, which
  is Lee's design for hearts/clocks): both were short continuations of models already trained toward exits, drops
  are rare in most families' training, the observation has no pointer to the target the shaping pulls toward, and
  r32 pulled toward hearts even at full health. Train from scratch (or long enough) where the treatment needs it.
- **Where it pays** (round 10 §5): d2-rope leaves 0.36/0.39 hearts per visit on the floor in 2:0x7e/0x4e, around
  the Goriya room whose deaths are decided by arrival health.
- [x] **Step 1 done (r40, round 10 §8, Nov 7):** all three ideas tried on d2-rope from scratch (2 seeds each).
  Item targets give the pickups (hearts left while hurt −70–80%); the fixed reward helps alone but adds nothing on
  top. The target observation (schema 2, merged) turns them into health: d2-rope heals 1.2 hearts per D2 run
  (was 0.45) and leaves almost none while hurt, eval ΔHP −0.05 (was −0.26), Goriya arrival 3.0 hearts (was 2.1).
  The pointer alone gives success and less damage but no pickups. Goriya widened from the promoted model with
  both: fewer deaths. **Promoted** d2-rope `r40-obs-s41` and goriya `r40-goriya-obs-s41`: Dodongo arrivals per
  120 runs D2 44 → 74, post-D1 12/18 → 28/56, full routes 3/6 → 16/29.
- [x] **Step 2 done (r41, round 10 §9, Nov 8):** every other schema-1 family widened with both, 1M steps, 2 seeds.
  Promoted stalfos-keese-gel `r41-obs-s41`, wallmaster `r41-obs-s40`, boss-aquamentus `r41-obs-s40` (D1: more
  health at the Goriya room 1:0x44, Aquamentus deaths 12 → 8; full main 16 → 21). The overworld families
  (ow-zora, ow-leever-peahat, ow-moblin) improve in evaluation and on the main road but lose health after the
  fairy pond; ow-octorok and ow-start lose arrivals. Not promoted.
- [x] **Step 3 done (r42/r43, round 10 §10, Nov 8):** the overworld families with the pointer alone (no item
  targets): all four better, together post-D1 28/56 → 43/66 and full 21/27 → 26/34; promoted. ow-start pointer
  only: promoted (neutral to positive, no swordless departures). Dungeon families keep item targets (pointer only
  lost for stalfos-keese-gel and wallmaster). **Series result** (§10 table): hearts healed per run 0.45–0.59 →
  1.2–2.0 on every segment, hearts left while hurt in D2/post-D1 0.6–0.9 → 0.07–0.17 per run; Dodongo arrivals
  D2 44 → 74, post-D1 12/18 → 43/61, full 3/6 → 27/46 of 120.
- [ ] **Remaining ideas (not run):** (a) more drops in training (`$52A` drop cycle) for families that rarely see
  a heart; few drops on the route are left on the floor while hurt now except on the full routes (0.33–0.36 per
  run). (b) The Goriya rooms remain the chain's main killer (1:0x44, 2:0x5e), not a pickup problem. (c) The fixed
  pickup reward was never tried with the pointer (it added nothing on top of item targets without it).


### Work in this order

From Lee watching `play-zelda.py` after round 6 (Nov 3), sorted: first the measurement everything else needs,
then clear bugs, then what decides how the next full retrain looks, then reward/target changes aimed at the
chain's death rooms, then the big ablation. Items 1–7 and the versioned observation comparison are complete;
the remaining list is deferred until Lee resumes work.

1. [x] **Track lingering (Lee: a standing requirement, see AGENTS.md).** Measure how long Link stays in a room
   without progress, in training, eval and chains, and investigate whenever it's long, even if success looks fine.
   Progress = a kill, a hit on an enemy, an item/key/treasure taken, a new best wavefront distance to the
   objective, leaving the room. Record per episode / per chain room: the longest no-progress streak and the steps
   after the last progress; report per specialist and room (training metrics + progress events, `eval.json`
   `by_room`, `RoomRecord`, a GUI chart). Note monsters can chase "shadow Link" (AGENTS.md, ram-catalog
   `$60–$62`), so a still Link in a corner can produce long fights with no hits. Item 6 (cutting such runs) builds
   on this metric.
   - **Done, round 7 §1:** behavior-neutral shared tracker, seeded gameplay parity, 383 tests, CUDA training events,
     real wide/narrow GUI and historical-data checks. Tracked evals and all 120 baseline chains complete.
2. [x] **Aquamentus wanders after the kill (Lee).** Link kills Aquamentus, then walks left/right in 0x35 for a very long
   time before leaving; one run sat there "for ages" and only got unstuck by chance. Even with a good success rate
   something is wrong, and Lee says it's unrelated to shadow Link: check it, or penalize hanging out in the boss
   room once the boss is dead. Likely cause: `boss-aquamentus.yaml` succeeds on `boss-dead-and-heart` (the episode
   ends when the heart container is taken), and its start states are all the fight entry, so walking out E into
   the triforce room 0x36 afterwards is never trained. The runner then gives the room up to 2000 steps
   (`runner.MIN_ROOM_STEPS`). Check: steps between the kill and the exit in chain records / a replay (item 1's
   metric). Fix options: success = exit-correct into 0x36 (with curriculum states after the kill), a lingering
   penalty once the boss is dead, or hand the room to stalfos-keese-gel once the boss is dead. Compare chain time
   and hearts at the triforce room.
   - **Round 7 §2 diagnosis:** old r23-more 60/60 to the heart but 59/60 through the exit; a reproduced seed
     lingered 1,963 actions after pickup. Both full chains had two post-heart 2,000-action timeouts.
   - **Done, round 7 §5:** promoted r30-exit and complete-visit curriculum. Exit-task eval 60/60; mean pickup-to-exit
     11.1 actions (old 92.0), worst 77 (old 1,963), health +0.69 (old +0.61). Both full chains removed old post-heart
     timeouts. Independent 20-run sweeps: full main/fairy 1/3, post-D1 main/fairy 0/4, D2 3; no execution errors.
3. [x] **Observation audit (Lee: does Link get enough about enemies and the world, e.g. does he know an enemy is
   stunned?).** Do it before the next full retrain: observation changes invalidate every model, so bundle them.
   Already there (`observation.ENTITY_FEATURES`): per-slot type embedding, projectile, blockable (rock, arrow,
   boomerang; not the Zora fireball), hp, stun, invincibility, immunities, spawning, dying, active, item lifetime.
   Checked: ordinary boomerang stun is 16 long-timer ticks (one per 10 frames), so the existing `/16` spans it
   correctly; no rescaling needed. Link's own invincibility frames and shield state;
   Link's boomerang in flight vs returning; which enemy threw a projectile; clock active; the monsters' chase
   target (`$60–$62`: Link or shadow Link), which tells where they are heading; anything v1's observation or the
   RAM catalog (`docs/ram-catalog.md`, entities) has that v2 lacks. Output: a list of proposed changes with reasons;
   decide, then implement with a full retrain.
   - **Done, round 7 §3:** source-backed audit and live boomerang/stun probe. Returning boomerang animation bounds
     had a real hex/decimal bug; fixed and regression-tested. Keep schema 1 for isolated experiments.
   - [x] **Bundled observation trial before the from-scratch item 8 study:** versioned chase target coordinates,
     Link/monster boomerang position/phase and corrected harm semantics. Completed r36 (§21): twenty matched
     fresh models, 12,400 evaluation episodes, 800 native chains, zero evaluation/chain errors. No cutover:
     Zora success 55.5% → 38.4%; bundle full routes stall at 0x48 in 79/80 runs each. D2 requested-key arrivals
     tie 7/80, full three-key arrivals 5/80 → 7/80. Keep current promoted schema-1/r35 team; schema 2 stays in dev.
   - **Full-study input prerequisite complete (§20):** retired six old Moblin curricula (297 snapshots) with
     three containers/no boomerang/no triforce in post-D1 rooms; preserved the valid four-heart 0_3dc path and
     installed verified current 0_3dn_2 coverage. All 14 Moblin goal replays and 41 default-track route audits pass.
     D2 Goriya replacements and all 21 Rope tracks are installed; r36 freezes all 62 enabled tracks.
   - [ ] **Temporal world snapshots (later; Lee, Nov 4).** Consider the current structured world observation
     plus 1–4 previous observations, so motion/changes are explicit to the policy. Full world-history stacking
     is not implemented: current inputs contain per-entity `vx/vy` estimated from the previous observation,
     plus last-action/timing/interrupt history, not previous world snapshots. Account for variable elapsed
     emulator frames and reset/mask history across episode/room changes and newly occupied object slots.
   - [ ] **Enemy facing versus movement (later; Lee, Nov 4).** Facing is already supplied as per-entity
     `facing_N/S/W/E`; observed movement is also supplied as `vx/vy`. Review their decoding/usefulness rather
     than assuming direction is absent. Facing is not necessarily current motion (e.g. stun or knockback).
   - [ ] **Distance plus unit direction from Link (later; Lee, Nov 4).** Lee's original Triforce supplied an
     enemy's distance and normalized direction vector, doing that geometry outside the network. Current
     inputs already include absolute `x/y` and Link-relative `dx/dy` scaled by room axes, but neither an explicit
     scalar distance nor a unit-length direction vector. Compare that explicit representation; use a consistent
     pixel metric and define the zero-distance case. These ideas are deferred, not added to frozen r36 inputs.
4. [x] **Goriya rooms: blocking, and the corner stalemate (Lee).** v1 rewarded the shield-block sound, and that taught
   Link to block instead of face-tanking. v2's old `critics.critique_block` rewarded ArrowDeflected sound onset;
   round 7 replaces that incomplete signal with per-frame confirmed shield collisions and per-collision decay.
   First count confirmed Goriya blocks per episode in evals and traces, then A/B a stronger
   block reward on goriya (0x44 is the chain's top death room). Caveat (Lee): the sound alone is wrong in rooms
   with Darknuts etc. (their shields clink too), so require a blockable projectile in the air near Link plus the
   clink. Lee also watched the stalemate: Link in the top-left corner not fighting while the Goriyas chase shadow
   Link in the bottom right (AGENTS.md "Game mechanics"), no hits on either side, for ages. The stalling penalty
   (−0.01 to −0.02 per step after 150 quiet steps) is apparently not enough to break it; measure it with item 1,
   then consider a stronger penalty for quiet steps while enemies remain. Related, from round 6: at 1 heart the
   goriya model turns passive (33% WAIT) and dies to contact; other ideas are demos or curriculum from damage-free
   Go-Explore paths (select by health, more patience) and a contact-damage penalty that grows as hp falls.
   - **Round 7 §6 prerequisite done:** natural Goriya trace credited only 1/7 real blocks before, 7/7 after at
     identical seed/action positions. Executed-frame counts reach training/eval/chains. 447 tests and CUDA smoke
     passed; schemas unchanged. All r31 arms and 800 frozen-team chains completed without execution errors.
   - **Measured baseline:** 178/240 successes, −0.96 hearts, 45 confirmed blocks. Faithful 0x44 seed-62 replay
     has three living Goriyas throughout a 1,971-action quiet stretch; existing stall cost does not break it.
     Ordinary r31-block1 promoted for lowest evaluation damage and mixed but better full-main/D2 chains; stronger costs off.
   - **Distinct pickup problem:** 0x23 seed 5 kills all three Goriyas, then wanders for 1,902 quiet actions without
     the key. A fight-only penalty cannot address this; do not label it shadow-Link stalling or silently exclude
     the room from family evaluation.
   - [x] **Seed prerequisite, round 7 §7:** corrected CLI-to-PPO-to-worker policy/reset seeding in dev. Native
     no-curriculum rollouts reproduce exactly with the same seed, differ with another seed, and advance across
     episodes/batches. CPU CLI, CUDA smoke and all 462 tests passed; committed cf65aa2.
     Shared curriculum scheduling remains asynchronous; r31 is budget/config matched, not paired-seed causal evidence.
   - **Remaining navigation problem, round 7 §6:** faithful r31 chain replays find no enemies during the final
     1,000 stalled actions in D2 0x5e/0x1e (MOVE objective). Quiet-fight cost cannot fix post-kill exit pacing.
     Preserve this distinction in later cutoff/curriculum work; do not call it shadow-Link fighting.
5. [x] **Hearts: aim at the nearer of heart or exit (Lee: "a big deal", v1 behaviour).** Agents leave many dropped
   hearts (round 6 §8: overworld 26 of 34). In v1, dropped items were among the wavefront's targets, and the
   wavefront is a multi-source BFS, so PBRS led to a heart whenever it was closer than the exit by walking distance
   (not straight-line) and ignored it otherwise. v2 adds items to the objective's `targets` but not to
   `pbrs_targets` (`objectives._GameMapObjective.get_current_objectives`), so shaping never points at a heart.
   Try: hearts and the clock (Lee: "certain items like the clock and hearts") in `pbrs_targets` alongside the
   exits. This differs from the dropped `collect_hearts` (heart was the only target whenever Link was hurt, kind
   ITEM). A/B on stalfos-keese-gel, d2-rope and an overworld family; measure taken/left, real vs total heart
   change and the chain.
   - **Done, round 7 §9:** optional `reward.nearby_items`, default off; real walker verified nearer
     clock/heart pickup versus nearer exit, correct exits and actual healed/left counts. Original r32 evaluations
     and 1,840 chain executions finished without errors (includes repeated paths). Disjoint D2 seeds reverse
     the initial ranking: shaped Rope leads combined success/health, but full chains remain mixed. Moblin has
     zero eval drops and its off arm harms the fairy chain. Octorok arms finished with sparse heart exposure and
     no pickup gain. Prerequisite fixes through round 7 §14 merged at 4cb8434 (470 tests). All 1,200 corrected
     runtime chains finished: shaped Rope improves D2 but regresses both full routes; keep old Rope/SKG/Moblin.
     Ordinary Octorok off promoted after disjoint 80-seed comparisons improve combined success/physical health
     on all four segments. No shaped model promoted; nearby shaping stays off. Matched cutoff trial next.
6. [x] **Cut runs that stop making progress (Lee: in v1 a run was truncated after ~100 steps without progress, e.g.
   an item pickup or a kill).** Today training ends `stuck` after 50 steps without moving and `no-progress` after
   2000 steps without a new correct room (`end_conditions.Timeout`, the same 50/2000 as v1's Timeout; the ~100-step
   rule isn't in v1's code, so define it fresh). A room-stalling penalty starts after 150 steps, and the runner
   allows `max(max_steps, 2000)` steps per room. Worth trying.
   - Pros: wandering episodes end early, so more episodes per iteration (the ow-moblin and Aquamentus `max_steps`
     cuts both helped a lot); chains stop spending minutes and hearts in a room the model can't leave; cleaner
     failure signal and a "stuck" count per room.
   - Cons: some rooms need waiting (the fairy pond refill, shutters, Leevers and Zoras that can't always be hit,
     Link hiding in a Goriya room's corner until they come; Lee: that's fine while enemies can still reach him).
     "Progress" has to count more than kills and pickups (hits, damage dealt, a new best wavefront distance to the
     objective, a door, key or treasure), or slow but correct approaches get cut. In the chain a cut is a failed run.
   - Design: progress as in item 1 (kill, hit, item/treasure/key taken, new minimum objective distance, room exit).
     N of 100–200 steps, after measuring the longest no-progress gap inside successful episodes per family (item 1's
     metric). Training: a truncation (value bootstraps), not a hard failure. Runner: `stuck` after N steps without
     progress instead of the flat 2000.
   - **Control merged, round 7 §10:** `no_progress_steps` remains default null; measured truncations now
     bootstrap final-state value without crossing resets. Native smoke and integration verified; merged 4cb8434.
   - **Measured cutoff risk:** successful baseline chain rope visits include three quiet gaps ≥200 (worst 2,410).
     Standalone Goriya 0x23 would lose 12/46 successes at 200, versus 1/44 in 0x44. No blanket threshold yet.
   - **Prerequisite defect corrected:** PPO had masked bootstrap on every truncation and discarded its final
     observation. The measured-cutoff path now adds its discounted final value before reset, preserving legacy
     max-step/death behavior and raw logged reward; regression covers batch edges and reset isolation.
   - **Expanded evidence:** independent r30 chain seeds expose successful quiet gaps of 670 actions (Stalfos),
     350 (Goriya) and 2,410 (Rope). No blanket 100–300 cutoff. Moblin's inspected successful maximum is 63;
     100 is a trial candidate, not a general safety guarantee. Route-paired visits are not independent samples.
   - **Current-policy screening:** Goriya has 10/8 failed episodes reaching 100/200 quiet actions, versus none
     in current SKG/Rope/Moblin standalone baselines. R33 Goriya off/100/200 training is active (round 7 §15):
     1,003,520 additional actions per arm from r31-block1, matched seed/budget/config, corrected bootstrap;
     common uncut/actual-cutoff evaluation and config-preserving chains. No canonical cutoff enabled.
   - **Live old-policy risk measured (round 7 §15):** all 400 actual-cutoff chains complete without errors.
     Both limits lose two post-D1 fairy successes; D2 loses three at 100/two at 200 (40 seeds each). Faithful
     seed-9 replay has full HP, open north, no enemies and 200 MOVE actions: post-kill exit pacing, not chase
     waiting. Do not select on shorter failed episodes or enable a blanket cutoff.
   - **R33 complete, round 7 §15:** all three arms/eight evaluations and 1,400 fresh common/actual chains
     finished without execution errors. 200's standalone gain does not carry into both full routes/D2; all
     new models worsen D2 Goriya damage. Keep r31-block1, canonical cutoffs null; no r33 promotion.
   - **Required-shutter prerequisite corrected in dev (round 7 §13):** north tiles 0xa8–0xab were misclassified;
     0x2e wrongly requested MOVE through its closed kill-all shutter. Live route-gated FIGHT now clears it;
     optional 0x6e/0x6d combat stays optional. Four targeted and all 468 tests pass; natural last-Rope → open
     shutter → north-exit smoke succeeds in 69 actions. Merged into main at 4cb8434.
   - **Terminal-health prerequisite complete, round 7 §14:** legacy room/run budgets left HP at entry health.
     One finisher captures live health on every finalized path; death stays zero. Two native fairy regressions,
     real healed-HP smoke, 16 runner checks and all 470 tests pass (merged 4cb8434). Old/off/on Rope reranking is
     complete on corrected runtime (§9); raw historical HP is retained, not retroactively imputed.
7. [x] **D2: east first for the 0x7e key (Lee).** From the D2 entrance go 0x7d E → 0x7e (kill the ropes, take the key)
   → N 0x6e, instead of 0x7d N → 0x6d → E 0x6e. Slower, but the key is needed later in the dungeon (past the
   milestone). `game.yaml` already declares Rope ×5 and a key in 0x7e; generic FIGHT/TREASURE objectives apply.
   The route currently treats it only as an alternate. Make it canonical on both routes, retrain d2-rope, and
   rebuild downstream states carrying the extra key, including S-entry 0x6e (`2_6es_0..2`).
   - **Pickup guard verified in dev (round 7 §12):** old builder physically accepted zero-key `exit-wrong` entry.
     Discovery/replay now require the source key's pickup flag; bombs are masked. All three native sources and
     saved payloads carry the key; 158 affected tests pass. Merged 4cb8434. Route/state migration pending.
   - **Inventory balance guarded in dev:** existing chains 0/1 skip 0x4e/0x3e keys; chain 2 also spends one on a
     side door. Builder now preserves carried keys plus required pickup, allowing only the target door's cost;
     native conserved-key, locked-side-exit and key-pickup paths pass. Full suffix audits remain required. Retire
     stale W-entry 0x6e states, create missing 2_0es_2,
     and refresh affected Rope and D2 Goriya trajectories before curriculum use.
   - **Dev preparation complete, round 7 §16:** all 24 downstream payloads have native nine-load audit and
     conserved keys; stale 2_6ew states/rows retired, 0x6d retained off-route with correct origin progress.
     Native preparation passed all 470 tests; refreshed paths and reset fix are verified below.
     Old policy is 0/60 at newly eastbound 0x7d: keep main unchanged until dev retraining/selection finishes.
   - **R34 completed, rejected (§17):** ab9bda5, r28 warm start, seed 34/eight workers, 2,007,040 extra actions,
     unchanged reward/HP/bombs/one_room and no curriculum treatment. Still 0/60 at the eastbound entrance;
     319/420 overall versus old 335/420. Main remains compatible/unchanged; retry with verified curriculum.
     All 21 refreshed Rope and six D2 Goriya paths succeed; 18 affected Rope tracks/2,174 snapshots pass loader
     and native-exit smoke. Retained Goriya scores 181/240; no Goriya model change in this Rope retrain.
   - **Reset prerequisite fixed and verified in dev (§18):** all 3,303 raw snapshots are playable, but old
     idle gain-control frames killed one live Goriya boundary. Controllable curriculum loads now preserve the
     first action; ordinary pre-entry loads are unchanged. Native regression, all 27 loader/end-action tracks,
     zero dead loaded boundaries, cave/default-run parity, CUDA smoke and 471 tests pass. Do not discard valid data.
   - **Both common-task chain teams complete:** 400 runs, zero east-first prefixes/requested-key arrivals;
     simple old/R34 Dodongo arrivals 2/1, 4/3, 4/3, 2/2, 14/10 on full main/fairy, post-D1 main/fairy and D2.
     Those are old-shortcut arrivals, not support for the new route. R34 was not promoted or activated.
   - **R35 completed/promoted (§19):** same r28 start, seed 35/eight workers/CUDA, 2,007,040 added actions;
     verified backward curriculum learned the eastbound entrance (60/60), overall 400/420, −0.205 real hearts
     versus old 335/420/−0.338. All 131 native D2 arrivals depart east; all 34 Dodongo arrivals carry the 7e key.
     Both routes, 24 audited suffix states, 21 Rope/six D2 Goriya curricula and the promoted pointer are active
     in main at source 47788f7. Canonical CUDA training, six native main smoke runs and 471 tests pass.
8. [x] **Boomerang after D1: does it help or hurt? (Lee: big but worth it; "get to the real bottom of this".)** Link
   picks the boomerang up in 0x44 (that stays) and uses it often afterwards. Question: from D1 complete (`0_37c_2`,
   back on the overworld) to the Dodongo room, are the models better with the boomerang usable or masked?
   - Arms: A = boomerang on; B = `items: {boomerang: off}` for every specialist that plays after D1: ow-octorok,
     ow-zora, ow-leever-peahat, ow-moblin, ow-start (pond), d2-rope, goriya (D2 rooms). The overworld families
     also play pre-D1 rooms, where Link has no boomerang anyway.
   - Training: train both arms the same way. Preferred: from scratch with the same budget and seeds (a warm start
     from today's boomerang-using models favours A). Cheaper first pass: both continue from the promoted models
     with equal budgets.
   - Evaluation: each specialist on its post-D1 states (success, real/total heart change, steps), then chains
     with carry-over: `--from 0_37c_2` on both routes plus the D2 segment, at least 80 runs per arm.
   - **Per-arm chain config prerequisite implemented in dev, round 7 §11:** repeatable `--specialist-config YAML`
     carries loaded family configs into direct/spawned chains. Native boomerang-mask/cutoff smoke and all
     464 tests passed at that step; merged into main at 4cb8434.
   - Run it after items 1–7 settle (or as part of the full retrain item 3 may force), so it measures the setup we
     keep. Expect about a day of GPU.
   - **Done, round 10 §2 (r37):** warm-started first pass, two training seeds per arm, 720 chain runs per arm.
     No detectable effect (chains 18.2% masked vs 16.7% usable, p = 0.49; post-D1 evals slightly favour usable);
     training-seed swings exceed the arm difference. Kept the boomerang usable; nothing promoted.
   - **Open follow-up: the from-scratch comparison** the item prefers. r36's scratch teams failed every chain at
     2M steps, so each family needs a much larger budget: roughly 7 families × 2 arms × ~5M steps ≈ 70M steps,
     ~20 h at ~1,000 steps/s per run with two runs side by side. Worth it only alongside a planned full retrain.

### Other open items

- [x] **Rust port: restore main's watch mode on the native pool** (`1830be3`).
- [x] **Rust port: finish plan step 17** (`round8.md`).
- [x] **Chains on the hardware core and regenerated states (round 9 §18b).** Round 10 §1: at 120 runs per
  segment the team matches the FCEUmm rates; the round-9 dip was 40-run noise. No fine-tune needed.
- [ ] **Power-on RAM fill (Lee to decide).** zcore powers on with $FF (FCEUmm's fill); real RAM is indeterminate.
  Twelve TetaNES test ROMs only match their screenshots with $00. Switching changes the base state (regenerate
  states and trajectories again).
- [ ] **Scratch-team convergence and Zora stalls (r36 §21; deferred).** At the matched 2,007,040-action budget,
  both fresh teams fail all full/post-D1 chains. Bundle Zora is worse in success, health and mean quiet time;
  its full-route 0x48 stalls last 2,000 actions. Baseline post-D1 fairy has 58/80 stalls in 0x59; baseline D2
  worst quiet gap is 2,978. Separate convergence/coverage from representation effects before drawing a general
  observation conclusion. One training-seed pair is not seed-robust evidence. No further run started.
  - **Prerequisite (Nov 6):** schema 2 exists only in the old Python dev tree (`round4-dev` `4d35155`); the Rust
    env is schema 1 (`OBS_SCHEMA_VERSION = 1`). Answering this means porting schema 2 into `crates/env`, then
    two scratch teams at a larger budget (roughly 10 families × 2 arms × 5M steps, a day of GPU). Lee to decide
    whether schema 2 is worth porting (r36: no better, Zora worse).
- [ ] **Arrival health is the D2 loss (round 10 §5).** 2:0x5e kills 24/28 arrivals at ≤1 heart and 4/130 above 3;
  post-D1 main reaches it with 1.64 hearts (fairy 2.39) after ~2.3 hearts of overworld damage spread over
  0x2b/0x3d/0x4c/0x2a/0x2d, with almost no drops there. Already ruled out: crossing-room damage penalty (round 6),
  heart objective (round 6), nearby-item shaping (r32), chain-hp Goriya (r39). Untried: a Goriya skill that works
  without beams (e.g. boomerang stun then melee, demonstrations), and d2-rope collecting the hearts it leaves in
  2:0x7e/0x4e (0.36/0.39 per visit) by some other means than shaping. Needs Lee's direction before more GPU.
- [x] **Audit full D2 key carriage separately from milestone entry (r36 §21; deferred).** Both fresh teams
  reach Dodongo with the requested 7e key 7/80 times, but only 5/80 baseline versus 7/80 bundle arrivals satisfy
  the full-three-key audit. Keep both counters; do not silently count all milestone arrivals as full key carriage.
  - **Done, round 10 §4 (`8456c51`):** chain records carry `keys_in/keys_out`; `chain_summary` reports
    `arrival_keys`. Promoted team: 72 of 74 Dodongo arrivals carry all three keys (two fairy-route arrivals have two).
- [x] **Swordless departure from the starting area (Lee, Nov 4).** Lee observes runs where Link leaves without
  taking the sword; these runs cannot win and waste training. Reduce how often this happens. His original
  training failed immediately on leaving the starting area swordless; that is an acceptable option, not a
  required implementation. Investigate the start specialist/task and chain behavior, then measure swordless
  departures before/after the chosen fix. Do not mistake later fairy-pond visits for the initial starting area.
  - **Measured (round 10 §1):** 10 of 120 full chains on each route (8.3%, the same seeds on both) exit 0x77 north
    without the sword; all 10 fail (on milestone1 they are all 10 deaths in 0:0x37). ow-start's eval agrees: the
    `start` state fails 2 of 40 sampled episodes (`exit-wrong`), every other ow-start state 0/240. `start` is 1 of
    ow-start's 10 start states. First try after item 8: continue ow-start with `start` as its only start state
    (`--stop-success 0.99`), then 120 full chains per route.
  - **Done, round 10 §3 (r38):** ow-start retrained on `start` then rebalanced on all its states; swordless
    departures 10/120 → 0/120 on both routes, full-chain success 3→7 and 6→8 of 120; r38b-rebalance promoted.
- [x] **Alternate-YAML experiment integrity (round 7 §4):** train/evaluate discarded supplied YAML overrides.
  Fixed and proved through real/spawned CLI paths. Round 6's r20-pt, r21/r22-chainhp and r25/r27-heart treatment
  comparisons are not valid treatment evidence; model/chain measurements remain real. Explicit rejections remain.
- **0x53 (5-Stalfos key room):** 5 deaths in the full chain after the D1 Goriya room's 15.
- [x] **Finite stall escape in training and `play-zelda.py` (Lee, Nov 4; round 7 §9).** Lee still observes
  half-heart/clock/stunned-enemy Goriya runs attacking while standing still indefinitely. Required:
  - **Training:** a finite no-progress timeout that stops such runs. It may be a while; endless attack loops
    must not survive merely because Link keeps selecting attacks. Choose/measure a suitable threshold.
  - **Player at N steps:** if no room progress (no damage taken or dealt), give that screen the same opportunity
    to be replaced by another game as a room transition. Eligibility, not an unconditional forced swap.
  - **Player at 2×N steps:** reset the stalled run, including runs with no replacement candidate. Track room-local
    policy steps, not GUI redraw ticks; actual damage in either direction counts, not the attack command itself.
  - Existing `LingerTracker` deliberately does not count damage taken as progress; do not blindly reuse that
    definition for the requested player counter. Preserve the existing replacement/fade path.
  - Earlier native Rope seed 15 also had clock-frozen one-HP foes, a closed required shutter and a 2,942-action
    quiet tail (last 1,000: 978 MOVE/11 SWORD). Clock is observed; this is not shadow-Link chasing or the repaired
    shutter objective. Implement only after the current experiment and Lee's requested pause/resumption.
  - **Done, round 10 §6:** one no-damage counter (damage dealt or taken). Player `--stall-steps 300`: eligible
    for replacement at 300, run ends `stuck` at 600. Training/eval `no_damage_steps` default 600 (bootstrapped
    `no-damage` truncation). Thresholds from 9,033 successful chain visits (10 reach 600; every `stuck` does).
- **Promote by damage too.** A model that wins 3% more but takes twice the damage is worse in a chain
  (ow-leever-peahat r9a vs r3). Evaluate at chain-like hp and rank by success and `mean_health_change`.
- [x] **Bow route (Lee, Nov 9): mapped; training next.** On the route since Nov 9 (`docs/routes/d2first.md` "The
  bow detour", `docs/decisions.md` 2026-11-09): 0x43 N → 0x33 (key) → 0x23 (key) → 0x22 (push the west block N or
  S, stairs) → the bow cellar 0x7f → back to 0x43 → 0x44. New families `trap-room` and `equipment-room` with states
  and curricula; **next: train them** (and retrain stalfos-keese-gel / goriya / the families after the detour on the
  rebuilt states, which now carry the bow).
- **0x22 pushed the wrong way** (open, Nov 9): pushed E the west block walls the stairs in until 0x22 is
  re-entered; the objective then still points at the unreachable stairs (no wavefront path, no PBRS gradient). If
  trained models do this, consider sending Link out E and back in (the route's 0x23 from W without the bow is off
  the route and already leads back W).
- **play-zelda GUI** (uncommitted, Lee trying it; see AGENTS.md): `--games N` with best-hidden-game swaps on room
  change / restart and a fade, fps in the window title, `--benchmark`. Offscreen 3x2 measurements: 6 games 55.8
  fps per cell, 32 games 56.9 (every game still emulating at 60). The cap is the GUI's 16 ms tick (about 57 ticks
  per second), not the game count: lower `TICK_MS` in `play-zelda.py` to reach 60. Each game opens its own CUDA
  context, so a large `--games` next to a training run can run out of GPU memory (`--cpu` avoids it).

### Tried and dropped, or rejected

- ~~Chain-matched hp~~ round 6's goriya/stalfos-keese-gel arms never applied it (round 7 §4). Retested properly on
  goriya in round 10 §5 (r39, two seeds per arm, 1,200 chains per arm, refreshed profile): no effect (162 vs 161
  successes, low-health Goriya deaths 52% in both). Off everywhere; the option stays.
- ~~Pass-through damage penalty~~ tested twice in round 6 (overworld, d2-rope): no chain gain; dropped (code
  removed). A damage-avoidance idea for later would need a different form.
- ~~Heart objective~~ (`collect_hearts`) tested on stalfos-keese-gel and d2-rope (round 6 §10): more pickups on
  d2-rope but no chain gain; dropped (code removed). Item 4 above is a different rule (nearest of heart or exit).
- ~~Refill hearts on the way~~ **done** as the second route `milestone1-fairy` (round 6 §9). No fairy stop exists
  before D1, and D2 has none.
- ~~Skip the optional boomerang pickup in 0x44~~ **Rejected by Lee:** keep the pickup. (Item 7 tests using it
  after D1, a different question.)
- ~~Split Dungeon 1 into one model per enemy type~~ **Rejected by Lee:** group models by the enemies in the room,
  but don't boil it down to one model per enemy type or per room (beyond what Lee asks for, e.g. `wallmaster`).

## Items

- **Bombs are off for milestone 1 (`items: {bombs: off}`, round 3 on).** Bomb use paid more than it cost: −0.25 to
  place, +0.5 per hit plus +0.25 hit reward. d2-goriya threw bombs on 22% of its actions in round 1, and no
  canonical milestone-1 room needs bombs (`docs/route.md`).
- **Bomb skill: with the Dodongo work, after the route reaches the Dungeon 2 boss room.**
  - Grant bombs at reset (`ram_randomization: {bombs: [n, 8]}`).
  - Add a bomb-into-Dodongo reward and a scarcity cost for bombs spent on other enemies.
  - Then decide which specialists get bombs back (bomb walls in later dungeons).
- **Boomerang stays on for now.** Route states carry it from 0x44 onward: the Wallmaster room, Aquamentus, the
  triforce room and everything after Dungeon 1. It stuns Wallmasters, Goriyas and Keese. Whether it helps after D1
  is "Next up" item 8. Watch `kind/BOOMERANG` in training metrics.

## Training methods (from "Training with Savestates and Branching", Nov 2, 2026)

- **Idea 2: branched rollouts for advantage estimation (`branch:K`, VinePPO/GRPO).** Deferred in round 2. Round 2
  tests Idea 1 first, because the failing rooms are exploration problems (the policy never reaches the exit), not
  credit-assignment problems. First comparison when picked up: boss-aquamentus, GAE vs branch:K at equal wall clock.
- **Idea 3: search distilled into the network (expert iteration).** Deferred. Reserve for a specialist where Ideas 1
  and 2 both fall short (Dodongo is the expected candidate).
- **Evaluation grid (entry × health × items × RNG seed).** Deferred. `triforce.evaluate` already breaks results
  down per start state; adding health/item axes is a reset-option change plus a report.
- **Go-Explore cell definition for puzzle rooms (block push, bomb walls).** Round 2's cell is (room, tile // 2,
  hearts, living enemies, keys, boomerang). Puzzle progress (block pushed, wall bombed, door opened) isn't in it.
  Add it when a puzzle specialist is explored.
- **Curriculum threshold: fixed vs adaptive.** Round 2 uses a fixed 0.6 over windows of 10 episodes.
- **Curriculum progress doesn't survive `--load` resume.** `<run>/curriculum.json` restarts with every training
  invocation.
- **d1-melee regresses under the curriculum (79% → 44%, round 2).** The control without curriculum gets 78%.
  Candidates to test, one at a time:
  - keep a 10–20% curriculum share after every trajectory completes, starting anywhere on the whole trajectory,
    instead of switching off abruptly;
  - run the curriculum only for specialists that plain PPO can't solve (make it a per-specialist config field);
  - keep periodic checkpoints and pick by a held-out eval, so a late collapse can be seen and avoided.
- **Explore misses progress that isn't in the cell.** It doesn't capture the sword pickup (`ow-start` `start` and
  `0_77c` unsolved) or long multi-room paths (d1-melee 13/26 solved, 1.17M explore steps). Add sword, triforce
  and the items in `progress_events` to the cell. Consider exploring room by room instead of per family.
  Round 3 added enemy health to the cell (Aquamentus 0/3 → 3/3); ow-zora and d1-wallmaster were explored before
  that, and d1-wallmaster `1_44w_2` is still unexplored: re-run it with the new cell.
- **Training-time `balanced_success_rate` lags** when a specialist finishes few room-entry episodes per iteration
  (d1-melee: 26 states, 5–17 episodes per iteration). Consider a time-based window.
- **Long episodes starve the curriculum** (round 3; fixed for boss-aquamentus in round 5 with `max_steps: 400`,
  0% → 93%). `max_steps` sets the episode rate; set it from the explored path length for any curriculum specialist
  whose failures are slow (fleeing, waiting) rather than quick deaths.
- **Single seed per configuration.** d1-wallmaster r2b (96%) vs r2c (87%) shows the run-to-run spread; repeat key
  comparisons over 2–3 seeds before acting on a few-point difference.

## Driver

- [x] **Harden long-lived launcher ownership/recovery (round 7 §21).** R36's baseline owner returned no output;
  its launcher was gone and Aquamentus had saved an interrupted 37-update checkpoint after six full families.
  Parent's temporary recovery validates/keeps full-budget families, preserves partial attempts and restarts
  incomplete families fresh. Make the permanent driver retain reliable ownership/completion reporting;
  never count an interrupted exit-zero checkpoint as a completed fixed-budget comparison.
  - **Done (Nov 6):** `train_summary.json` records `status` (completed | early_stop | interrupted) and this run's
    iterations/steps; an interrupted run exits 130, not 0 (`triforce train` also stopped exiting 1 on success).
    `train_round.py` promotes only finished runs, and `--skip-existing` skips only finished runs: an unfinished run
    dir becomes `<run>.partial-N` and the family trains fresh. The r37/r39 runners skip only finished summaries
    (all 32 of their runs trained the full 245 updates). Detached launches that survive the tool shell:
    `setsid nohup CMD >> LOG 2>&1 < /dev/null & disown`.

- **Keep training GUI logs in the standard main `logs/` root (Lee, Nov 4).** R36's isolated arm logs were put
  under `models/r36-observation-full-retrain-evidence/logs/schema1|schema2`. Those producer paths stay stable,
  but live ten-specialist series views now exist under standard `logs/` (round 7 §21), so normal GUI launch works.
  Future producer logs must also use the normal root, with arm identification in run/session metadata.

- **Re-running a subset with the same `--round` overwrites `models/<round>-summary.json`** (round 3: the
  boss-aquamentus r3c run replaced the ow-zora / d1-wallmaster summary). Merge into the existing summary, or
  refuse to reuse a round ID.
- **Mapper runs log JSONL and get their own GUI view (Lee, Nov 3; later, not urgent).** `entry_states build/path`
  and `triforce.explore` print text only. Write event logs (`logs/<yymmdd>_<NNN>_<what>/`, like training):
  per room/state attempts, iterations, cells, env steps, best progress, success/failure. Then teach
  `training-gui.py` to recognize a mapper log folder and show it sensibly (likely a different view that opens for
  that kind of folder: route steps as rows, per-attempt progress).

## Start states

- Round 3 replaced the state set (`round3.md`): pre-entry, route-generated, route-true hearts and inventory. That
  resolved the round 2 items (1-step states, triforce 0 on post-D1 visits, 16-heart states, missing D2 states).
- **No state for 0x0e** was the case until round 6: `2_0es_0..1` now exist (Dodongo room entries, for later).
- **Post-D1 states:** the fairy route's rooms (0x48 … 0x4d from S), 0x38 from W, 0x4c from E and all of Dungeon 2
  have chains 0–2 (round 6). milestone1's north road after D1 (0x28 … 0x2d, 0x3d, 0x4d from N, 0x3c) still has
  only chain 2 (`*_2`), apart from `0_3cs_0..1`. More chains there would add spawn/health variety.
- **`0_4ce` (ow-octorok, the 0x4d → 0x4c route entry) was the weak state** (65% in round 3; 0x4c 92% in round 6).

## Runner / objectives

- Round 4 fixed 0x43 → 0x33 (route-only exits, pathfinding through tagged rooms only, route order on revisited
  rooms; `round4.md`).
- **The chain records no keys or route step per room.** Adding them to `RoomRecord` would have found the 0x72/0x33
  key problem without a throwaway script.

## Rewards

- **Damage still resets the room-stalling counter** (v1 behaviour, `critics.critique_movement`). In rooms with
  frequent ranged damage (Zora fireballs) this delays the stalling penalty. Not changed in round 2.
