# Experiment backlog

Ideas and problems found but not acted on yet, with the round that found them.

## Next up

Goal until further notice: a set of specialists that carries the milestone-1 chain to the Dungeon 2 boss (Dodongo).

- **Where it stands:** round 7 active (`round7.md`); lingering, Aquamentus exit training and observation audit done.
  Round-6 baseline best chains (40 runs, to Dodongo): full 1/40 (milestone1), 2–3/40 (milestone1-fairy), fairy
  post-D1 8/40, D2 entrance 9/40. D1 Goriya 0x44 and Stalfos 0x53 remain the full-chain bottlenecks.
- **Promoted now:** ow-zora / ow-octorok / ow-start r24-fairy, ow-moblin r26c-moblin, ow-leever-peahat r23-more,
  boss-aquamentus r30-exit, stalfos-keese-gel r25-skg-plain, goriya r21-goriya-chainhp, d2-rope r28-d2,
  wallmaster r10 copy. Overworld families train on both routes' rooms; D2 rooms have 3 start states each.

### Work in this order

From Lee watching `play-zelda.py` after round 6 (Nov 3), sorted: first the measurement everything else needs,
then clear bugs, then what decides how the next full retrain looks, then reward/target changes aimed at the
chain's death rooms, then the big ablation. Round 7 items 1–3 complete; Goriya blocking/corner behavior is next.

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
   - [ ] **Bundled observation trial before the from-scratch item 8 study:** versioned chase target coordinates,
     Link/monster boomerang position/phase and corrected harm semantics; compare a full retrain. No RNG, guessed
     universal owner, redundant stun/clock flags or incidental timer rescaling. This bundle is not yet implemented.
4. **Goriya rooms: blocking, and the corner stalemate (Lee).** v1 rewarded the shield-block sound, and that taught
   Link to block instead of face-tanking. v2 already has `critics.critique_block` (`reward-block-projectile`,
   REWARD_MEDIUM × combat decay, on the onset of the ArrowDeflected sound via `Link.is_blocking`). First check that
   it fires on Goriya boomerangs (count blocks per episode in goriya evals and a few traces), then A/B a stronger
   block reward on goriya (0x44 is the chain's top death room). Caveat (Lee): the sound alone is wrong in rooms
   with Darknuts etc. (their shields clink too), so require a blockable projectile in the air near Link plus the
   clink. Lee also watched the stalemate: Link in the top-left corner not fighting while the Goriyas chase shadow
   Link in the bottom right (AGENTS.md "Game mechanics"), no hits on either side, for ages. The stalling penalty
   (−0.01 to −0.02 per step after 150 quiet steps) is apparently not enough to break it; measure it with item 1,
   then consider a stronger penalty for quiet steps while enemies remain. Related, from round 6: at 1 heart the
   goriya model turns passive (33% WAIT) and dies to contact; other ideas are demos or curriculum from damage-free
   Go-Explore paths (select by health, more patience) and a contact-damage penalty that grows as hp falls.
5. **Hearts: aim at the nearer of heart or exit (Lee: "a big deal", v1 behaviour).** Agents leave many dropped
   hearts (round 6 §8: overworld 26 of 34). In v1, dropped items were among the wavefront's targets, and the
   wavefront is a multi-source BFS, so PBRS led to a heart whenever it was closer than the exit by walking distance
   (not straight-line) and ignored it otherwise. v2 adds items to the objective's `targets` but not to
   `pbrs_targets` (`objectives._GameMapObjective.get_current_objectives`), so shaping never points at a heart.
   Try: hearts and the clock (Lee: "certain items like the clock and hearts") in `pbrs_targets` alongside the
   exits. This differs from the dropped `collect_hearts` (heart was the only target whenever Link was hurt, kind
   ITEM). A/B on stalfos-keese-gel, d2-rope and an overworld family; measure taken/left, real vs total heart
   change and the chain.
6. **Cut runs that stop making progress (Lee: in v1 a run was truncated after ~100 steps without progress, e.g.
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
7. **D2: east first for the 0x7e key (Lee).** From the D2 entrance go 0x7d E → 0x7e (kill the ropes, take the key)
   → N 0x6e, instead of 0x7d N → 0x6d → E 0x6e. Slower, but the key is needed later in the dungeon (past the
   milestone). Today 0x7e is an equal-length alternate with exit N and no key objective (`routes.yaml`
   alternates). Make it the step on both routes with a FIGHT/TREASURE key objective, retrain d2-rope (states
   `2_7ew_0..2` exist), and rebuild the D2 states downstream so they carry the key.
8. **Boomerang after D1: does it help or hurt? (Lee: big but worth it; "get to the real bottom of this".)** Link
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
   - Needs: per-arm configs in the chain (the runner reads `specialists/<name>.yaml`; add a config override or a
     specialists-dir option to `eval-chain`).
   - Run it after items 1–7 settle (or as part of the full retrain item 3 may force), so it measures the setup we
     keep. Expect about a day of GPU.

### Other open items

- [x] **Alternate-YAML experiment integrity (round 7 §4):** train/evaluate discarded supplied YAML overrides.
  Fixed and proved through real/spawned CLI paths. Round 6's r20-pt, r21/r22-chainhp and r25/r27-heart treatment
  comparisons are not valid treatment evidence; model/chain measurements remain real. Explicit rejections remain.
- **0x53 (5-Stalfos key room):** 5 deaths in the full chain after the D1 Goriya room's 15.
- **Promote by damage too.** A model that wins 3% more but takes twice the damage is worse in a chain
  (ow-leever-peahat r9a vs r3). Evaluate at chain-like hp and rank by success and `mean_health_change`.
- **Bow route** (later): 0x43 N → 0x33 (key) → 0x23 (key) → 0x22 (push block, stairs to the bow cellar 0x7F).
  Needs a route/objective for the bow and a specialist for 0x22 and the cellar.
- **play-zelda GUI** (uncommitted, Lee trying it; see AGENTS.md): `--games N` with best-hidden-game swaps on room
  change / restart and a fade, fps in the window title, `--benchmark`. Offscreen 3x2 measurements: 6 games 55.8
  fps per cell, 32 games 56.9 (every game still emulating at 60). The cap is the GUI's 16 ms tick (about 57 ticks
  per second), not the game count: lower `TICK_MS` in `play-zelda.py` to reach 60. Each game opens its own CUDA
  context, so a large `--games` next to a training run can run out of GPU memory (`--cpu` avoids it).

### Tried and dropped, or rejected

- ~~Chain-matched hp~~ tested on goriya (within noise) and stalfos-keese-gel (worse); off everywhere. The option
  (`ram_randomization.chain_hp`) stays for a later retry.
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
