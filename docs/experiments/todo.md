# Experiment backlog

Ideas and problems found but not acted on yet, with the round that found them.

## Next up

Goal until further notice: a set of specialists that carries the milestone-1 chain to the Dungeon 2 boss (Dodongo).

- **Where it stands:** round 7 active (`round7.md`); items 1–7 and their prerequisites complete. East-first D2
  r35 team carries the required 7e key into Dodongo: full main/fairy 1/3, post-D1 main/fairy 4/7, D2 19 (40 each).
  Route/states/default curricula and model cutover are verified in main. Next: versioned full observation trial.
- **Promoted now:** ow-zora / ow-start r24-fairy, ow-octorok r32-nearby-off, ow-moblin r26c-moblin,
  ow-leever-peahat r23-more, boss-aquamentus r30-exit, stalfos-keese-gel r25-skg-plain, goriya r31-block1,
  d2-rope r35-east-curriculum, wallmaster r10 copy. Overworld families use both routes' rooms; D2 rooms have 3 states each.
- **Lee's pause request (Nov 4):** finish the current r36 observation-bundle experiment, record and commit
  its result, then pause. Do not start the boomerang study or the swordless-start fix before Lee resumes work.

### Work in this order

From Lee watching `play-zelda.py` after round 6 (Nov 3), sorted: first the measurement everything else needs,
then clear bugs, then what decides how the next full retrain looks, then reward/target changes aimed at the
chain's death rooms, then the big ablation. Round 7 items 1–7 complete; versioned observation comparison next.

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
   - **Full-study input prerequisite complete (§20):** retired six old Moblin curricula (297 snapshots) with
     three containers/no boomerang/no triforce in post-D1 rooms; preserved the valid four-heart 0_3dc path and
     installed verified current 0_3dn_2 coverage. All 14 Moblin goal replays and 41 default-track route audits pass.
     One old D2 Goriya goal path is stale; the already verified replacement belongs to the pending D2 cutover.
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
   - **Per-arm chain config prerequisite implemented in dev, round 7 §11:** repeatable `--specialist-config YAML`
     carries loaded family configs into direct/spawned chains. Native boomerang-mask/cutoff smoke and all
     464 tests passed at that step; merged into main at 4cb8434.
   - Run it after items 1–7 settle (or as part of the full retrain item 3 may force), so it measures the setup we
     keep. Expect about a day of GPU.

### Other open items

- [ ] **Swordless departure from the starting area (Lee, Nov 4).** Lee observes runs where Link leaves without
  taking the sword; these runs cannot win and waste training. Reduce how often this happens. His original
  training failed immediately on leaving the starting area swordless; that is an acceptable option, not a
  required implementation. Investigate the start specialist/task and chain behavior, then measure swordless
  departures before/after the chosen fix. Do not mistake later fairy-pond visits for the initial starting area.
- [x] **Alternate-YAML experiment integrity (round 7 §4):** train/evaluate discarded supplied YAML overrides.
  Fixed and proved through real/spawned CLI paths. Round 6's r20-pt, r21/r22-chainhp and r25/r27-heart treatment
  comparisons are not valid treatment evidence; model/chain measurements remain real. Explicit rejections remain.
- **0x53 (5-Stalfos key room):** 5 deaths in the full chain after the D1 Goriya room's 15.
- [ ] **Finite stall escape in training and `play-zelda.py` (Lee, Nov 4; round 7 §9).** Lee still observes
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
