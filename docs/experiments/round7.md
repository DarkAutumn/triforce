# Round 7: lingering measurements and post-round-6 behavior experiments

## Summary

Lee resumed experiment mode on Nov 3: keep the omp todo live, reconcile it with `todo.md` between experiments,
run experiments, log every result and commit each finished item. Aquamentus r30-exit is now promoted.

- Backlog item 1 complete: behavior-neutral lingering measurement, seeded gameplay parity, training/eval/runner/GUI
  proof and tracked baseline results. Outputs: `models/r29-linger-baseline/`; no model promoted.
- Goriya and Aquamentus parity smoke: three seeded episodes each (seed 700 onward), one entry state each,
  produced identical pre/post gameplay records after removing the added measurement field.
- Targeted checks: 53 tests passed (tracker, env, runner, evaluation, rollout aggregation, events).
- Full dev-worktree suite: 383 tests passed in 104.47 seconds.
- CUDA training smoke: Aquamentus continued for 2 iterations / 1,024 actions with no curriculum; 29 completed
  episodes, all successful. Per-update mean longest quiet stretches 5.73 and 6.29 actions; worst 10 in both.
  This is a smoke checkpoint in `/tmp`, not a candidate for promotion.
- GUI verified on measured training, measured per-room eval and historical absent-data logs, at wide/narrow sizes.
  Logs: `logs/261103_029_boss-aquamentus/`, `logs/261103_030_r29-linger-baseline/` and old r28-d2.
- Lee's running player uses most VRAM. He authorizes killing `play-zelda.py` without asking **only if CUDA memory
  exhaustion crashes the experiment**. The CUDA smoke succeeded; the player was not killed.
- Backlog item 2 complete: r30-exit trains kill → heart → east exit. Matched exit-task eval improved 59/60 →
  60/60; mean pickup-to-exit 92.0 → 11.1 actions, worst 1,963 → 77; health change +0.61 → +0.69 hearts.
  Both matched full chains removed the two old post-heart timeouts. Other families still limit the chain.
- Worker-seed prerequisite fixed and verified (§7): real rollout tensors repeat with the same seed,
  distinct workers/seeds differ, episode streams advance; CPU CLI/CUDA smoke and all 462 tests passed.
- Backlog item 4 complete: r31-block1 promoted after 800 frozen-team chains (§6). Ordinary reward has the lowest
  evaluation damage; stronger block/quiet rewards stay disabled. Post-fight D2 navigation remains a measured problem.
- Backlog item 5 complete: eight r32 arms/four baselines, 1,840 old-runtime and 1,200 corrected-runtime chain
  executions (§9). Ordinary Octorok `r32-nearby-off` promoted after disjoint-seed validation; shaping stays off.
- Shaped Rope improves D2 but regresses both full routes; keep old Rope/SKG/Moblin. Octorok's ordinary gain is
  modest and consistent across the combined success/health comparisons, not evidence that item shaping works.
- Native replay found required-shutter objective/classification and terminal-health defects. Prerequisites
  through §14 merged at `4cb8434`; **470 tests passed**.
- Backlog item 6 complete (§15): all three r33 arms, common/actual evaluations and 1,400 fresh chains finished.
  No general chain gain; keep r31-block1 and canonical cutoffs null.
- Backlog item 7 complete (§16–19): r34 continuation failed the east exit; verified-curriculum r35 learned it
  (60/60 entrance exits, 400/420 room successes), promoted after 200 native chains with all 34 Dodongo arrivals
  carrying the required 7e key. Both routes/states/default curricula and the action-boundary fix are active in main.
- Final canonical verification: **471 tests passed**, CUDA default-curriculum update completed; main's native
  promoted-team smoke reached Dodongo with three keys on two D2 seeds and the post-D1 fairy seed.
- R36 observation-bundle comparison is active on frozen sources (§21); schema 2 is verified in dev, not main.
  Lee requests recording/committing this experiment's result and then pausing; no next experiment starts.

## 1. Shared lingering metric

`LingerTracker` counts the longest quiet stretch (`max_steps`, `max_frames`) and the final quiet tail
(`tail_steps`, `tail_frames`), plus elapsed actions/frames. Progress: an enemy hit/kill, pickup (including
weapon-collected hearts at full health), health gained (including the fairy pond), a new best objective distance,
or a room exit. Taking damage alone is not progress.

Distance uses a wavefront frozen at each tangible-progress/objective-phase boundary. Moving enemies cannot reset
stationary Link's clock, and round trips cannot repeatedly earn new best distance. Objective phase changes
rebaseline distance but do not themselves reset the quiet clock. A progress action ends a streak: that action's
frames are not included in the preceding quiet interval. Actions and emulator frames are both retained because
option durations vary.

Terminal env info and runner room records carry the same summary. Training reports mean per-episode longest/tail
and per-iteration worst, separately for room-entry and curriculum starts, plus entry-room breakdowns. Eval keeps
raw episode summaries and overall/per-room/per-state means and maxima. Historical logs without tracking remain
unavailable rather than showing fabricated zeros. Actual boss-death and heart-container-pickup timestamps support
kill → heart → exit diagnostics without changing the policy observation schema.

### Tracked baseline protocol

- Current promoted Goriya and Aquamentus: 20 sampled episodes per start state, seed 0 onward, current YAML health
  distribution, 6 evaluation processes. Baseline results are stored separately, not over their old `eval.json`.
- Full milestone1 and milestone1-fairy chains: 40 sampled runs each, seeds 0–39, 12 processes.
- Dungeon 2 segment: 40 runs from `2_7ds_2`, same seeds/process count.
- No rewards, timeouts, start states, schemas or promoted checkpoints changed for these measurements.

### Baseline results

| Specialist / room | Episodes | Success | Health change | Mean longest quiet actions | Worst quiet actions |
|---|---:|---:|---:|---:|---:|
| Goriya overall | 240 | 178/240 (74%) | −0.96 | 86.4 | 1,971 |
| Goriya 1:0x44 | 60 | 73% | −0.91 | 109.0 | 1,971 |
| Goriya 1:0x23 | 60 | 77% | −0.62 | 175.4 | 1,902 |
| Goriya 2:0x5e | 60 | 75% | −1.08 | 20.3 | 127 |
| Goriya 2:0x1e | 60 | 72% | −1.23 | 41.1 | 559 |
| Aquamentus (ends at heart) | 60 | 60/60 (100%) | +0.48 | 6.2 | 12 |

| Chain segment | Runs | Dodongo room reached | Important findings |
|---|---:|---:|---|
| Full milestone1 | 40 | 0/40 | D1 Goriya: 15 deaths, 1 stuck; Aquamentus: 2 deaths, 2 stuck |
| Full milestone1-fairy | 40 | 2/40 | Same D1 results; D2 0x2e also stuck once |
| D2 entrance | 40 | 9/40 | 0x2e: 6 deaths, 4 stuck; 0x3e: 8 deaths; Goriya 0x1e: 8 deaths, 1 stuck |

These are measurement baselines, not an improvement claim. Both full chains had the same 12 Aquamentus visits:
eight exited, two died in the fight, two killed the boss and took the heart but exhausted 2,000 actions. Their
post-heart delays were 1,960 and 1,964 actions. One successful visit took another 491 actions after the heart.
The D2 segment's 0x2e visits averaged a 393.6-action longest gap, worst 2,982; 0x4e worst 2,410. The main chain's
1:0x43 worst was 2,881 (specialist budgets can exceed the runner's 2,000-action minimum).

Long success gaps matter too: successful Goriya 1:0x23 episodes had mean longest 154.4 actions, worst 1,274;
successful 1:0x44 episodes worst 399. A blanket 100-action cutoff would reject some current successes: use these
measurements when testing backlog item 6, and improve the behavior rather than call every long run acceptable.

## 2. Aquamentus post-heart diagnosis

Extended the **evaluation task only** through the correct exit (temporary config: success `[exit-correct]`,
max_steps 2,000), retaining the current promoted checkpoint and YAML hp distribution. The objective has no correct
next room while the boss/heart remain, so leaving early is not a success. 20 episodes per each of 3 fight-entry
states, seeds 0–59, six workers. Output: `aquamentus-through-exit.json` in the baseline folder.

- 59/60 correct exits (98%); mean 126 actions, mean health change +0.61 hearts.
- Among successful visits, post-heart delay mean 60.3, median 17, worst 819 actions.
- Seed 58 (`1_35s_2`, two starting hearts): killed Aquamentus at action 33/frame 353, took the container at
  action 37/frame 379, then made no progress for 1,963 actions / 17,627 frames before max-steps.
- Replayed that exact seed through the real env: reproduced all 2,000 actions and saved screenshots at boss death,
  heart pickup, 100 actions after pickup and the end, only with `$12 == 5` (playable, not mid-scroll).
- Post-heart trace: objective MOVE throughout; 770 `MOVE E 8`, 757 `MOVE W 8`, and 94 `WAIT 8` actions dominate.
  Final positions bounce between (148,101) and (160,101), north of the east door. This confirms Lee's observation,
  not a shadow-Link fight explanation. The training task ends before this behavior is evaluated.

Decision: train the same boss model through the required heart pickup and east exit, not a stronger penalty first.
Observation audit runs before that training so any schema decision is explicit. No start states or model changed
for diagnosis; no post-kill training states saved/promoted.

## 3. Observation audit and returning-boomerang correction

Read observation.py, Link/enemy/projectile decoding and the corresponding collision/weapon routines in the Zelda
disassembly. Read-only scouts independently covered stun/collision and chase/weapon/owner semantics. The old archive
could not be read by the archive tool (921 MB exceeds its 256 MB limit); streamed only its observation wrapper into
`/tmp`. That archived wrapper is a pixel viewport/grayscale transform, not evidence of an additional structured
feature list (`triforce_lib/zelda_observation_wrapper.py` L41–105).

| Input/mechanic | Finding and decision | Evidence |
|---|---|---|
| Enemy boomerang stun | `$3D+i ← $10`, one tick per 10 frames: nominal 160 frames. Existing `/16` is correct; do not rescale. Dodongo's bomb stun is a different `$20` mechanism. | Z_01 L5914–5919; Z_07 L481–507; runtime injected timer decremented every 10 frames |
| Damage invincibility | Link `$4F0 ← $18` (nominal 48 frames); ordinary enemies `$10` (32 frames), half-rate countdown. Existing `/32` does not clip these. Clock is already observed and separately controls protection. | Z_01 L6589–6605,L6687–6691; Z_07 L5782–5788,L1877–1884 |
| Chase target | Actual refresh flag + target X/Y missing. Bundle actual `$60/$61/$62` with full-byte coordinate normalization; no RNG/timer initially. The ASM's last-frame comment contradicts executed instruction order. | Z_07 L1888–1951,L3788–3811 |
| Link boomerang flight | Active alone omits position and tactical phase; bundle outbound/slow-outward/spark/slow-return/fast-return. `$80+` is bait, not boomerang. | Z_05 L2901–2948; Z_07 L3857–4280 |
| Projectile harm | Stunned monsters and returning/bouncing projectiles currently can read hurts_on_touch. Bundle corrected harmful-state semantics and boomerang return phase; present/active remains separate. | observation.py L435–451; Z_01 L5613–5714; Z_07 L3857–4280 |
| Projectile owner | Reciprocal `$42C` refs exist for Goriya boomerangs only. Owner relation is optional and must validate both slots/types; ordinary shots retain stale scratch data. No nearest-enemy guessed owner. | Z_04 L590–596,L11361–11391; Z_01 L4074–4086 |
| Shield blocking | ArrowDeflected is also Darknut/immunity parry. Goriya blocking produces a spark, then return; return/catch alone is not a block. Use per-frame collision evidence before increasing rewards (item 4). | Z_01 L5707–5714,L5897–5912,L6312–6333; Z_07 L4265–4280 |

**Schema decision:** keep schema 1 during the isolated Aquamentus/task/reward experiments. No speculative input
changes or incidental timer rescalings now. The chase/phase/harm bundle needs a deliberately versioned full-retrain
comparison, not loading existing weights under changed meanings; keep it recorded as a prerequisite to the later
from-scratch boomerang study. Audit is complete; the proposed schema bundle is not implemented or claimed complete.

### Return-state decoding bug: fixed before reward comparisons

`Link.get_animation_state(BOOMERANG)` used **decimal 10–57** instead of hex `$10–$57`. Raw `$40–$57` return states
therefore reported inactive, ending future-credit simulation before the weapon finished its flight.

- Live probe through `GameSession.load/act`: after leaving the south doorway, threw east; all 63 observed return
  frames were incorrectly INACTIVE. Initial attempted throws in the doorway correctly did not launch anything;
  moving once was insufficient because the first move was interrupted by the door event.
- Changed only the two bounds to `0x10` and `0x57`, leaving observation/action schemas and masks unchanged.
- Same live probe afterwards: all 63 return frames ACTIVE, zero incorrectly inactive. Output is stored in
  `models/r29-linger-baseline/boomerang-decode-probe.json` (post-fix trace).
- Real-emulator regression failed before on the ACTIVE assertion, then passed after; it checks return → catch
  and excludes bait from boomerang activity. Core/env/reward/tracker checks: 75 passed.

This is a correctness fix, not evidence of a model/chain improvement. Re-evaluate old checkpoints under the corrected
credit path and the experimental task before comparing continued models.

## 4. Alternate-YAML CLI integrity fix (prerequisite to r30)

The first CLI evaluation with `/tmp/boss-aquamentus-exit.yaml` still printed `boss-dead-and-heart` and 60/60 success,
despite the file requesting only `exit-correct`. This was a real config-loss bug, not a valid exit-task baseline.

- `evaluate.main` loaded the supplied file only to extract `config.name`, then reloaded the default specialist.
- Training similarly discarded the file before `_train`, model/curriculum setup and `ZeldaEnvFactory`.
- Fix: carry the loaded `SpecialistConfig` into evaluation and training workers; use its name only for registry
  identity/log labeling. Worker factory explicitly accepts the config object, including spawn/pickle use.
- Two real CLI regressions failed before: evaluation ran three default seed states instead of the single supplied
  one; spawned training completed zero episodes instead of eight one-action episodes. Both pass after the fix.
- Training/eval/PPO checks: 15 passed. Actual CLI smoke on the supplied exit config ran CUDA training with two
  workers (256 actions, 3 successful completed episodes averaging 54 actions) and evaluated the old promoted model.
- Full post-fix dev suite: 386 tests passed in 86.59 seconds.
- Correct old-model baseline with max_steps 400: **57/60 correct exits (95%)**, mean 90 actions, +0.61 hearts;
  mean longest quiet stretch 48 actions, worst 365. Stored as `models/r30-aquamentus-exit-old.json`.

### Historical experiment implications

`git log -S` shows name-only loading dates back to the initial/round-1 CLI implementation. `run.json` confirms the
following arms supplied temporary YAML files, whose overrides the CLI discarded:

| Arm | Supplied file |
|---|---|
| r20-rope-pt | `/tmp/d2-rope-pt.yaml` |
| r21-goriya-chainhp | `/tmp/goriya-chainhp.yaml` |
| r22-skg-chainhp | `/tmp/skg-chainhp.yaml` |
| r25-skg-hearts | `/tmp/skg-hearts.yaml` |
| r27-rope-hearts | `/tmp/rope-hearts.yaml` |

Their measured model/chain scores remain real, but those comparisons **do not establish the requested treatment's
effect**: the variant file was not used for training. Old main-YAML values cannot be recovered from these run files,
so do not retrospectively invent exact treatment settings. Round 6 is annotated accordingly. Lee's explicit rejected
ideas stay rejected; this audit is not authorization to reintroduce pass-through damage or heart-only objectives.

## 5. Aquamentus exit training protocol (r30)

Variant file preserved at `models/boss-aquamentus/r30-exit/config.yaml`: success `[exit-correct]`, otherwise current
settings including max_steps 400 and hp `[0.75,max]`, half full health. Bombs off, schema 1 unchanged. Warm-start from
promoted r23-more; stop at balanced 99%, at most 2M more actions. Keep the current promotion until comparison finishes.

Re-explored with the exit task: 3/3 successful replays, 42,957 env actions; paths 109/91/91 actions, final health
2.0/1.5/1.5. Files are under `trajectories/r30-exit/boss-aquamentus/`; snapshots are before each action, so the terminal
0x36 room is not a curriculum start. Use this separate directory for the candidate, not the old heart-ending paths.
Compare against the corrected old-model exit-task baseline, then both full chains and the post-D1/D2 segments if
promoted. Track boss-kill/heart/exit timing, health and quiet-gap distribution—not success alone.

### Results and promotion

Training stopped early after 50 additional PPO updates / 204,500 additional actions, 248.71 seconds. The checkpoint
stores cumulative 633 updates / 2,590,136 actions; these are not the experiment's additional budget.

Both models were evaluated on the exit task with the same current health distribution, 20 episodes per entry state
(60 total), sampled policies and seeds 0 onward. The old baseline is `models/r29-linger-baseline/aquamentus-through-exit.json`;
the candidate's records are `models/boss-aquamentus/r30-exit/eval.json`.

| Exit-task measurement | r23-more | r30-exit |
|---|---:|---:|
| Successful exits | 59/60 | 60/60 |
| Mean health change, hearts | +0.608 | +0.692 |
| Mean boss-kill action | 27.15 | 27.95 |
| Mean pickup-to-exit actions | 91.97 | 11.13 |
| Worst pickup-to-exit actions | 1,963 | 77 |
| Mean total actions | 125.78 | 46.88 |

Matched full chains, 40 runs per route / seeds 0–39: the candidate reached Aquamentus 12 times on each route,
with 10 exits and two low-health fight deaths instead of the old eight exits, two deaths and two post-heart
timeouts. Successful candidate visits averaged 50.7 actions. Dodongo-room reaches were 0/40 on milestone1 and
3/40 on milestone1-fairy; this does not establish a general chain-success improvement. The demonstrated gain is
eliminating the post-heart stall with better matched specialist health and boss-kill timing within one action.

After promotion, an independent full-chain sweep used seeds 40–59; post-D1 and D2 segments used seeds 0–19:

| Segment | Runs | Dodongo room reached |
|---|---:|---:|
| Full milestone1 | 20 | 1 |
| Full milestone1-fairy | 20 | 3 |
| Post-D1 milestone1 | 20 | 0 |
| Post-D1 milestone1-fairy | 20 | 4 |
| D2 entrance | 20 | 3 |

No execution errors in the sweep. Five D1 Goriya deaths and three 0x53 deaths on each full route remain the first
bottlenecks; D2 rope rooms and Goriyas limit the later segments. Post-D1/D2 skip Aquamentus, so their outcomes are
current-team measurements, not evidence of an Aquamentus treatment effect. Raw chains and sweep summaries are in
`models/boss-aquamentus/r30-exit/`.

**Promoted r30-exit.** `boss-aquamentus.yaml` now succeeds on the correct exit, after the objective requires the
kill and heart container. Replaced its default curriculum trajectories with the explored complete-visit paths.
No observation-schema change, damage-penalty addition or special post-kill controller.

## 6. Confirmed shield collisions and isolated experiment controls

Before increasing the Goriya reward, reproduced its missing credit using naturally thrown boomerangs, not injected
projectiles or a model evaluation. Protocol: fresh clean `1_44w`, RNG seeds 0–4; leave the doorway, place Link at
(112,144) with three hearts, clock/invincibility off; up to 250 WAIT 32 actions per seed, facing opposite the closest
outgoing Goriya boomerang via RAM. Inspect consecutive actually executed frames, with look-ahead disabled.

The old trace found seven shield-eligible parry collisions but rewarded only one. Replaying after the fix found
the same seven collisions at exactly the same seed/action pairs: `(0,106)`, `(0,121)`, `(0,122)`, `(1,65)`, `(3,18)`,
`(3,67)`, `(4,218)`; all seven earned block rewards. Example: frame 2973→2974, slot 11 outgoing `$17`→spark `$20`,
`$604=1` while `$605=0`; both controllable sound endpoints were silent. The sound-onset implementation missed it.
Evidence: `models/r31-block-evidence/{before.json,before-summary.json,after.json}`.

`projectile_blocks.py` now confirms projectile-specific collision signatures together with shield/facing/proximity
and the current-frame parry request. It rejects immune/unrelated clinks, harmless returns, unsupported/magic-only
shots without an eligible shield, stationary wall sparks and boundary fireball destruction. Actual executed
frames only: no replacement of the user's callback, altered input duration or speculative look-ahead collection.
Block events retain collision frame, slot, type and room; reward/decay counts each collision, including simultaneous
shots and attack/settling/uncontrollable frames. A heart pickup in the collision frame does not erase the block.
Episode, curriculum, evaluation and chain metrics retain confirmed counts; absent historical measurements stay absent.

Two requested later experiments have default-off controls; the cutoff still needs the PPO correction found in §8:

- `reward.nearby_items`: multi-source PBRS includes hearts/clocks alongside exits, without replacing the room task.
  Live greedy walker: nearby clock then heart then correct exit, +0.10 shaping, one heart healed (2→3 hearts).
  With a nearer exit: +0.05 shaping, correct exit at two hearts, distant heart left. This is runtime proof of the
  control, not evidence that trained models improve; backlog item 5 remains pending.
- `no_progress_steps`: positive shared quiet-tail cutoff, optional and disabled by default. Live WAIT actions 1/2
  continued; action 3 truncated with `no-room-progress`, soft failure, 27 quiet frames and no terminal penalty.
  Legacy timeouts remain when disabled; runner per-room cutoff retains the global guard. Backlog item 6 needs
  family-specific successful-gap measurement and comparison before enabling any threshold.

First integrated regressions exposed NumPy byte arithmetic and invalid WAIT constructions in new test fixtures;
fixed integral modular arithmetic and valid actions. Removed an incidental serialized-emulator-clock comparison
in favor of actual RAM/screen/timing/callback parity, and an unrelated exit-classification test pin. Full dev suite:
**447 passed in 98.43 seconds**. No observation/action schema change.

CUDA smoke with the triple-block config, warm-started from promoted r21-goriya-chainhp: two updates / 1,024 actual
actions, five completed episodes (four successes), 10.7 seconds. Training emitted measured `blocks=0` for these
five episodes; this proves the updated CUDA/metric path runs, not a nonzero-block reward or a model improvement.
The natural trace above provides the nonzero-block proof. No CUDA exhaustion; Lee's player was left running.

### r31 matched block-reward comparison

Configs preserved at `models/goriya/r31-block{1,3}/config.yaml`. Both continue the same r21-goriya-chainhp checkpoint,
with current YAML health mix `[0.15,max]` / half full health, bombs off, the same curriculum, seed 31, 245 updates
(about 1M additional actions) and no success early stop. Only the confirmed-block multiplier differs: 1 versus 3.
Separate matched corner arm `models/goriya/r31-quiet005/config.yaml`: ordinary confirmed-block multiplier 1 plus
0.05 additional quiet active-fight cost after the existing grace. Same seed, six workers, 245 updates and canonical
main curriculum; no nearby-item shaping or measured truncation. The block arms run on main code `6421275`; this
arm runs from immutable dev code `aa30a2f`, verified by 462 tests and faithful fight/post-fight replays. Do not merge
source while the cohort runs. All checkpoints/logs are under main's normal model/log roots. No candidate promoted.
Keep nearby-item shaping and measured cutoff off. Re-evaluate the old model on the corrected detector/current YAML;
compare success, real/total health, confirmed blocks and successful/failing quiet-gap distributions per room, then
both full chains and the relevant post-D1/D2 segments. A stronger block reward is not promoted merely for more blocks;
it must preserve health/success and avoid reward-producing stalemates. Evaluate the corner-stall lever separately.

### Current-detector baseline and faithful stall replays

Promoted r21-goriya-chainhp re-evaluated on the corrected current code/YAML: 178/240 success, health −0.96 hearts,
45 confirmed blocks (0.1875 per episode). Output: `models/r31-block-evidence/promoted-eval.json`.

| Room | Episodes | Successes | Confirmed blocks | Mean health change |
|---|---:|---:|---:|---:|
| 1:0x44 | 60 | 44 | 15 | −0.908 |
| 1:0x23 | 60 | 46 | 13 | −0.617 |
| 2:0x5e | 60 | 45 | 7 | −1.083 |
| 2:0x1e | 60 | 43 | 10 | −1.225 |

All 240 episode outcome/time/health/heart records match the earlier baseline after excluding the new block field
and lingering measurements. Of the lingering summaries, 239 match; seed 207 changes only longest quiet frames
55→79, retaining six quiet actions and 60 actions/657 frames overall. Cause not isolated; use the current baseline
for the r31 comparison rather than claiming every historical measurement is unchanged.

Faithful sampled-policy replays, same entry state/RNG/torch seeds as measured failures:

- `1_44w_0`, seed 62: 2,000-action timeout, 1,971-action quiet stretch. All 1,822 actions at quiet-tail ≥150 still
  had three living Goriyas and FIGHT objective. Link stayed by the left wall; the Goriyas remained at x=208.
  Quiet actions: WAIT 760, SWORD 694, MOVE 368. Existing stall cost totaled −27.47 during that interval, without
  breaking the stalemate. This validates testing a stronger fight-only quiet penalty, not merely counting WAIT.
- `1_23s_0`, seed 5: same 2,000-action timeout but no enemies remained during its 1,753 late quiet actions.
  Objective was TREASURE; 1,751 MOVE actions failed to collect the key. This is a distinct post-fight pickup problem,
  not evidence of shadow-Link avoidance and not something a fight-only penalty can fix.

Raw per-action positions, chase target/flag, enemies, objective and costs: `models/r31-block-evidence/corner-replays.json`.

Prepared an independent default-zero `reward.quiet_enemy_penalty` in the dev worktree: additional per-action cost
after the existing 150-action grace, based on the shared actual-progress tail, only while targetable living enemies
remain. Damage and round trips do not reset this grace; inactive/burrowed/dying enemies and post-fight pickup time
do not trigger it. Main is unchanged while r31 trains.

Live replay with coefficient 0.05 keeps the same sampled-policy trajectories: the 0x44 stalemate receives −91.05
additional cost; the 0x23 post-fight wander receives zero additional cost. This proves gating/reward behavior,
not that training improves. Output: `models/r31-block-evidence/corner-replays-quiet005.json`.

### Successful-gap guard for the later truncation experiment

Historical successful chain visits across the 120 tracked baseline runs: 656 stalfos-keese-gel visits had worst
quiet gap 94 actions; 84 Goriya visits had worst 126, with none reaching 200. D2 rope had 263 successful visits,
five reaching 100 and three reaching 200 (worst 2,410). A blanket 200 cutoff would discard real rope successes.
Standalone Goriya also exposes a different distribution: 0x23 would lose 12/46 successful episodes at 200, versus
1/44 in 0x44, 1/43 in 0x1e and 0/45 in 0x5e. Measure by family/room and compare training effects before enabling
thresholds; current promoted configs retain no measured cutoff.

Quiet-fight control verification after retaining Lee's newer GUI changes: **462 full-suite tests passed in
231.79 seconds** under concurrent r31 training. The new real-emulator regression checks the exact grace boundary
and removes targetable enemies without clearing the long quiet tail: the additional penalty stops after the fight.
No model improvement is claimed from this control verification.

Additional real weapon-option reward proof: an injected outgoing hostile boomerang at x=180 hits Link's shield
on frame 17 of `BEAMS E`, after its 16 control frames. No health loss; the unmocked full gameplay critic pays
`reward-block-projectile=0.5` (alongside the ordinary attack-miss cost). Replaced the mocked action-routing test
with this native settling-frame regression; removed fake legacy-sound fields, a repeated positive-multiplier row
and private-counter assertions. Reward suite: **25 passed in 2.55 seconds**. No live runtime code/YAML changed.

### Completed cohort evaluation and frozen-team selection

All three r31 arms finished 245 updates / 1,002,540 additional actions. Fixed seed-0 evaluation uses the canonical
Goriya task/current detector, 20 episodes per start state (240 total), without enabling nearby-item shaping or
measured truncation. Evaluation outputs: `models/r31-block-evidence/{promoted,r31-block1,r31-block3,r31-quiet005}-eval.json`.

| Policy | Successes / 240 | Real health change | Confirmed blocks | Mean longest quiet actions | Worst quiet actions |
|---|---:|---:|---:|---:|---:|
| Promoted r21 | 178 | −0.96 | 45 | 86.4 | 1,971 |
| Ordinary block reward | 184 | −0.77 | 45 | 45.1 | 1,228 |
| Triple block reward | 182 | −0.87 | 37 | 24.1 | 1,404 |
| Ordinary + 0.05 quiet-fight cost | 189 | −0.85 | 45 | 22.4 | 497 |

The quiet arm improves D1 0x44 versus ordinary (49/60 versus 43/60; −0.775 versus −0.942 hearts), but costs more
health in D2 0x5e (−1.033 versus −0.700; 48/60 versus 51/60). Triple block also reaches 49/60 in 0x44 but falls
to 42/60 in D2 0x1e versus ordinary's 49/60. More reward did not produce more measured blocks here. Section 7's
seed flaw prevents attributing these differences solely to the reward coefficients.

Frozen-team comparison completed after all training ended: each of the four Goriya policies, with every other
promoted specialist pinned, 40 runs per segment (seed 0–39). All **800 chains completed without execution errors**.
Provenance/results: `models/r31-block-evidence/{team-pointers.json,comparison.json,*-summary.json,*.jsonl}`.

| Policy | Full main / 40 | Full fairy / 40 | Post-D1 main / 40 | Post-D1 fairy / 40 | D2 / 40 |
|---|---:|---:|---:|---:|---:|
| Promoted r21 | 0 | 3 | 0 | 8 | 9 |
| Ordinary block reward | 2 | 3 | 2 | 4 | 11 |
| Triple block reward | 1 | 4 | 2 | 6 | 7 |
| Ordinary + 0.05 quiet-fight cost | 2 | 2 | 0 | 6 | 13 |

**Promoted r31-block1.** Lowest evaluation damage (−0.769 versus old −0.958 hearts), better evaluation success,
two full-main successes versus zero, unchanged full-fairy successes and a modest D2 gain. The post-D1 fairy
result regresses 8→4/40; these are mixed small-sample chain results, not a statistically established dominance.
Ordinary is the conservative damage-first choice over triple/quiet, neither of which wins consistently.
Keep block multiplier 1, additional quiet cost 0, nearby-item shaping off and measured truncation disabled.
Per-run `eval.json` copies retain the measured cohort reports; no other model was promoted.

All three new models reduce D1 0x44 deaths 15→13 of 28 visits at the same arrival health. Ordinary's longest
quiet stretch there falls 1,950→27 actions in both full routes, but it does not eliminate D2 stalls: 0x5e reaches
1,792 quiet actions post-D1 fairy and 0x1e reaches 1,207 in a successful full-main chain. Quiet's D2 segment
also stalls for 1,820 actions despite its better standalone worst of 497. Do not equate aggregate eval with
chain behavior, or describe every remaining stall as shadow-Link avoidance.

Faithful native chain replays exactly match the saved room/health/time/outcome records:
- Quiet policy, D2 seed 3, 0x1e: last 1,000 actions all MOVE objective with **zero enemies**, Link fixed at
  (162,93); 996 MOVE and four WAIT actions. Additional living-enemy cost cannot address that post-fight failure.
- Ordinary policy, post-D1 fairy seed 16, 0x5e: last 1,000 actions all MOVE objective with **zero enemies**,
  all MOVE actions, largely pacing across y=93 instead of exiting. Not a shield-block stalemate.

Raw replay traces: `models/r31-block-evidence/chain-stall-replays.json`. Post-fight navigation/curriculum and the
measured-cutoff trial remain open; the quiet-fight lever is not silently broadened to penalize pickups or navigation.

## 7. Training seed integrity finding (r31 prerequisite)

The two completed arms both received `--seed 31`, but first-update episode/health metrics already differed.
Code inspection finds that `train.main` seeds only parent Torch; spawned `_worker_loop` seeds neither its policy
sampler nor environment, and `RolloutBuffer.collect` starts with bare `env.reset()`. `ZeldaEnv`'s private RNG and
the game RNG are only seeded when reset receives an explicit seed. Thus the r31 runs match budget/configuration,
not worker random streams. Do not attribute a policy difference solely to the reward coefficient.

Fixed-seed evaluation remains valid: `_play` explicitly seeds Torch and resets the environment with each episode
seed. Completed results: ordinary block arm 184/240, −0.77 hearts; triple block arm 182/240, −0.87 hearts. Triple
helps D1's 0x44 in this comparison, while ordinary is better in both D2 rooms. Keep empirical policy selection
separate from causal reward claims; wait for the quiet-fight arm and matched frozen-team chains before promotion.

Fix seed propagation after the live cohort finishes, before later A/B trials. Seed each spawned policy sampler
and initial environment reset by worker index, preserve advancing streams across subsequent episodes, and verify
actual rollout reproducibility without curriculum. The existing shared asynchronous curriculum also depends on
worker completion order; seeding alone will not promise bitwise-identical curriculum or GPU training.

Reproduced before fixing, with curriculum and GPU nondeterminism removed: two fresh CPU CLI invocations, same
checkpoint/config, seed 31, two workers, one 128-action update, explicit `1_44w_0`, no curriculum, CPU thread counts
fixed to one. Run A completed no episode; B died after 43 actions/640 frames, losing three hearts. Action counts
also differed (A: WAIT/MOVE/BEAMS/SWORD = 3/78/1/46; B: 6/77/2/43). Evidence preserved in
`models/r31-block-evidence/seed-before.json`. This is a real seeded-gameplay failure, not merely differing timing.

After the cohort's training finished, corrected propagation in dev: CLI seed reaches `PPO.train`, which seeds
training RNGs; each spawned worker seeds Python/NumPy/Torch and its first environment reset using its indexed
stream. In-process collection also seeds its first reset. Later episode resets and rollout batches do not reseed.
Torch-normalized unsigned seeds avoid overflow at the worker offset; NumPy receives its supported 32-bit seed.

The new real-game CLI regression failed before the fix: repeated two-worker runs differed in episode counts,
actions, reset health, frames, rewards and lingering. It now also covers in-process training with the shared emulator.

Native worker-pool smoke (no curriculum, CPU, r21 Goriya weights, two workers, two 64-action batches each,
eight-action episode cap): two seed-31 invocations match every observation/action/reward/frame/boundary/mask tensor;
seed 32 differs. The workers' action streams differ, and both workers' reset hp fractions vary over 16 episodes,
proving streams advance across episode and batch boundaries rather than replaying the first draw. Full digests and
reset hp sequences: `models/r31-block-evidence/seed-after.json`.

The original two fresh CLI invocations now both produce WAIT/MOVE/BEAMS/SWORD = 2/59/10/57 over 128 actions.
CUDA smoke also passed: two updates / 256 actions, two completed successful episodes, mean health change −0.75
hearts. This is path verification, not a candidate model or a claim of deterministic CUDA training. Lee's player
was not killed. Shared asynchronous curriculum and resumed RNG continuation retain the limits stated above.

Integrated dev suite after the seed fix: **462 passed in 204.83 seconds**, including actual single-process and
spawned two-worker gameplay reproducibility regressions. Temporary smoke scripts/checkpoints were removed;
the recorded native-rollout evidence remains in the normal ignored model root.

## 8. Later-experiment prerequisites found during the frozen-team comparison

Read-only evidence review while r31 chains run; no source, state, model or configuration changed for this review.

**Cutoff correctness:** the §6 live smoke proved env flags, no terminal penalty and runner cutoff, not PPO value
bootstrapping. Current `RolloutBuffer.collect` combines terminated/truncated into one done flag, discards the
terminal observation on reset, and `frame_aware_gae` masks its value. Thus the intended measured-cutoff target
`reward + gamma**(frames/8) * V(final_observation)` is missing. Correct that path before any threshold training,
stop the trace across reset, and preserve the existing max-step treatment. A bootstrap from the next episode's
reset observation would also be wrong. Required regression boundaries: nonzero final value, truncation at the
rollout edge, true termination and reset isolation. This is a source-backed finding, not a runtime reproduction.

Expanded successful-gap evidence comes from completed r29 baselines and r30-exit full/post-D1/D2 JSONL artifacts,
including `promoted-full-{main,fairy}.jsonl` seeds 40–59. Successful Stalfos/Keese/Gel visits reach 670 quiet
actions, Goriya 350 in chains (1,274 standalone), and Rope 2,410. No uniform 100–300-action threshold preserves
these measured successes. Moblin's inspected successful maximum is 63 actions/308 frames, making 100 a trial
candidate, not a proven-safe setting. Pairwise main/fairy records often duplicate pre-divergence visits and must
not be pooled as independent evidence. These are historical/current promoted teams, not the live r31 candidates.

**D2 route/state integrity:** `game.yaml` already declares five Ropes and a key in 0x7e; generic FIGHT/TREASURE
objectives already apply. The actual change is canonical routing through E then N, not inventing a key objective.
`entry_states.cross_room` overrides success with target-room arrival, including `exit-wrong`; this cannot prove
required key collection. `step_config` also omits item masks, defaulting bombs on. Before rebuilding the requested
suffix, enforce correct required pickups/inventory and milestone bomb masking. Rebuild all three chains from
0x7d, introduce `2_6es_*` instead of stale W-entry `2_6ew_*`, and validate the carried key counts/collection flags
through Dodongo entry at four hearts. Refresh affected Rope and D2 Goriya curriculum tracks: matching filenames
do not invalidate stored snapshots. No binary state inventories were measured in this review.

## 9. Nearest-heart/clock-or-exit trial (r32)

Started after r31 selection and the verified seed fix merged into main (`f5cc63b`). Families: stalfos-keese-gel,
d2-rope and ow-moblin; each has `r32-nearby-off` and `r32-nearby-on`. Both arms continue the same promoted
checkpoint for that family. Only `reward.nearby_items` differs; current health mix, task, max steps, item masks
and curriculum settings remain unchanged. Bombs off; no measured cutoff or extra quiet-fight cost.

Each arm: seed 32, four workers, 245 updates ×4,096 = 1,003,520 additional actions, CUDA, no success early stop.
At most four train/eval jobs run at once. Config copies live in each run directory; source/seed/protocol snapshots
at `models/r32-nearby-evidence/protocol.json`. Runtime main source, canonical specialist YAML and entry states
stay frozen through all training and background evaluations. Lee's player has not been killed.

All three old promoted checkpoints are re-evaluated under current canonical YAML; each new model uses the same
common evaluation task, 20 episodes per start state, seed 0, three CPU workers. Selection must compare success,
real/total health, hearts taken/left and quiet gaps per room, then both full routes and relevant post-D1/D2 chains
with pinned teams. Nearby-item PBRS changes the learning potential, not runner inputs or accepted exits. Shared
asynchronous curriculum remains scheduling-dependent despite correct seed propagation. No r32 model promoted.

### Completed original cohort evaluations

All six 245-update arms and three current-YAML baselines completed. Paired on/off evaluation records match
initial state, seed and starting health in all 680 SKG, 420 Rope and 280 Moblin episodes. The treatment does
not improve heart collection here; neither continued SKG/Rope arm beats its old model on damage.

| Policy | Successes | Real ΔHP | Total ΔHP | Hearts dropped | Taken | Left | Worst quiet actions |
|---|---:|---:|---:|---:|---:|---:|---:|
| SKG old | 667/680 | −0.094 | −0.076 | 60 | 38 | 17 | 420 |
| SKG off | 661/680 | −0.121 | −0.107 | 48 | 27 | 20 | 179 |
| SKG on | 668/680 | −0.122 | −0.109 | 41 | 20 | 19 | 457 |
| Rope old | 397/420 | −0.286 | −0.225 | 181 | 87 | 73 | 129 |
| Rope off | 394/420 | −0.321 | −0.269 | 168 | 76 | 82 | 118 |
| Rope on | 391/420 | −0.326 | −0.289 | 154 | 56 | 87 | 258 |
| Moblin old | 272/280 | −0.195 | −0.195 | 0 | 0 | 0 | 37 |
| Moblin off | 271/280 | −0.166 | −0.166 | 0 | 0 | 0 | 38 |
| Moblin on | 276/280 | −0.173 | −0.173 | 0 | 0 | 0 | 34 |

Heart columns are cohort totals, health columns per-episode means. Moblin's higher success is not evidence
of heart-seeking: none of these evaluations offers a dropped heart. The common PBRS code measures both old/new
positions on the previous state's wavefront; inspection does not support a pickup-disappearance potential jump.
Full reports: `models/r32-nearby-evidence/evaluations.json` and each run's `eval.json`.

### Completed frozen-team chains

All **1,200 chains completed without execution errors**: six pinned teams, 40 seeds (0–39) per segment, both
full routes, both post-D1 routes and D2. No source, canonical YAML, state or promotion mutation. Each isolated
on swap below starts from the all-off team, not the promoted team.

| Team | Full main /40 | Full fairy /40 | Post-D1 main /40 | Post-D1 fairy /40 | D2 /40 |
|---|---:|---:|---:|---:|---:|
| Promoted | 2 | 3 | 2 | 4 | 11 |
| All off | 2 | 2 | 2 | 5 | 16 |
| All on | 1 | 0 | 2 | 5 | 11 |
| Only SKG on | 1 | 2 | 2 | 5 | 16 |
| Only Rope on | 1 | 1 | 0 | 3 | 11 |
| Only Moblin on | 0 | 5 | 2 | 4 | 16 |

Repeated D2 results for all-off/SKG-only-on/Moblin-only-on are the same evidence: those other two families do
not play the D2 segment. Likewise SKG swaps do not affect post-D1 chains. Do not pool duplicated paths or the
shared part of route pairs as independent samples. Full reports: `models/r32-nearby-evidence/comparison.json`.

D2 therefore isolates the Rope policy. Its extra **unshaped** training improves physical chain health and
collection, despite slightly worse standalone health. These health changes sum all Rope visits within a run,
then average over all 40 runs; they are not per-visit means or survivor-only statistics.

| Rope policy | Rope deaths | Real ΔHP/run | Total ΔHP/run | Hearts taken | Hearts left |
|---|---:|---:|---:|---:|---:|
| Old r28 | 18 | −2.288 | −2.113 | 29 | 34 |
| R32 off | 6 | −1.400 | −1.138 | 40 | 27 |
| R32 on | 15 | −1.925 | −1.775 | 25 | 47 |

The paired old/off success transition has 12 new wins, seven lost wins and four shared wins; the aggregate
chain gain is still small-sample evidence, not established dominance. Moblin off is unsuitable as a team-wide
upgrade: post-D1 fairy Moblin deaths rise from two to thirteen, with per-visit real health −0.279 → −0.588.
SKG on has more chain heart collection than off, unlike its standalone result; do not call the treatment
uniformly ineffective based on the standalone cohort alone. Neither grouped team is selected for promotion.

### Isolated candidates and independent-seed reversal

Completed 360 isolated selection chains with old Moblin and other teammates pinned, then 120 D2 validation
chains on disjoint seeds 40–79, and 160 isolated shaped-Rope full/post-D1 chains. All finished without execution
errors. These are **1,840 chain executions** including the original cohort, not 1,840 independent observations;
the repeated seed-0 Rope-off D2 segment is reproducibility/isolation evidence only.

| Isolated candidate | Full main /40 | Full fairy /40 | Post-D1 main /40 | Post-D1 fairy /40 |
|---|---:|---:|---:|---:|
| Rope off, old SKG | 0 | 3 | 3 | 7 |
| Rope off + SKG off | 0 | 3 | — | — |
| Rope off + SKG on | 1 | 3 | — | — |
| Rope on, old SKG | 1 | 1 | 3 | 7 |

The new D2 seeds **reverse the initial Rope-arm ranking**. Combined health averages use disjoint seeds 0–79,
sum all Rope visits per run and include failures. Do not select the off arm solely from the first 40 seeds.

| Rope policy | New-seed wins /40 | Combined wins /80 | Real ΔHP/run | Total ΔHP/run | Hearts taken | Hearts left |
|---|---:|---:|---:|---:|---:|---:|
| Old r28 | 10 | 21 | −1.944 | −1.706 | 73 | 73 |
| R32 off | 10 | 26 | −1.600 | −1.419 | 58 | 69 |
| R32 on | 22 | 33 | −1.556 | −1.369 | 66 | 91 |

Shaped Rope now leads D2 success and real/total health, but not aggregate heart pickup or both full routes.
Keep old SKG/Moblin rather than adopting the inferior standalone-damage SKG arms or the harmful Moblin off
continuation. No candidate is promoted before the remaining Octorok comparison and the objective defect below
are resolved. Evidence: `selection/`, `validation/`, `selection-on/` under the r32 evidence root.

### Faithful stall diagnosis before cutoff training

Three native replays match every saved room/health/frame/outcome field in the original all-off D2 records:
- Seed 18, 0x6e: optional combat, two active Ropes remain, north physically open, finite wavefront distances;
  final 1,000 actions are MOVE objective and mainly pacing at y=93. This is navigation stalling.
- Seed 8, 0x6e: the same open-exit MOVE task eventually succeeds after 978 actions, including a 953-action
  quiet gap. A blanket 100/200 cutoff would discard this successful old-policy path.
- Seed 10, 0x2e: one active Rope remains after seven kills, two keys, north **SHUTTER**, ALL_DEAD trigger,
  physically closed, north absent from live exits. Yet the objective is MOVE toward N for the final 1,000 actions;
  its wavefront has no reachable target. This is a required-combat objective omission, not merely slow navigation.

Playable snapshot inspection also identifies a separate classifier error: closed north tile 0xa8 is excluded
by `BARRED_DOOR_TILES` (old 0xac–0xaf), so the tile helper calls the shutter a wall and not barred. Native audit
of 73 configured dungeon states finds vertical 0xa8 and horizontal 0xac closed shutters. Do not add a room-id
exception or force optional fights in 0x6e/0x6d; correct live shutter classification and route-gated combat before
using timeouts to mask this failure. All r32 candidate comparisons above still use unchanged old main source.
Evidence: `rope-stall-replays.json`, `rope-stall-geometry.json`, `shutter-tile-audit.json` and playable PNG/state
snapshots in the r32 evidence root. Source correction is isolated in dev; main stays frozen for Octorok.

**Terminal-health warning discovered in the replay audit:** seed 10's stuck-room record says 1.0 HP out,
but its executed trace/snapshot has 1.5 HP. Legacy room/run step-budget exits do not refresh `health_out`.
Thus the recorded HP tables above include stale entry health for stuck visits; success counts remain valid,
but health ranking must be remeasured after correction. Retain the original evidence; do not impute missing
damage from pickups alone. This is another prerequisite before promotion/cutoff comparisons.

### Pickup-exposed overworld follow-up

Added ow-octorok off/on arms, same promoted r24-fairy warm start and r32 seed/budget/workers/health setup,
plus a current-YAML baseline. Runs remain `models/ow-octorok/r32-nearby-{off,on}`; separate evidence/protocol:
`models/r32b-octorok-evidence/`. It shares unchanged main source with the frozen-team comparisons. Selection
reason: the completed r29 full-chain artifacts contain four Octorok heart drops in 387 visits versus zero in
46 Moblin visits (route pairs are not independent). This follow-up avoids calling the overworld heart question
answered by a zero-opportunity evaluation. No Octorok or other r32 candidate is promoted yet.

Both added Octorok arms and the old-model baseline completed. On/off records match initial state, seed and
health in all 620 paired episodes. There is real but sparse heart exposure; shaping still does not increase
pickup here. Success differences are one episode, not evidence of a robust success gain.

| Octorok policy | Successes | Real ΔHP | Total ΔHP | Hearts dropped | Taken | Left | Worst quiet actions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Old r24-fairy | 614/620 | −0.078 | −0.074 | 13 | 4 | 9 | 308 |
| R32 off | 614/620 | −0.064 | −0.061 | 12 | 3 | 9 | 94 |
| R32 on | 615/620 | −0.070 | −0.068 | 12 | 2 | 10 | 62 |

Health columns are per-episode means; heart columns cohort totals. Both continuations reduce measured damage,
off more than on, but require chain selection before replacing r24. Reports: `r32b-octorok-evidence/evaluations.json`.

### Completed corrected-runtime comparisons

The original main-source freeze ended only after the Octorok training/evaluation jobs completed. Verified
bootstrap/config transport, entry inventory guards, shutter objectives and terminal health were then merged
from dev to main (`4cb8434`), with all 470 tests and native smokes passing. No model or canonical YAML changed.

Completed comparisons pin that corrected runtime and unchanged teams/configs/states:
- Rope old/off/on: common 420-episode task per policy, then 80 D2 seeds 0–79 per policy; source checkout dev,
  explicit main model paths. Evidence/protocol: `r32-nearby-evidence/corrected-runtime/`.
- Octorok old/off/on: both full and both post-D1 routes, 40 seeds 0–39 per segment, every other old promoted
  specialist pinned; source checkout main. Evidence/protocol: `r32b-octorok-evidence/chains/`.
- Shaped Rope alone: both full and both post-D1 routes, 40 seeds 0–39 each, all other old promoted weights
  pinned. Evidence: `r32-nearby-evidence/final-selection/rope-on/`.

The corrected evaluations, not the legacy timeout-HP aggregates, decide selection. All 880 corrected-runtime
chain executions completed without execution errors; route pairs/shared prefixes are not independent samples.

| Rope policy | Eval successes /420 | Eval real ΔHP | Eval total ΔHP | D2 wins /80 | Rope real ΔHP/run | Rope total ΔHP/run | Taken | Left |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Old r28 | 394 | −0.305 | −0.248 | 27 | −1.881 | −1.638 | 82 | 66 |
| R32 off | 394 | −0.349 | −0.301 | 28 | −1.656 | −1.475 | 50 | 64 |
| R32 on | 388 | −0.336 | −0.295 | 35 | −1.663 | −1.494 | 66 | 95 |

Eval health is per episode; D2 health sums all Rope visits per run, including failed runs, then averages over
80 disjoint seeds. Taken/left are D2 cohort totals. Shaped Rope wins more D2 runs but does not improve pickup
or standalone success/damage; off has essentially equal D2 damage. Corrected full chains reject a general
Rope upgrade:

| Isolated swap | Full main /40 | Full fairy /40 | Post-D1 main /40 | Post-D1 fairy /40 |
|---|---:|---:|---:|---:|
| None: promoted team | 2 | 3 | 1 | 2 |
| Rope on | 0 | 1 | 2 | 4 |
| Octorok off | 2 | 6 | 4 | 4 |
| Octorok on | 1 | 3 | 2 | 4 |

Octorok health below is that family's aggregate real/total ΔHP per run, not survivor-only or per-visit means:

| Octorok policy | Full main real/total | Full fairy real/total | Post-D1 main real/total | Post-D1 fairy real/total |
|---|---:|---:|---:|---:|
| Old r24-fairy | −0.238 / −0.238 | −0.113 / −0.100 | −0.350 / −0.325 | −0.500 / −0.475 |
| R32 off | −0.300 / −0.300 | −0.075 / −0.075 | −0.388 / −0.388 | −0.463 / −0.450 |
| R32 on | −0.175 / −0.175 | −0.100 / −0.088 | −0.325 / −0.313 | −0.338 / −0.338 |

Neither Octorok treatment establishes a heart-pickup benefit. Independent old/off validation then completed
320 chains on seeds 40–79, all without execution errors (`r32b-octorok-evidence/validation/`). Its new-seed wins
are old/off 1/2 full main, 2/2 full fairy, 1/1 post-D1 main and 8/7 post-D1 fairy: small, mixed holdout changes,
not an independently decisive win. Combined with the disjoint first cohort, the ordinary continuation has
modestly higher success and lower physical Octorok damage on all four segments:

| Policy | Full main wins /80 | Full fairy wins /80 | Post-D1 main wins /80 | Post-D1 fairy wins /80 |
|---|---:|---:|---:|---:|
| Old r24-fairy | 3 | 5 | 2 | 10 |
| R32 off | 4 | 8 | 5 | 11 |

| Policy | Full main real/total | Full fairy real/total | Post-D1 main real/total | Post-D1 fairy real/total |
|---|---:|---:|---:|---:|
| Old r24-fairy | −0.269 / −0.269 | −0.131 / −0.125 | −0.413 / −0.400 | −0.475 / −0.463 |
| R32 off | −0.244 / −0.244 | −0.125 / −0.125 | −0.356 / −0.356 | −0.394 / −0.388 |

Health is Octorok ΔHP per run, including failed runs. All other weights stay pinned. Route pairs/prefixes are
correlated; this is a conservative model-selection decision, not a significance or causal-training claim.
Promoted **ow-octorok/r32-nearby-off**; unchanged standalone success with lower damage supports the same
choice. Keep old Rope/SKG/Moblin, no shaped model promoted, nearby-item PBRS remains disabled. No combined
losing-Rope/Octorok team is selected. Persisted aggregate: `validation/combined.json`.

**Remaining corrected-runtime fight stall:** a native seed-15 replay exactly matches the complete saved shaped
Rope D2 record. In 0x2e the last 1,000 actions all have FIGHT, finite wavefront distance 1–12, and a physically
closed north shutter. Two one-HP Ropes stay at (32,127)/(208,116) throughout; clock is active in every sample.
Link moves on 888 actions but uses 978 MOVE versus 11 SWORD choices, with a 2,942-action quiet tail. This is
fighting failure against clock-frozen foes, not the repaired MOVE-through-shutter defect or evidence of
shadow-Link chasing. Clock is already observed; log a clock-active combat curriculum idea rather than add a
redundant flag or silently suppress clock pickup. Evidence: `corrected-runtime/fight-stall-replay.json`.



## 10. Measured-cutoff bootstrap correction (dev prerequisite)

Reproduced the §8 value-target gap with a permanent collector regression: a truncation inside a batch, another
at its edge, distinct terminal/reset values, and a later batch containing a legacy time limit, true death and
open continuation. Before fixing, targets `[1.0, 3.6, 3.0]` should have been approximately
`[2.75, 6.145584, 9.363961]`. Env flags alone had not delivered the promised bootstrap.

Dev now marks only `no-room-progress` with `info.bootstrap_truncation`; the collector adds the frame-discounted
actual final-observation value before resetting. The done flag still cuts the GAE trace; it never substitutes
the next episode's reset value. Legacy max-step/death behavior and raw logged reward are unchanged. No cutoff
is enabled in canonical configs. Main remains frozen on r32's runtime; this correction is not merged mid-cohort.

Real Zelda smoke: three WAIT 8 actions in `1_73s_0`, actual nine frames each, cutoff at action three. With a known
constant critic value 10, gamma 0.9 and lambda 1, learning targets are `[7.007587, 7.889431, 8.882247]`, matching
the frame-aware calculation; raw logged reward remains zero. The same three actions under the legacy max-step
cap still produce `[0,0,0]`. Both are playable mode 5 after automatic reset. Evidence:
`models/r33-cutoff-evidence/bootstrap-smoke.json`. This proves collector integration, not a trained model gain.

Current r31-block1 standalone successful-gap exposure differs from the older baseline: 1:0x23 has 41 successes,
worst 79 quiet actions; 1:0x44 has 43 successes, worst 901 (one would be cut at either 100 or 200); 2:0x1e has
49 successes, worst 154 (two at 100, none at 200); 2:0x5e has 51 successes, worst 60. These retrospective counts
do not establish a safe production threshold, and the saved full-main chain also has a 1,207-action successful
gap. Re-measure whichever model is used for each trial, rather than treating historical thresholds as permanent.

Integrated dev suite: **463 passed in 779.75 seconds** while r32 occupied the main training workers. The new
value-target regression passes, including legacy truncation/death and open-rollout controls. Temporary native
smoke script removed; persisted JSON evidence retained. No r32 runtime source or canonical config changed.

### Current-policy threshold exposure for the next trial

The completed current-YAML baselines refine the earlier historical screening. Counts below are retrospective
episodes whose longest quiet stretch reaches the threshold, not live cutoff results or predicted retraining gains.

| Family | Successes / failures | Successful ≥100 | Successful ≥200 | Failed ≥100 | Failed ≥200 |
|---|---:|---:|---:|---:|---:|
| Goriya r31-block1 | 184 / 56 | 3 | 1 | 10 | 8 |
| Stalfos/Keese/Gel r25 | 667 / 13 | 3 | 1 | 0 | 0 |
| Rope r28 | 397 / 23 | 1 | 0 | 0 | 0 |
| Moblin r26c | 272 / 8 | 0 | 0 | 0 | 0 |

Use Goriya off/100/200 for the next matched threshold study: it has observed failed-episode stall exposure,
whereas a 100-action Moblin cutoff would not address any failure in this standalone sample. Full-chain successes
still contain longer gaps, so these counts do not justify enabling a production cutoff. Compare trained weights
on a common uncut task as well as their actual cutoff task, then pass each arm's config into frozen-team chains;
otherwise altered episode denominators or dropping the runtime cutoff would confound selection. Study not started;
r32 remains active. Evidence with per-room rows/source provenance:
`models/r33-cutoff-evidence/current-baseline-cutoff-exposure.json`.


## 11. Per-arm chain configuration (dev prerequisite)

Added repeatable `triforce eval-chain --specialist-config PATH.yaml`, keyed by each loaded config's family name.
The loaded objects reach `Runner` in both direct and spawned execution; `_Specialist` no longer reloads a
default config over an override. Programmatic callers use `specialist_configs={name: loaded_config}`.
Canonical YAML, promoted pointers, route selection and carried health/inventory remain untouched. This supports
the cutoff and later boomerang comparisons without mutating a live experiment's global config.

Native smoke uses a valid temporary boomerang-biased neural policy in D2 0x5e. The first attempt correctly found
boomerang masked in the south doorway; after two scripted northward moves, enabled config chooses BOOMERANG,
disabled config chooses WAIT instead. The unchanged cutoff is honored too. Ordinary waiting runs with cutoff 2
produce exactly matching native room/health/frame/outcome records through direct API, CLI parallel 1 and CLI
parallel 2. Evidence: `models/r33-cutoff-evidence/config-override-smoke.json`. These are smoke policies, not
trained candidates. The spawned real-game CLI regression passed (two runs, each stopped at two quiet actions).
Integrated dev suite after config transport: **464 passed in 871.42 seconds**. Temporary neural checkpoints,
YAML files and smoke script were removed; JSON evidence remains. This feature stays in dev until r32's source
freeze ends; no canonical specialist config or promoted pointer was changed by these checks.

## 12. D2 key-builder defect reproduced (before route migration)

Native `cross_room` from clean `2_7ew_0` toward 0x6e, seed 0, 3,000 exploration iterations/200 patience/one
attempt: accepted a 56-action path after 6,293 exploratory actions (68.6 seconds), ending `exit-wrong` at 1.5
hearts. Link entered 0x6e with **zero keys and 0x7e's key still untaken**, playable mode 5. Source also had zero
keys and an untaken key. This confirms the §8 acceptance hole through real gameplay; no bad state file was kept.
Evidence: `models/r34-d2-key-evidence/builder-before.json`.

Separate native prototype, same source/seed/search limits, with required source-room item-taken gate and bombs
masked: accepted a 155-action **exit-correct** path after 14,886 exploratory actions (202.3 seconds), arriving at
three hearts with **one key and 0x7e's pickup flag set**, playable mode 5. Evidence:
`models/r34-d2-key-evidence/builder-prototype-after.json`. This validates the guard through real collection,
not an injected inventory or metadata-only check.

Permanent dev builder now enforces the declared source-room key pickup during discovery **and exact replay**;
its milestone step config masks bombs. Actual `cross_room` smoke succeeds for all three `2_7ew_0..2` sources:
155/103/149-action paths, each entering 0x6e with one key and the pickup flag set. Reloading each captured
pre-entry byte payload through `GameSession.load` preserves that inventory, four max hearts and playable mode 5.
Evidence: `models/r34-d2-key-evidence/builder-after.json`. Affected entry/exploration/room-entry suites:
**158 passed in 33.23 seconds**. The previously integrated full suite was 464; no new full-suite result is claimed
for this subsequent guard. Slow natural exploration remains a throwaway smoke rather than a permanent search
test. No permanent state was replaced, and no r32 runtime source was changed.

Native inventory audit of all 29 existing numbered D2 files (through `GameSession.load`, mode 5): every file
has four hearts, wooden sword/boomerang and exactly the D1 triforce piece. All 0x7d/0x7e entries have zero keys
and an untaken 0x7e key. Chains 0/1 also reach 0x3e, 0x2e, 0x1e and Dodongo entry with **zero keys and both
0x4e/0x3e pickup flags still unset**. Chain 2 instead has one key at 0x3e, two at 0x2e and one at 0x1e, with
both earlier pickups set. Its 0x2e E-door-open bit is set at the later entry; the route's N door is not locked,
while the side E door is. Thus room pickup flags alone do not prove the carried-key count the rebuilt route needs;
the suffix rebuild must check inventory balance and reject unneeded side-door spending. No 2_0es_2 file exists.
Full inventory, current locked doors and persistent room flags: `models/r34-d2-key-evidence/inventory-before.json`.
No state was mutated by this audit.

### Carried-key balance guard

Reproduced the inventory acceptance hole using the builder's actual predicate and existing native state files:
source `2_2es_2` has two keys and an unlocked N exit, but the original predicate accepts `2_1es_2` with one key
and 0x2e's side door opened. Both states load in playable mode 5. This is a predicate reproduction on real
saved inventories, not a new before-fix exploratory trajectory. Evidence: `key-balance-before.json` in the §12 root.

The dev guard now requires source keys + any newly required source-room key − the direct target door's initial
key cost, in discovery and replay. It rejects the historical one-key candidate while preserving intended
locked/off-route destinations. Native exploration and reload of captured pre-entry payloads prove:

| Source → target | Actions | Keys before → after | Result |
|---|---:|---:|---|
| 2:0x2e → N 0x1e | 159 | 2 → 2 | Correct exit; rejects the old one-key candidate |
| 2:0x2e → E 0x2f | 68 | 2 → 1 | Intended locked off-route exit still accepted |
| 2:0x7e → N 0x6e | 155 | 0 → 1 | Required key physically collected |
| 1:0x33 → N 0x23 | 121 | 1 → 1 | Off-route: collects the source key, spends one on the locked target door |

Every reloaded payload has route-correct maximum hearts (D2 four, D1 three) and mode 5. Evidence:
`key-balance-{north,east,pickup,locked-pickup}-after.json`.
Affected state-builder/exploration/entry suites: **158 passed in 44.60 seconds**. No permanent state, main
runtime source, specialist YAML or promoted pointer changed. Integrated dev suite after both builder guards:
**464 passed in 599.75 seconds**. The temporary native probe script was removed; JSON evidence remains.


## 13. Required shutter combat corrected in dev

The §9 native failure has two independent causes: north shutter tiles 0xa8–0xab were missing from the barred
tile catalog, and null-treasure dungeon rooms never selected combat to clear a required shutter. Both are now
corrected in dev without room-id exceptions or additional metadata. The selector checks the routed exits,
requires all alternatives to be barred, then uses the existing decoded native living-monster/secret-trigger
semantics. Block/money triggers and non-foes do not force fights. Open-route 0x6e and the irrelevant west
shutter in 0x6d remain MOVE tasks with living Ropes.

Permanent native regressions reproduced two failures before the fix (north classification and required FIGHT),
with both optional-combat controls passing. After correction all four pass. Integrated dev suite:
**468 passed in 120.21 seconds**. Observation/action schemas unchanged; no main runtime/YAML/state mutation.

Actual gameplay smoke starts from the saved seed-10 stalled room: one live Rope, closed north shutter, two
keys, 1.5 physical hearts. The corrected objective is FIGHT with finite walking distance 9. Go-Explore finds a
69-action exact replay (7,075 exploratory actions) that physically kills the Rope, opens the shutter, changes
to MOVE and exits north into 0x1e with `exit-correct`/success. No injected kill/key/door or synthetic policy
result. This proves the objective/gate transition, not a learned-model performance gain. Evidence:
`models/r33-cutoff-evidence/shutter-objective-smoke.json`.

Source remains isolated in dev until Octorok's old-source cohort finishes. Re-evaluate old/new Rope policies
under the corrected common objective before selecting a promotion; legacy timeout HP also needs correction
as noted in §9. Temporary native investigation/smoke scripts are removed; their JSON/state/PNG evidence remains.

## 14. Live terminal health on chain budgets

Native replay established stale HP on a legacy stuck visit (§9). Both the room-step and global run-step limits
finalized dropped-heart counts but left `RoomRecord.health_out` at the value initialized on entry. The measured
cutoff already copied live health, so comparing it against the legacy budget could also bias health results.

Dev now has one record-finishing path that captures live ending HP and heart counts for every finalized exit,
timeout and action-error record. Death stays zero. All six callers migrate to `_finish_record`; the obsolete
helper name and duplicated per-branch assignments are removed. Gameplay, limits and outcome reasons unchanged.

The two permanent native regressions fail before the fix: the fairy heals Link from one to four hearts, but
both budget paths record one. After correction all **16 runner checks pass**. Separate real Runner smoke uses
legal north/wait actions at the fairy pond: each budget ends after 12 actions/313 frames in mode 5, recorded
and physical ending HP both four, entry HP one. Only the initial hurt state is configured; healing is native.
Evidence: `models/r33-cutoff-evidence/terminal-health-smoke.json`. Integrated dev suite:
**470 passed in 113.57 seconds**.

Old chain files are not retroactively rewritten: pickups alone cannot reconstruct unrecorded damage. Before
promotion, compare old/off/on Rope on the common corrected objective/runtime and live terminal-health metrics.
Main remains frozen for the active Octorok cohort. Temporary smoke source/state are removed; JSON remains.


## 15. Matched Goriya no-progress cutoff trial (r33)

Started after final r32 selection and ordinary Octorok promotion. Main runtime is `ed2d8b1` (source correction
`4cb8434`); pinned teams include Octorok r32-nearby-off and Goriya r31-block1. Evidence/protocol and pointer
snapshot: `models/r33-cutoff-evidence/`. All canonical cutoffs remain null; nearby-item shaping stays off.

Three arms continue the identical r31-block1 weights/optimizer: `goriya/r33-cutoff-{off,100,200}`. Only
`no_progress_steps` differs. Each has seed 33, four workers, 245 updates ×4,096 = 1,003,520 additional actions,
CUDA, unchanged curriculum/full-and-partial-health mix/bomb mask, no success early stop. At most three
train/eval jobs run together. Main source, canonical YAML, entry states and promotion pointers stay frozen.

Fresh corrected-runtime uncut baseline completed: 184/240 successes, −0.77 real/−0.69 total hearts. Its successful
episode quiet-gap exposure is unchanged from §10: 0x23 worst 79, 0x44 worst 901, 0x1e worst 154, 0x5e worst 60.
Thus 100 exposes three old successful episodes and 200 one, versus ten/eight failed episodes respectively;
these are risk counts, not predictions of live cutoff performance. Baseline JSON: `corrected-goriya-baseline.json`.

Evaluate old weights under each actual cutoff; evaluate all new weights under the same uncut task and each
cutoff model under its actual task (20 episodes/state, seed 0, three workers). Then compare frozen-team full,
post-D1 and D2 chains with loaded per-arm configs. Measure success, health, steps/frames and quiet streaks;
never interpret early failed cutoffs as successful navigation or select on shorter episodes alone.

The exact promoted-team uncut chain baseline reuses completed r32 records rather than re-running them: Octorok
off's four 40-seed route segments and the first 40 corrected old-Rope D2 records. Team/source/task are unchanged;
provenance is retained in `chains/old-off/comparison.json`. Old weights with actual 100/200 configs are running
separately. After Octorok promotion, the default (no override) runner also reached D1 0x73 at full three hearts,
91 actions/2,545 frames, seed 0. No user player process was killed.

While this cohort runs, the upcoming D2 east-first route and three native conserved-key suffixes are prepared
in **dev only**, not main. No D2 model training, main activation or schema changes before cutoff selection.
Fresh trajectories wait for audited basename-addressable dev states; no alternate exploration loader is added.

### Old weights under live cutoff configs

All 400 actual-cutoff chains completed without execution errors (40 seeds ×5 segments ×2 limits), alongside
the retained exact uncut baseline. Both thresholds shorten some runs but discard successful post-D1 fairy
and D2 paths. No canonical threshold enabled; retrained-model selection remains pending.

| Segment | Uncut wins /40 | 100 wins /40 | 200 wins /40 | Uncut mean actions | 100 mean actions | 200 mean actions |
|---|---:|---:|---:|---:|---:|---:|
| Full main | 2 | 2 | 2 | 798.875 | 613.400 | 628.400 |
| Full fairy | 6 | 6 | 6 | 886.925 | 660.175 | 677.675 |
| Post-D1 main | 4 | 4 | 4 | 339.125 | 339.125 | 339.125 |
| Post-D1 fairy | 4 | 2 | 2 | 547.200 | 494.550 | 500.925 |
| D2 | 14 | 11 | 12 | 771.650 | 599.875 | 627.725 |

Means include failures; shorter time and less damage before an early failed cutoff are not a survival gain.
Route pairs share prefixes. Config preservation is exercised natively: explicit `no-room-progress` reasons
stop the room at 100/200 quiet actions. Full records: `chains/old-{off,100,200}/`.

Lost-success seeds are post-D1 fairy 22/31 at both limits, D2 9/24 at both, plus D2 28 at 100. A native D2 seed-9
200-cutoff replay exactly matches the complete saved result. Its last 200 actions are all MOVE objective/MOVE
actions, zero living enemies, clock off, north OPEN and physically open, four hearts at cutoff, finite distance
16 at the final action. Uncut eventually succeeds; this is post-fight navigation pacing, not waiting for a
Goriya or shadow-Link chase. Evidence: `cutoff-loss-replay.json`. Do not misdiagnose the removed successes as
all intrinsically failed fights or use the cutoff to conceal a required-shutter objective problem.

### Completed matched training and standalone evaluation

All three 245-update arms and all eight common/actual baseline/candidate evaluations completed without errors.
Each checkpoint stores 1,713 cumulative updates/7,014,488 actions; **1,003,520 actions are additional** in this
trial. Training walls off/100/200: 4,323/4,195/4,259 seconds. No user player was killed or main runtime altered.

| Policy / evaluation task | Wins /240 | Real ΔHP | Total ΔHP | Mean actions | Worst quiet actions |
|---|---:|---:|---:|---:|---:|
| Old r31 / uncut | 184 | −0.769 | −0.694 | 130.9 | 1,228 |
| Old r31 / 100 | 183 | −0.740 | −0.665 | 91.2 | 100 |
| Old r31 / 200 | 186 | −0.752 | −0.677 | 97.8 | 200 |
| R33 off / uncut | 193 | −0.815 | −0.769 | 94.1 | 1,801 |
| R33 100 / uncut | 189 | −0.819 | −0.763 | 83.7 | 406 |
| R33 100 / actual 100 | 185 | −0.810 | −0.754 | 78.5 | 100 |
| R33 200 / uncut | 197 | −0.729 | −0.673 | 94.6 | 1,665 |
| R33 200 / actual 200 | 196 | −0.725 | −0.669 | 83.3 | 200 |

Common/actual record tuples match start state, episode seed and initial health. Enabling this training control
**replaces** legacy Timeout (including 50 stationary actions), not simply adding an earlier abort: old-weight
100 rescues two old `stuck` episodes but loses three successes; 200 rescues three and loses one. This explains
why old-weight actual 200 can score above old uncut; do not credit the difference to retraining.

Per-room common-task comparison (successes /60; physical ΔHP per episode):

| Room | Old r31 wins / ΔHP | R33 off wins / ΔHP | R33 100 wins / ΔHP | R33 200 wins / ΔHP |
|---|---:|---:|---:|---:|
| D1 0x23 | 41 / −0.817 | 46 / −0.725 | 41 / −0.633 | 44 / −0.783 |
| D1 0x44 | 43 / −0.942 | 52 / −0.592 | 44 / −0.875 | 51 / −0.667 |
| D2 0x1e | 49 / −0.617 | 49 / −0.808 | 47 / −0.975 | 50 / −0.792 |
| D2 0x5e | 51 / −0.700 | 46 / −1.133 | 57 / −0.792 | 52 / −0.675 |

200 leads combined standalone success/health, but off has the best D1 damage and regresses D2. Long uncut
tails remain even after cutoff training. All trained-policy actual/common uncut chain comparisons are now
complete; their general chain regression overrides standalone ranking. Keep r31-block1, no canonical cutoff
enabled. Full reports: `evaluations.json`, each arm's eval JSON and `chains/`.


### Completed chain selection; cutoffs remain disabled

All **1,400 fresh chains** completed without execution errors: 400 old-weight actual-cutoff runs, 600 trained
actual-task runs and 400 trained cutoff-policy/common-uncut runs. The unchanged baseline additionally reuses
200 prior records. These are executions, not independent route-pair/prefix observations.

| Goriya policy / runtime | Full main /40 | Full fairy /40 | Post-D1 main /40 | Post-D1 fairy /40 | D2 /40 | D2 Goriya real/total ΔHP per run |
|---|---:|---:|---:|---:|---:|---:|
| Old r31 / uncut | 2 | 6 | 4 | 4 | 14 | −1.038 / −0.988 |
| R33 off / uncut | 1 | 2 | 3 | 4 | 14 | −1.638 / −1.638 |
| R33 100 / uncut | 3 | 2 | 4 | 6 | 12 | −1.450 / −1.450 |
| R33 100 / actual 100 | 3 | 2 | 4 | 6 | 11 | −1.475 / −1.475 |
| R33 200 / uncut | 1 | 3 | 3 | 4 | 12 | −1.500 / −1.500 |
| R33 200 / actual 200 | 1 | 3 | 3 | 4 | 12 | −1.500 / −1.500 |

All other weights/configs stay pinned; health includes failed runs. No candidate improves both full routes
and D2 survival/physical damage. 100's post-D1 fairy gain is mixed with full-fairy/D2 regression; 200's better
standalone score is not a chain upgrade. Its runtime cutoff only shortens eventual failures in this cohort;
100 additionally removes one new-model D2 success. No holdout promotion selection for inferior general teams.

**Keep Goriya r31-block1 and `no_progress_steps: null`.** Record this trial result in the ledger only; the
already-merged dormant control/bootstrap infrastructure is not new experiment code. No r33 model promoted.
Lingering measurement remains mandatory; shorter failed episodes are not a solved navigation/fighting skill.

## 16. Isolated D2 east-first preparation (before r34 training)

Dev route now uses 0x7d E → 0x7e key → 0x6e from S for both routes, removing the old D2 alternate without
changing enemy-family membership. Route consumers/tests/docs and Rope's S-entry state selector are updated
in dev only. Main keeps the old route/states during r33 selection; no D2 model training has started.

Native chains 0/1 completed all eight crossings from matching `2_7ds_0/1`, using existing `build_chain` at
12,000 iterations/patience 500, no inventory injection. All 16 payloads pass nine-load audit: clean pre-entry,
correct room, full enemy count, four-heart maximum; playable inspections are mode 5. Per-destination keys:
0x7e 0, 0x6e 1, 0x5e 1, 0x4e 1, 0x3e 2, 0x2e 3, 0x1e 3, 0x0e 3. Persistent flags prove every required key pickup.

All 24 suffix payloads are now audited and installed in dev. Chain 2 completed its remaining three crossings
with native `path` after the initial 0x3e Moldorm/key search failed three attempts/197,562 actions; retry budget
6,000/patience 200. It retains the same 0,1,1,1,2,3,3,3 key sequence, all pickup flags and four-heart maximum.
Evidence roots: `/tmp/d2-east-chain{0,1,2}/` manifests/audits/inventory and native build/generation logs.

Retired all three stale `2_6ew_*` states/manifest records; kept off-route 0x6d family/states with explicit origin
step 35. Canonicalized relocated path metadata, not gameplay statistics: raw path ending `exit-wrong` on the
terminal crossing stays recorded and is not called a policy success. Terminal chain 0/2 naturally carry four
bombs; preserve native pickup, mask use with `items.bombs=False`, milestone ends without fighting the boss.

Initial full suite: three off-route heart-audit failures, 467 passes. Native D2 hearts were correctly four;
the test treated off-route step −1 as game-start progress. It now derives hearts from the completed routed
origin when available; source provenance is supplied for retained 0x6d. After correction **470 passed in
112.51 seconds**. No room-id exception or lowered-heart state.

Fresh family-contained Rope/Goriya curricula for all three chains run under dev `trajectories/r34-east-key/`.
Old Rope under the new YAML/states: 335/420 (80%), −0.34 real/−0.26 total hearts; empty 0x7d is **0/60**, always
leaving the old north way. Do not activate the east-first main route under this old policy: retrain on frozen
dev, then cut over route/data/model together after verification. Main states/trajectories/pointers unchanged.

## 17. East-first Rope retraining (r34)

Verified route/state/consumer preparation is committed in dev at `ab9bda5`; all 470 tests pass. Main remains
on its old compatible route/state/model team until candidate selection. Source/YAML/states in dev now freeze
for `models/d2-rope/r34-east-key`; protocol/evidence at `models/r34-east-key-evidence/` in main's model root.

Warm-start r28-d2, seed 34, eight CPU env workers, CUDA; at most 490 updates ×4,096 = 2,007,040 additional
actions, balanced-success stop 0.99. Same max_steps 3,000, one-room task, half full health/partial hp down to
0.15, bombs off, ordinary PBRS, no nearby-item reward or measured cutoff. Rope curriculum remains disabled:
this is the requested route/state retrain, not a simultaneous new curriculum-treatment trial. Background
trajectory refresh uses only ignored dev outputs, not running training inputs or main default curricula.

Common-task old Rope baseline already completed (335/420, empty entrance 0/60); candidate gets the identical
20 episodes/state, seed 0 task after training. Then both full/post-D1 routes and D2 with pinned teams, explicit
native 0x7e key-route/carry checks, before model/route/data cutover. No candidate promotion or main activation.

Retained Goriya was also remeasured on the migrated state cohort: 181/240, −0.86 real/−0.81 total hearts,
worst quiet 1,843. D1 records stay unchanged; D2 0x1e is 47/60, −0.82 hearts, and 0x5e 50/60, −0.85. This
measures the new state cohort, not an isolated causal key-count effect. Keep its model unchanged for the
requested Rope retrain; new D2 curricula are being refreshed for later bundled retraining.

Rope trajectory refresh is complete for all 18 affected entry states: 12 chain-0/1 paths use 377,375 exploratory
actions, six chain-2 paths 138,681. All exact native replays succeed; terminal snapshots are not curriculum
starts. All 18 tracks / 2,174 snapshots also pass actual curriculum-stage reset and exact last-action native
exit smoke. Temporary completed entry-state installation/build/audit scripts are removed; native payloads
are committed and audit/inventory JSON is copied into the r34 evidence root.

Goriya refresh completed 6/6 native replays: 318,918 exploratory actions for chains 0/1, 158,057 for chain 2.
Default Rope curricula still use four obsolete seed labels; refresh also covers retained off-route 0x6d rather
than keeping those old paths. All three 2_6ds paths succeed in 110/55/43 actions (21,662 exploratory actions).
The replacement therefore covers all 21 configured Rope starts and the six affected D2 Goriya starts.

**Native curriculum reset prerequisite found:** all 27 tracks / 3,303 raw snapshots are alive playable mode 5,
but ordinary GameSession.load turns Goriya `2_1es_1/0100.state` into mode 17/death. Raw frame: one heart,
x=120/y=93; exact restore plus the recorded MOVE W 4 survives at one heart. The loader advances two idle
frames before the first choice (gain_control's initial step plus skip_uncontrollable's final step). Do not
filter this valid snapshot or suppress the failure. Preserve controllable curriculum boundaries, while keeping
normal pre-entry loading unchanged; fix after r34's frozen training/evaluation. A deterministic half-heart
stone/shield-turn regression also reproduces: old load dies, immediate MOVE E 4 blocks once and survives.
The permanent native regression now fails before the fix (health 0 rather than 0.5). The full raw/load audit
finds exactly one fatal boundary among 3,303, not a failed exploration or bad entry state.

Both completed 40-seed teams under the new route still bypass 0x7e entirely; all 400 runs have no execution errors:

| Segment | Old R28 Dodongo | R34 Dodongo | East-prefix runs, either team | Requested-key arrivals, either team |
|---|---:|---:|---:|---:|
| Full main | 2/40 | 1/40 | 0/40 | 0/40 |
| Full fairy | 4/40 | 3/40 | 0/40 | 0/40 |
| Post-D1 main | 4/40 | 3/40 | 0/40 | 0/40 |
| Post-D1 fairy | 2/40 | 2/40 | 0/40 | 0/40 |
| D2 | 14/40 | 10/40 | 0/40 | 0/40 |

Requested-key arrivals require the first D2 rooms 7d→7e→6e, native 7e pickup flag and added key at 6e, and
that key still carried into 0e. Native per-visit inventory is recorded; all-three-key carry is separate.
`models/r34-east-key-evidence/chains/{old,new}-0/` contains all records. Candidate selection used an immutable
detached ab9bda5 worktree so the later curriculum fix could not alter its runtime; that temporary checkout is retired.

R34 completed all 490 updates / 2,007,040 added actions (3,869.9 training seconds), no early stop. Its common
entry evaluation is not a route success: the entrance is still 0/60 and other rooms regress. No promotion,
no activation and no training-code treatment retained from this failed continuation. The requested route/data
preparation stays isolated for the next attempt, using the verified backward curriculum from the same r28 start.

| Checkpoint | Successes | Real hearts/episode | Total hearts/episode | 7d east exits | 3e successes | 7e successes |
|---|---:|---:|---:|---:|---:|---:|
| Old r28-d2 | 335/420 | −0.338 | −0.263 | 0/60 | 54/60 | 58/60 |
| r34-east-key | 319/420 | −0.338 | −0.290 | 0/60 | 46/60 | 56/60 |

## 18. Preserve controllable curriculum action boundaries

`GameSession.load(..., controllable=True)` preserves a saved action boundary instead of running idle
gain-control frames. Curriculum `ZeldaEnv.reset` selects it; ordinary room-entry loads, spawn/RAM randomization,
fresh-room handling and all non-curriculum callers remain unchanged. A saved cave sets the executor's existing
in-cave flag, preventing an extra entry cooldown. No observation/action schema change or input-specific filter.

The native half-heart/approaching-stone regression fails before at reset health 0; after the fix the policy
gets its first legal shield turn and survives/moves at half a heart. Full suite: **471 passed in 118.11 seconds**.
All 21 Rope tracks / 2,382 snapshots and six D2 Goriya tracks / 921 snapshots pass family, playable-mode,
four-heart maximum, actual curriculum-stage reset and exact last-action exit smoke. The comprehensive loaded
boundary audit is **0 dead / 3,303**, versus 1 before. Native cave load/first action matches exact restoration
(exit to OW, 103 frames), and the old D2 seed-1000 run remains byte-for-byte identical (249 actions).

CUDA CLI smoke from r28: two updates / 1,024 actions, two workers, 848,672-parameter model; curriculum success
0.818 then 1.0, all 21 expected track names present. Evidence under `models/r34-east-key-evidence/`, including
`cuda-curriculum-smoke/`; the smoke checkpoint and completed throwaway verification scripts are removed.
This fix is committed in dev, not activated in main ahead of a compatible east-first model.

## 19. Verified-curriculum east-first retrain (r35)

R34 never departed east, despite objective observations already being on. Retry from the same r28-d2 checkpoint
with the existing backward curriculum rather than hard-coding the entrance. Frozen runtime source `304fb45`,
seed 35, eight workers/CUDA, at most 490×4,096 = 2,007,040 additional actions, balanced 0.99 stop. Same schema 1,
HP mix, bombs off, ordinary reward, one_room and 3,000-action cap. Curriculum fraction 0.5, threshold 0.6; all
21 verified Rope tracks plus six refreshed D2 Goriya tracks are copied into the ignored r35 evidence root,
not read from mutable default trajectory folders. Only Rope is trained; retained Goriya is unchanged.

Checkpoint/config: `models/d2-rope/r35-east-curriculum/`. Protocol/data:
`models/r35-east-curriculum-evidence/`. Main D2 route/states/Rope-and-D2-Goriya curricula/promoted pointer remain
old until compatible native key-route behavior and matched full/post-D1 main/fairy plus D2 chains are verified. Old entry
evaluation and baseline chains remain applicable: same cohort, config outside curriculum and non-curriculum
execution, with native default-run parity proven in §18. No native step, observation or action-policy override.

R35 key-route selection measures the first D2 departure east into 7e, native pickup/key balance before the
first 6e entry, and carry into 0e. Backtracking after that first east departure is handled like the existing
runner's tagged-room recovery; strict 7d→7e→6e prefix compliance is recorded separately. The already completed
400 r34 chains also have zero east-first departures under this definition. The updated native harness is
smoked on old D2 seed 1000 (same 249-action death, no false requested-key success).

The later full-retrain input audit started read-only while r35 trained against its frozen copy. It found
route-incompatible legacy Moblin inputs; their isolated replacement is logged in §20 and does not alter the
running Rope corpus, runtime source, task YAML, model pointer or D2 activation.

R35 completed all 490 updates / 2,007,040 added actions (3,853.6 training seconds), no early stop. Its complete
entry evaluation learns the required entrance while reducing damage and lingering; no reward shaping/cutoff added.

| Checkpoint | Successes | Real hearts/episode | Total hearts/episode | 7d east exits | Worst quiet |
|---|---:|---:|---:|---:|---:|
| Old r28-d2 | 335/420 | −0.338 | −0.263 | 0/60 | 247 |
| r35-east-curriculum | 400/420 | −0.205 | −0.137 | 60/60 | 136 |

Current 60-episode rooms: 7d 60, 7e 55, 6e 59, off-route 6d 60, 4e 57, Moldorm 3e 53, 2e 56 successes.
Every one of the 131 native D2 arrivals in the frozen-team chains departs east. All 34 Dodongo arrivals carry
the requested 7e key; 33 also preserve the two optional 4e/3e keys. No execution errors across 200 new runs.

| Segment | Old shortcut arrivals | R35 requested-key arrivals | All-three-key arrivals | D2 arrivals/east departures |
|---|---:|---:|---:|---:|
| Full main | 2/40 | 1/40 | 1/40 | 12/12 |
| Full fairy | 4/40 | 3/40 | 3/40 | 12/12 |
| Post-D1 main | 4/40 | 4/40 | 4/40 | 30/30 |
| Post-D1 fairy | 2/40 | 7/40 | 6/40 | 37/37 |
| D2 | 14/40 | 19/40 | 19/40 | 40/40 |

| Segment | Old real Rope hearts/run | R35 real Rope hearts/run | Old total Rope hearts/run | R35 total Rope hearts/run | Old/R35 worst Rope quiet |
|---|---:|---:|---:|---:|---:|
| Full main | −0.050 | −0.225 | −0.050 | −0.225 | 446/86 |
| Full fairy | −0.363 | −0.350 | −0.338 | −0.350 | 447/117 |
| Post-D1 main | −0.625 | −0.450 | −0.625 | −0.450 | 111/129 |
| Post-D1 fairy | −1.225 | −1.113 | −1.225 | −1.088 | 476/402 |
| D2 | −2.138 | −1.025 | −1.925 | −0.713 | 1,496/174 |

Old arrivals omit the requested key room; they are not successes on the new task. The extra full-main room
costs health and its simple milestone count is one lower, while both post-D1 segments/D2 improve or tie and
standalone damage is lower. Selected and promoted R35 for the mandated route: it supports the native key carry
that old/R34 never did. Source cutover `47788f7` activates both routes, all audited downstream states and the
curriculum-boundary fix; canonical Rope curriculum is enabled.

Canonical default-path verification: all **471 tests passed in 111.98 seconds**. The real `triforce.train`
CLI loaded r35 with the canonical Rope YAML/default trajectory root, two workers and CUDA, completing one
512-action update. All 21 configured entry tracks appear in `curriculum.json`; curriculum success was 1.0,
and five room-entry episodes completed successfully. This `/tmp` smoke checkpoint is not promoted.

Main cutover installed all 21 verified Rope tracks and the six refreshed D2 Goriya tracks, preserving retired
default inputs under `models/r35-east-curriculum-evidence/retired-default-d2-curricula/`. Native smoke uses
main imports, default YAMLs and promoted pointers, with no checkpoint/config overrides: D2 seeds 0–3 all
depart east, seeds 1/2 enter Dodongo with three keys (517/543 actions); post-D1 fairy seed 0 also carries three
keys through the milestone (1,036 actions). Post-D1 main seed 0 dies before D2. Six runs, no execution errors;
every successful run satisfies the native required-key guard. Default Rope track names exactly match all 21
configured entries. Evidence: `default-cutover.json` and `main-cutover-smoke.json` under the r35 evidence root.


## 20. Align default Moblin curriculum with post-D1 route inventory

Audit currently enabled defaults (Aquamentus, Goriya, Moblin, Zora, Wallmaster). Mode/family checks and an
upper heart bound alone pass all inputs, but exact route-container checks expose six obsolete Moblin tracks:
`0_3dn`, `0_3ds`, `0_4de`, `0_4dn`, `0_4ds`, `0_4dw`. Their 297 snapshots have three containers, no boomerang
and no triforce; the current post-D1 cohort has four containers, wooden boomerang and one triforce. Those
inputs would contaminate both the full-retrain baseline and the post-D1 usable-versus-masked comparison.

Moved these six ignored directories out of the default loader into
`models/r35-east-curriculum-evidence/retired-moblin-curricula/`, preserving their bytes/retirement manifest.
Kept valid legacy `0_3dc`: its inventory exactly matches current `0_3dn_2` (wood sword/boomerang, four hearts,
one triforce, no bow/arrows/ring/magic shield). No valid unrelated data deleted or repaired by RAM injection.

Explored the actual current entrance `0_3dn_2`, seed 35: 318 iterations/5,723 exploratory actions, successful
50-action native replay ending at three health/four containers. Installed its trajectory into default
`trajectories/ow-moblin/`; 14 Moblin tracks now pass actual loader/reset and exact last-action current-goal smoke.
No Moblin model retrained or promoted, and no running r35 Rope input changed.

| Enabled-default audit | Tracks | Snapshots | Route-container mismatches |
|---|---:|---:|---:|
| Before | 46 | 4,225 | 297 |
| After | 41 | 3,978 | 0 |

The original all-family goal smoke also identifies one old D2 Goriya trajectory (`2_1es_2`) whose last action
now ends exit-wrong rather than boss-room entry (45/46 successes). This is already superseded by the six
refreshed D2 paths from §18 (all six native current-goal replays pass); default D2 installation remains part
of the compatible route/model cutover. Evidence: route audits, inventory/goal reports, current path and
retired inputs under the r35 evidence root. Completed throwaway audit scripts are removed; runtime stays frozen.

## 21. Versioned observation full-team trial (r36, active)

Schema 2 is implemented/verified only in dev (`4d35155`): shared chase refresh/coordinates, Link/Goriya
boomerang phases/position and corrected contact-hazard states. Native before/after weapon execution is
identical across 94 frames/all five phases; old stunned/returning hazard and bait flags are corrected.
All 485 dev tests pass; CUDA scratch training and serialized/spawned evaluation complete. Main remains
schema 1 with the promoted r35 team; old checkpoints are rejected by schema 2, not relabeled.

Current comparison freezes schema 1 at `83bd2bb` (`/tmp/triforce-r36-schema1`) and schema 2 at `4d35155`
(`/tmp/triforce-r36-schema2`). Both complete ten-family teams start from scratch, seed 36, eight workers/CUDA,
490×4,096 = 2,007,040 actions/family, no early stop/load/promotion. Identical canonical task settings and
the same frozen 62-track curriculum corpus (6,541 SHA-256 recorded files). Preserve Aquamentus's 0.75 HP
floor, bombs off, ordinary rewards and null measured cutoffs. This is one matched training-seed pair,
not a seed-robustness claim. Evaluate every start at seeds 0/10,000 (20 episodes/start/block), then both
full/post-D1 routes plus D2 (40 runs/block, 80/segment/team), pinning all ten checkpoints/configs. Native
7e pickup/carry guards distinguish real requested-route arrivals from shortcut milestone counts.

First schema-1 ow-start optimization hit actual CUDA OOM before completing an update: 44 MiB requested,
26.12 MiB free, no candidate checkpoint. Failed attempt preserved separately under `schema1/failed-oom/`.
Under Lee's standing actual-OOM permission, stopped the dedicated `play-zelda.py 3x2 --games 64` process
group; observed 19,311 MiB GPU memory free afterward. Retry is fresh with unchanged settings, not loaded
partial weights. Full-team results are pending; no treatment/source/data changes mid-run.

Protocol/evidence: `models/r36-observation-full-retrain-evidence/`. Lee's new swordless-start and finite
training/player-stall requirements are recorded in `todo.md`, not implemented during this trial. Finish,
record and commit this experiment, then pause; no boomerang ablation or other next experiment before he resumes.

GUI correction during the same trial: individual CLI invocations were sequential per arm but exposed only
one-run folders. Added observational ten-specialist `TrainingPlan` views under standard main logs:
`261104_010_r36-schema1-baseline` and `261104_011_r36-schema2-bundle`. Plans reference the real absolute
event/console paths; no producer files, sources, task settings, checkpoints or running jobs change. A live
view process discovers each next actual run and updates its plan atomically. Normal `uv run training-gui.py`
now discovers both series in [currently running]. Rendered actual Qt windows offscreen for default-live and
each arm: two ten-row plans, completed ow-start/current ow-octorok/future pending rows, real curves and console
paths verified. Screenshots/status evidence: `gui-series-smoke/`; index: `gui-series.json` in the r36 root.

